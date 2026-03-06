# =============================================================
# DATABRICKS FRAUD DETECTION — COMPLETE NOTEBOOK
# =============================================================
# Run each cell sequentially in a Databricks ML GPU Runtime
# Recommended cluster: g4dn.xlarge (1 GPU) or g4dn.2xlarge
# Runtime: 13.x ML GPU
# =============================================================

# ─────────────────────────────────────────────────────────────
# CELL 1 — Install missing libraries
# ─────────────────────────────────────────────────────────────
# %pip install category-encoders imbalanced-learn shap mlflow

# ─────────────────────────────────────────────────────────────
# CELL 2 — Imports
# ─────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import tensorflow as tf
import mlflow
import mlflow.tensorflow
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PowerTransformer, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from category_encoders import TargetEncoder
from imblearn.over_sampling import SMOTE
import shap

print("TensorFlow version:", tf.__version__)
print("GPUs available:", tf.config.list_physical_devices('GPU'))

# ─────────────────────────────────────────────────────────────
# CELL 3 — Enable GPU optimisations
# mixed_float16 = use 16-bit precision on GPU → 2x faster!
# ─────────────────────────────────────────────────────────────
tf.keras.mixed_precision.set_global_policy('mixed_float16')
print("Mixed precision enabled:", tf.keras.mixed_precision.global_policy())

# ─────────────────────────────────────────────────────────────
# CELL 4 — Load data from Delta Lake
# Replace this path with your actual Delta table path
# ─────────────────────────────────────────────────────────────
# OPTION A: Load from Delta Lake (production)
# df = spark.read.format("delta") \
#           .load("/mnt/fraud/transactions") \
#           .toPandas()

# OPTION B: Sample large Delta table (avoid memory crash)
# df = spark.read.format("delta") \
#           .load("/mnt/fraud/transactions") \
#           .sample(fraction=0.1, seed=42) \
#           .toPandas()

# OPTION C: Fake data for testing (use this first!)
np.random.seed(42)
n = 10000
df = pd.DataFrame({
    'city': np.random.choice(
        ['Mumbai', 'Delhi', 'Chennai', 'Kolkata', 'TinyVillage'],
        n, p=[0.35, 0.30, 0.20, 0.14, 0.01]
    ),
    'merchant_type': np.random.choice(
        ['retail', 'online', 'atm', 'restaurant'],
        n
    ),
    'amount': np.random.exponential(scale=1000, size=n),
    'age_days': np.random.exponential(scale=365, size=n),
    'hour_of_day': np.random.randint(0, 24, n),
    'is_fraud': np.random.choice([0, 1], n, p=[0.99, 0.01])
})

print(f"Dataset shape: {df.shape}")
print(f"Fraud rate: {df['is_fraud'].mean()*100:.2f}%")
print(df.head())

# ─────────────────────────────────────────────────────────────
# CELL 5 — Feature setup
# ─────────────────────────────────────────────────────────────
HIGH_CARD_FEATURES  = ['city']                    # embedding / target encode
MED_CARD_FEATURES   = ['merchant_type']           # target encode
LOW_CARD_FEATURES   = ['hour_of_day']             # keep as number
CONTINUOUS_FEATURES = ['amount', 'age_days']      # Box-Cox + scale
TARGET              = 'is_fraud'

X = df.drop(TARGET, axis=1)
y = df[TARGET]

# ─────────────────────────────────────────────────────────────
# CELL 6 — Train/Test Split (stratified → preserve fraud ratio)
# ─────────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

print(f"Train size: {len(X_train)}, fraud: {y_train.mean()*100:.2f}%")
print(f"Test size:  {len(X_test)},  fraud: {y_test.mean()*100:.2f}%")

# ─────────────────────────────────────────────────────────────
# CELL 7 — Target Encoding (Bayesian smoothing)
# Replaces city/merchant with smoothed fraud rate
# Fit on TRAIN only → transform both → no leakage!
# ─────────────────────────────────────────────────────────────
target_encoder = TargetEncoder(smoothing=10, min_samples_leaf=1)

encode_cols = HIGH_CARD_FEATURES + MED_CARD_FEATURES

X_train[encode_cols] = target_encoder.fit_transform(X_train[encode_cols], y_train)
X_test[encode_cols]  = target_encoder.transform(X_test[encode_cols])

print("After target encoding:")
print(X_train.head())

# ─────────────────────────────────────────────────────────────
# CELL 8 — Yeo-Johnson Transform (fix skewed continuous features)
# Works on both positive and negative values
# Fit on TRAIN only → transform both
# ─────────────────────────────────────────────────────────────
power_transformer = PowerTransformer(method='yeo-johnson')

X_train[CONTINUOUS_FEATURES] = power_transformer.fit_transform(X_train[CONTINUOUS_FEATURES])
X_test[CONTINUOUS_FEATURES]  = power_transformer.transform(X_test[CONTINUOUS_FEATURES])

# ─────────────────────────────────────────────────────────────
# CELL 9 — StandardScaler (normalise all features to mean=0, std=1)
# Fit on TRAIN only → transform both
# ─────────────────────────────────────────────────────────────
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

print(f"Final feature shape: {X_train.shape}")

# ─────────────────────────────────────────────────────────────
# CELL 10 — SMOTE (balance fraud/real in training set only!)
# Creates synthetic fraud samples
# NEVER apply to test set!
# ─────────────────────────────────────────────────────────────
smote = SMOTE(random_state=42, k_neighbors=5)
X_train, y_train = smote.fit_resample(X_train, y_train)

print(f"After SMOTE — Train size: {len(X_train)}, fraud: {y_train.mean()*100:.2f}%")

# ─────────────────────────────────────────────────────────────
# CELL 11 — Build Model
# Single GPU → MirroredStrategy handles multiple GPUs if present
# ─────────────────────────────────────────────────────────────
strategy = tf.distribute.MirroredStrategy()
print(f"Number of GPUs: {strategy.num_replicas_in_sync}")

with strategy.scope():
    model = tf.keras.Sequential([

        # Layer 1
        tf.keras.layers.Dense(
            128,
            input_shape=(X_train.shape[1],),
            kernel_regularizer=tf.keras.regularizers.l2(0.01)
        ),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.Dropout(0.3),

        # Layer 2
        tf.keras.layers.Dense(
            64,
            kernel_regularizer=tf.keras.regularizers.l2(0.01)
        ),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.Dropout(0.3),

        # Layer 3
        tf.keras.layers.Dense(
            32,
            kernel_regularizer=tf.keras.regularizers.l2(0.01)
        ),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.Dropout(0.2),

        # Output — sigmoid for binary fraud probability
        tf.keras.layers.Dense(1, activation='sigmoid',
                              dtype='float32')   # float32 for mixed precision
    ])

    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=0.001,
        decay_steps=1000,
        decay_rate=0.95
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
        loss='binary_crossentropy',
        metrics=[
            'accuracy',
            tf.keras.metrics.AUC(name='auc'),
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall')
        ]
    )

model.summary()

# ─────────────────────────────────────────────────────────────
# CELL 12 — Callbacks
# ─────────────────────────────────────────────────────────────
callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor='val_auc',
        patience=20,
        restore_best_weights=True,
        mode='max'              # higher AUC = better
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,             # halve LR when stuck
        patience=10,
        min_lr=1e-6
    ),
    tf.keras.callbacks.ModelCheckpoint(
        filepath='/dbfs/models/fraud_best_model',
        monitor='val_auc',
        save_best_only=True,
        mode='max'
    )
]

# ─────────────────────────────────────────────────────────────
# CELL 13 — Train with MLflow tracking
# MLflow autolog captures everything automatically
# View in Databricks UI → Experiments
# ─────────────────────────────────────────────────────────────
mlflow.tensorflow.autolog()

with mlflow.start_run(run_name="fraud_detection_gpu_v1") as run:

    history = model.fit(
        X_train, y_train,
        epochs=500,
        batch_size=256,             # larger batch → better GPU utilisation
        validation_data=(X_test, y_test),
        callbacks=callbacks,
        verbose=1
    )

    # Log extra params manually
    mlflow.log_param("smote", True)
    mlflow.log_param("target_encoding_smoothing", 10)
    mlflow.log_param("power_transform", "yeo-johnson")
    mlflow.log_param("architecture", "128-64-32-1")

    # Log final metrics
    y_pred = (model.predict(X_test) > 0.5).astype(int)
    y_prob = model.predict(X_test)

    report = classification_report(y_test, y_pred,
                                   target_names=['Real', 'Fraud'],
                                   output_dict=True)

    mlflow.log_metric("test_f1_fraud",     report['Fraud']['f1-score'])
    mlflow.log_metric("test_recall_fraud", report['Fraud']['recall'])
    mlflow.log_metric("test_precision_fraud", report['Fraud']['precision'])

    try:
        auc = roc_auc_score(y_test, y_prob)
        mlflow.log_metric("test_roc_auc", auc)
        print(f"ROC-AUC: {auc:.3f}")
    except Exception as e:
        print(f"AUC error: {e}")

    run_id = run.info.run_id
    print(f"\nMLflow Run ID: {run_id}")

# ─────────────────────────────────────────────────────────────
# CELL 14 — Evaluation
# ─────────────────────────────────────────────────────────────
print("\n" + "="*50)
print("EVALUATION RESULTS")
print("="*50)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Real', 'Fraud']))

train_loss, train_acc = model.evaluate(X_train, y_train, verbose=0)[:2]
test_loss,  test_acc  = model.evaluate(X_test,  y_test,  verbose=0)[:2]

gap = (train_acc - test_acc) * 100
print(f"\nTraining Accuracy: {train_acc*100:.1f}%")
print(f"Testing  Accuracy: {test_acc*100:.1f}%")
print(f"Gap:               {gap:.1f}%")

if gap < 5:
    print("✅ Generalised well!")
elif gap < 15:
    print("⚠️  Slight overfitting")
else:
    print("🚨 Overfitting! Increase dropout or reduce model size")

# ─────────────────────────────────────────────────────────────
# CELL 15 — SHAP Explainability
# ─────────────────────────────────────────────────────────────
try:
    feature_names = ['city', 'merchant_type', 'amount', 'age_days', 'hour_of_day']

    background   = X_train[:100]
    explainer    = shap.DeepExplainer(model, background)
    shap_values  = explainer.shap_values(X_test[:50])

    print("\nSHAP Feature Importance:")
    mean_shap = np.abs(shap_values[0]).mean(axis=0)
    for feat, imp in sorted(zip(feature_names, mean_shap),
                             key=lambda x: x[1], reverse=True):
        bar = "█" * int(imp * 50)
        print(f"  {feat:15} {bar} {imp:.4f}")

except Exception as e:
    print(f"SHAP error: {e}")

# ─────────────────────────────────────────────────────────────
# CELL 16 — Register Model in MLflow Model Registry
# This makes it available for serving!
# ─────────────────────────────────────────────────────────────
model_name = "FraudDetectionNN"

registered_model = mlflow.register_model(
    model_uri=f"runs:/{run_id}/model",
    name=model_name
)

print(f"Model registered: {model_name}")
print(f"Version: {registered_model.version}")

# Transition to Production stage
client = mlflow.tracking.MlflowClient()
client.transition_model_version_stage(
    name=model_name,
    version=registered_model.version,
    stage="Production"
)
print(f"Model transitioned to Production ✅")

# ─────────────────────────────────────────────────────────────
# CELL 17 — Batch Scoring with Pandas UDF
# Score millions of new transactions on Spark cluster!
# ─────────────────────────────────────────────────────────────
import pandas as pd
from pyspark.sql.functions import pandas_udf, struct, col
from pyspark.sql.types import FloatType
import pickle, base64

# Serialise preprocessors to pass into UDF
preprocessors = {
    'target_encoder': target_encoder,
    'power_transformer': power_transformer,
    'scaler': scaler
}

preprocessors_b64 = base64.b64encode(
    pickle.dumps(preprocessors)
).decode('utf-8')

@pandas_udf(FloatType())
def predict_fraud_udf(city: pd.Series,
                      merchant_type: pd.Series,
                      amount: pd.Series,
                      age_days: pd.Series,
                      hour_of_day: pd.Series) -> pd.Series:

    # Load model fresh in each worker
    model = mlflow.tensorflow.load_model(f"models:/{model_name}/Production")

    # Reconstruct preprocessors
    preprocessors = pickle.loads(base64.b64decode(preprocessors_b64))
    te = preprocessors['target_encoder']
    pt = preprocessors['power_transformer']
    sc = preprocessors['scaler']

    # Build dataframe
    X = pd.DataFrame({
        'city': city,
        'merchant_type': merchant_type,
        'amount': amount,
        'age_days': age_days,
        'hour_of_day': hour_of_day
    })

    # Apply same preprocessing pipeline
    X[['city', 'merchant_type']] = te.transform(X[['city', 'merchant_type']])
    X[['amount', 'age_days']]    = pt.transform(X[['amount', 'age_days']])
    X                             = sc.transform(X)

    preds = model.predict(X)
    return pd.Series(preds.flatten())

# Apply to new Spark DataFrame
new_transactions = spark.createDataFrame(df.head(100))

scored = new_transactions.withColumn(
    "fraud_probability",
    predict_fraud_udf(
        col("city"),
        col("merchant_type"),
        col("amount"),
        col("age_days"),
        col("hour_of_day")
    )
).withColumn(
    "is_fraud_predicted",
    (col("fraud_probability") > 0.5).cast("int")
)

scored.select("city", "amount", "fraud_probability", "is_fraud_predicted").show(10)

# Write predictions back to Delta Lake
scored.write \
    .format("delta") \
    .mode("overwrite") \
    .save("/mnt/fraud/predictions")

print("Predictions written to Delta Lake ✅")

# ─────────────────────────────────────────────────────────────
# CELL 18 — Serve as REST API Endpoint
# Databricks Model Serving → real-time predictions!
# ─────────────────────────────────────────────────────────────

# NOTE: Run this in a separate notebook cell or via Databricks UI
# Go to: Databricks UI → Serving → Create Endpoint

# OR programmatically via Databricks API:
import requests, json

DATABRICKS_HOST  = "https://<your-workspace>.azuredatabricks.net"
DATABRICKS_TOKEN = dbutils.secrets.get(scope="fraud", key="db_token")

endpoint_config = {
    "name": "fraud-detection-endpoint",
    "config": {
        "served_models": [{
            "model_name": model_name,
            "model_version": registered_model.version,
            "workload_size": "Small",          # Small/Medium/Large
            "scale_to_zero_enabled": True      # saves cost when idle
        }]
    }
}

response = requests.post(
    f"{DATABRICKS_HOST}/api/2.0/serving-endpoints",
    headers={"Authorization": f"Bearer {DATABRICKS_TOKEN}"},
    json=endpoint_config
)

print("Endpoint creation status:", response.status_code)
print(response.json())

# ─────────────────────────────────────────────────────────────
# CELL 19 — Call the REST Endpoint (real-time prediction)
# Call from any application — FastAPI, Spring Boot, etc.
# ─────────────────────────────────────────────────────────────
endpoint_url = f"{DATABRICKS_HOST}/serving-endpoints/fraud-detection-endpoint/invocations"

# Single transaction prediction
payload = {
    "inputs": {
        "city":          ["Mumbai"],
        "merchant_type": ["online"],
        "amount":        [15000.0],
        "age_days":      [365.0],
        "hour_of_day":   [2]          # 2 AM transaction → suspicious!
    }
}

response = requests.post(
    endpoint_url,
    headers={
        "Authorization": f"Bearer {DATABRICKS_TOKEN}",
        "Content-Type": "application/json"
    },
    json=payload
)

result = response.json()
fraud_prob = result['predictions'][0]

print(f"\nTransaction: Mumbai, online, ₹15000, 2AM")
print(f"Fraud probability: {fraud_prob*100:.1f}%")
print(f"Decision: {'🚨 FRAUD' if fraud_prob > 0.5 else '✅ REAL'}")

# ─────────────────────────────────────────────────────────────
# CELL 20 — Schedule as Databricks Job (run daily)
# ─────────────────────────────────────────────────────────────

# Go to: Databricks UI → Workflows → Create Job
# Or via API:

job_config = {
    "name": "fraud-model-daily-scoring",
    "tasks": [{
        "task_key": "score_transactions",
        "notebook_task": {
            "notebook_path": "/Shared/fraud_detection",
        },
        "job_cluster_key": "fraud_cluster"
    }],
    "job_clusters": [{
        "job_cluster_key": "fraud_cluster",
        "new_cluster": {
            "spark_version": "13.3.x-gpu-ml-scala2.12",
            "node_type_id": "g4dn.xlarge",
            "num_workers": 1
        }
    }],
    "schedule": {
        "quartz_cron_expression": "0 0 1 * * ?",  # run daily at 1 AM
        "timezone_id": "Asia/Kolkata"
    }
}

response = requests.post(
    f"{DATABRICKS_HOST}/api/2.0/jobs/create",
    headers={"Authorization": f"Bearer {DATABRICKS_TOKEN}"},
    json=job_config
)

print("Job created:", response.json())
