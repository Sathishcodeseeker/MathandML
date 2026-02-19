# ML Lifecycle Guide
### From Feature Store to Production — Route Prediction Edition

---

## Roadmap

```
1. Feature Store → Train/Val/Test Split
2. Model Training & Validation
3. Packaging Weights & Serving via Inference API
4. Monitoring Predictions & Customer Satisfaction
5. Detecting Data Drift (with metrics)
6. Identifying if Drift is Causing Problems
7. Periodic Data Integration back into Feature Store
```

---

## Step 1: Feature Store → Train / Val / Test Split

### What is a Feature Store?

A database specifically designed to store ML features — cleaned, transformed, versioned. It has two sides:

- **Offline store** — historical data for training (data warehouse or parquet files)
- **Online store** — low latency lookup for real-time inference (Redis or DynamoDB)

### How to Split

```
Total Data
├── Train      (70%)  → model learns from this
├── Validation (15%)  → you tune hyperparameters here
└── Test       (15%)  → final unbiased evaluation, touch once
```

**Critical rule** — for time series or production data, never split randomly:

```
❌ Random split
[t1, t5, t2, t8...] → train
[t3, t6, t9...]     → val

✅ Time-based split
[t1 → t7] → train
[t7 → t9] → val
[t9 → t10] → test
```

Random splitting causes **data leakage** — future data bleeds into training.

### Code Sketch

```python
df = feature_store.get_offline_features(start="2023-01-01", end="2024-01-01")
df = df.sort_values("timestamp")

n = len(df)
train = df.iloc[:int(0.70*n)]
val   = df.iloc[int(0.70*n):int(0.85*n)]
test  = df.iloc[int(0.85*n):]
```

### Why Val and Test Are Separate

- **Val** = you look at it many times while tuning → gets "contaminated" by your decisions
- **Test** = you look at it exactly once at the end → gives honest performance estimate

If you tune on test, you're cheating without knowing it.

---

## Step 2: Model Training & Validation

### The Training Loop

```python
for epoch in range(num_epochs):

    # Train on training data
    model.train()
    for X_batch, y_batch in train_loader:
        y_pred = model(X_batch)
        loss = criterion(y_pred, y_batch)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    # Evaluate on validation data
    model.eval()
    with torch.no_grad():
        val_pred = model(X_val)
        val_loss = criterion(val_pred, y_val)

    print(f"Epoch {epoch} | Train Loss: {loss:.4f} | Val Loss: {val_loss:.4f}")
```

### What to Watch For

```
Epoch 1  | Train: 0.90 | Val: 0.88  ✅ both falling together
Epoch 10 | Train: 0.40 | Val: 0.39  ✅ good
Epoch 20 | Train: 0.20 | Val: 0.41  ❌ overfitting — val rising while train falls
```

The gap between train loss and val loss tells you everything:

- **Both falling** → model is learning
- **Val stops falling** → time to stop (early stopping)
- **Val rising, train falling** → overfitting, model memorizing train data

### Early Stopping

```python
best_val_loss = float('inf')
patience = 5
counter = 0

if val_loss < best_val_loss:
    best_val_loss = val_loss
    save(model)          # save best weights
    counter = 0
else:
    counter += 1
    if counter >= patience:
        print("Early stopping")
        break
```

Always save weights at the best val loss — not the last epoch.

### Validation Metrics Beyond Loss

| Problem | Metrics |
|---|---|
| Regression | MAE, RMSE, R² |
| Classification | Accuracy, F1, AUC-ROC |
| Ranking | NDCG, MRR |

Always pick metrics that reflect **business impact**, not just mathematical convenience.

### Final Test Evaluation

```python
model.load(best_weights)
test_pred = model(X_test)
test_loss = criterion(test_pred, y_test)
# this number is your honest performance estimate — never touch test again
```

---

## Step 3: Packaging Weights & Serving via Inference API

### What Are Weights at This Point?

After training you have a file — your model's learned parameters:

```
best_model.pt        ← PyTorch
best_model.pkl       ← sklearn
best_model.h5        ← Keras
best_model.onnx      ← framework-agnostic (best for production)
```

ONNX is preferred in production — train in PyTorch, export to ONNX, serve anywhere.

### Exporting Weights

```python
# Save from PyTorch
torch.save(model.state_dict(), "best_model.pt")

# Export to ONNX (recommended for serving)
torch.onnx.export(
    model,
    dummy_input,
    "best_model.onnx",
    input_names=["features"],
    output_names=["prediction"]
)
```

### Where Do Weights Live in Production?

```
Training machine
      ↓
Model Registry (MLflow, W&B, S3)   ← versioned storage
      ↓
Inference API pulls weights on startup
```

The **model registry** stores every version with metadata:

```
model_registry/
├── v1.0  → trained 2024-01-01, val_rmse=0.42, deployed=false
├── v1.1  → trained 2024-02-01, val_rmse=0.38, deployed=true  ← current
└── v1.2  → trained 2024-03-01, val_rmse=0.36, deployed=false ← candidate
```

Never overwrite. Always version. This lets you roll back instantly.

### The Inference API

```python
from fastapi import FastAPI
import onnxruntime as ort
import numpy as np

app = FastAPI()

# Load weights once at startup
session = ort.InferenceSession("best_model.onnx")

@app.post("/predict")
def predict(features: dict):

    # 1. Get features from online feature store
    X = feature_store.get_online_features(features["customer_id"])

    # 2. Run inference
    X_array = np.array(X).reshape(1, -1).astype(np.float32)
    prediction = session.run(None, {"features": X_array})[0]

    # 3. Log the prediction (critical for monitoring)
    log_prediction(
        customer_id=features["customer_id"],
        input=X,
        prediction=prediction,
        timestamp=now()
    )

    return {"prediction": float(prediction)}
```

### The Prediction Log — Critical

Every prediction gets stored with:

```json
{
  "customer_id": 123,
  "input_features": [...],
  "prediction": 42.3,
  "timestamp": "2024-03-01T14:22:00",
  "model_version": "v1.1"
}
```

This log enables everything in later steps — drift detection, satisfaction monitoring, retraining. Without it you're blind.

### The Full Flow So Far

```
Feature Store (offline) → Train → Validate → Save weights
                                                    ↓
Customer request → Feature Store (online) → Inference API → Prediction
                                                    ↓
                                            Prediction Log
```

---

## Step 4: Monitoring Predictions & Customer Satisfaction

### The Core Problem

In training you had `X` and `y` — you always knew the true answer. In production you only have `X`. The true `y` doesn't exist yet, or takes time to arrive.

```
Training:   X → model → ŷ,  you know y  ✅
Production: X → model → ŷ,  y = ???     ❓
```

### For Route Prediction Specifically

You predict a route. The true y arrives when:

```
Prediction:  "take route A"
True y:      customer took route B       → ignored suggestion
             customer took A, arrived late → prediction was off
             customer took A, on time    ✅ correct
```

Your label latency is very short — route outcomes arrive in minutes to hours.

### Your Prediction Log Should Look Like

```json
{
  "customer_id": "c123",
  "timestamp": "2024-03-01T08:22:00",
  "origin": [lat, lng],
  "destination": [lat, lng],
  "predicted_route": ["A", "B", "C"],
  "predicted_eta_mins": 24,
  "features": {
    "traffic_density": 0.7,
    "time_of_day": "morning_peak",
    "weather": "clear",
    "road_incidents": 0
  },
  "model_version": "v1.1"
}
```

Then when trip ends, append:

```json
{
  "actual_route_taken": ["A", "B", "D"],
  "actual_eta_mins": 31,
  "customer_accepted": false,
  "rerouted": true
}
```

### Key Metrics for Route Prediction

| Metric | What It Tells You |
|---|---|
| Route acceptance rate | Are customers trusting predictions? |
| ETA accuracy (MAE) | How wrong are arrival time estimates? |
| Reroute rate | How often does real-time override model? |
| Route optimality score | Was suggested route actually fastest? |

### Two Types of Dissatisfaction Signals

**Explicit** — user tells you directly: thumbs down, support ticket, rating

**Implicit** — behavior tells you: user ignores suggestion, user reroutes mid-journey, engagement drops

### Computing Live Metrics

```python
# Join predictions with actual outcomes
df = prediction_log.join(actual_outcomes, on=["customer_id", "timestamp"])

current_mae  = mean_absolute_error(df["prediction"], df["actual"])
baseline_mae = 4.2   # from test evaluation

if current_mae > baseline_mae * 1.2:   # 20% degradation threshold
    alert("Model performance degrading!")
```

### Monitoring Dashboard

```
metric          baseline    current     status
─────────────────────────────────────────────
MAE             4.2         4.4         ✅ ok
RMSE            6.1         7.8         ⚠️  warning
Error rate      2.1%        5.3%        🔴 alert
Prediction bias +0.1        +2.4        🔴 systematic overestimate
```

### Proxy Signals (When Labels Take Too Long)

```
Real signal (slow):   actual outcome vs prediction
Proxy signal (fast):
  - input feature distribution shifting
  - prediction distribution shifting
  - user engagement dropping
  - prediction confidence dropping
```

Proxy signals fire early. Real signals confirm later.

---

## Step 5: Detecting Data Drift

### What Is Data Drift?

Your model was trained on historical data. The world changes — new roads, seasonal traffic, construction. Input features in production start looking different from training.

```
Training data:    traffic_density mean=0.4  (trained in winter)
Production data:  traffic_density mean=0.8  (summer, more cars)

Model sees inputs it never really learned from → predictions degrade
```

### Two Types of Drift

**Feature Drift (covariate shift)** — input X distribution changes. "Traffic patterns look different than training."

**Concept Drift** — relationship between X and y changes. "Same traffic density now means different travel time because a new road opened."

Feature drift is easier to detect. Concept drift is sneakier.

### How to Detect It — Reference vs Production Window

```
Reference window  = your training data distribution  (ground truth)
Production window = recent incoming features          (what model sees now)
```

If these diverge → drift detected.

### Metric 1: PSI (Population Stability Index)

Most widely used in production. Measures how much a feature distribution has shifted:

```
PSI = Σ (Actual_i - Expected_i) × ln(Actual_i / Expected_i)
```

```
PSI < 0.1   → no significant drift    ✅
PSI 0.1-0.2 → moderate drift          ⚠️ monitor closely
PSI > 0.2   → significant drift       🔴 investigate
```

```python
def compute_psi(reference, production, bins=10):
    ref_counts, edges = np.histogram(reference, bins=bins)
    prod_counts, _    = np.histogram(production, bins=edges)

    ref_pct  = ref_counts / len(reference)
    prod_pct = prod_counts / len(production)

    ref_pct  = np.where(ref_pct == 0, 0.0001, ref_pct)
    prod_pct = np.where(prod_pct == 0, 0.0001, prod_pct)

    psi = np.sum((prod_pct - ref_pct) * np.log(prod_pct / ref_pct))
    return psi

for feature in ["traffic_density", "time_of_day", "weather", "road_incidents"]:
    psi = compute_psi(train_df[feature], production_df[feature])
    print(f"{feature}: PSI={psi:.3f}")
```

### Metric 2: KS Test

Statistical test for continuous features:

```python
from scipy.stats import ks_2samp

stat, p_value = ks_2samp(train_df["traffic_density"], production_df["traffic_density"])

if p_value < 0.05:
    print("Drift detected")
```

### Metric 3: Chi-Square Test

For categorical features like weather type, time of day bucket:

```python
from scipy.stats import chisquare

stat, p_value = chisquare(prod_counts, ref_counts)

if p_value < 0.05:
    print("Categorical drift in weather feature")
```

### Metric 4: Prediction Drift

Even without labels, watch your model's output distribution:

```
Training predictions:  mean=22.4, std=6.1
Production week 1:     mean=21.8, std=8.1  ✅ stable
Production week 4:     mean=34.1, std=11.2 ❌ something changed
```

### What to Monitor for Route Prediction

| Feature | Test | Why |
|---|---|---|
| traffic_density | PSI + KS | continuous, changes seasonally |
| road_incidents | PSI + KS | continuous, event-driven spikes |
| weather | Chi-Square | categorical, seasonal drift |
| time_of_day_bucket | Chi-Square | categorical, should be stable |
| origin_region | Chi-Square | new areas being requested? |
| predicted_eta | PSI | output drift = early warning |

### Running in Real Time — Sliding Windows

```
Reference window:  all training data (fixed)
Production window: last 7 days of incoming features (sliding)

Every 24 hours:
  → compute PSI for each feature
  → compute KS test for continuous features
  → compute prediction distribution shift
  → alert if thresholds breached
```

For event-driven spikes (road incidents, accidents) use a **1-hour window** alongside weekly PSI — short-lived but severe.

---

## Step 6: Is Drift Actually Causing the Problem?

### The Key Question

```
Drift detected ≠ model is broken
Drift detected + performance drop = drift is causing the problem
```

### Three Scenarios

```
Scenario 1: Drift detected, performance stable
  → model is robust to this shift
  → monitor but don't retrain yet

Scenario 2: Drift detected, performance degrading
  → drift is likely causing the problem
  → investigate which features, retrain

Scenario 3: No drift detected, performance degrading
  → concept drift (rules changed, not inputs)
  → harder to fix, need new labels + retrain
```

### Step 1: Segment Predictions by Drifted Features

```python
df = prediction_log.join(actual_outcomes)

for weather_type in df["weather"].unique():
    segment = df[df["weather"] == weather_type]
    mae = mean_absolute_error(segment["prediction"], segment["actual"])
    print(f"weather={weather_type}: MAE={mae:.3f}")
```

Output:

```
weather=clear:        MAE=4.1   ✅ baseline performance
weather=rain:         MAE=4.3   ✅ acceptable
weather=heavy_storm:  MAE=9.8   🔴 model failing here
```

### Step 2: Feature Importance Under Drift (SHAP)

```python
import shap

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_production)

feature_importance = pd.DataFrame({
    "feature": feature_names,
    "importance": np.abs(shap_values).mean(axis=0)
}).sort_values("importance", ascending=False)
```

Output:

```
feature             importance    drifted?
──────────────────────────────────────────
traffic_density     0.42          ✅ stable
road_incidents      0.38          🔴 drifted
weather             0.31          🔴 drifted
time_of_day         0.18          ✅ stable
```

High importance + high drift = confirmed root cause.

### Step 3: Counterfactual Test

Replace drifted features with training distribution values. If performance recovers → drift is the cause:

```python
X_counterfactual = X_production.copy()
X_counterfactual["weather"] = train_df["weather"].mode()[0]
X_counterfactual["road_incidents"] = train_df["road_incidents"].mean()

mae_original       = mae(model.predict(X_production), actuals)        # 9.8
mae_counterfactual = mae(model.predict(X_counterfactual), actuals)    # 4.3 ← recovered
```

### Step 4: Detect Concept Drift via Residuals

```python
df["residual"] = df["prediction"] - df["actual"]
df.groupby("week")["residual"].mean().plot()
```

```
week 1:  residual mean = +0.2   ✅ unbiased
week 8:  residual mean = +4.1   🔴 systematic overestimate
week 12: residual mean = +7.3   🔴 getting worse
```

Systematic residual drift over time = concept drift even when input features look stable.

### Decision Tree

```
Drift detected?
├── NO  + performance degrading → concept drift → collect new labels, retrain
├── YES + performance stable    → monitor, no action yet
└── YES + performance degrading
        ↓
    Are drifted features high importance?
    ├── NO  → monitor, no action yet
    └── YES → confirmed root cause
                ↓
            New data available for those conditions?
            ├── YES → retrain with recent data
            └── NO  → collect more data first, apply fallback rules
```

---

## Step 7: Integrating New Data Back into the Feature Store

### Two Pipelines You Need

```
Pipeline 1: Batch ingestion   → runs every night / weekly
Pipeline 2: Stream ingestion  → runs in real time
```

Route prediction needs both — batch for historical enrichment, stream for live traffic and incidents.

### What New Data Looks Like

Every completed trip generates ground truth:

```json
{
  "trip_id": "t789",
  "customer_id": "c123",
  "timestamp": "2024-03-01T08:45:00",
  "actual_route_taken": ["A", "B", "D"],
  "actual_eta_mins": 31,
  "predicted_eta_mins": 24,
  "traffic_density_actual": 0.82,
  "weather_actual": "heavy_storm",
  "road_incidents_actual": 2,
  "customer_accepted": false
}
```

Every trip = one new labeled training example.

### Batch Ingestion Pipeline

```python
# Runs every night at 2am
def nightly_feature_ingestion():

    # 1. Pull completed trips from last 24 hours
    new_trips = operational_db.query("""
        SELECT * FROM trips
        WHERE completed_at >= now() - interval '24 hours'
        AND actual_route IS NOT NULL
    """)

    # 2. Feature engineering — same transformations as training
    new_trips["hour_of_day"] = new_trips["timestamp"].dt.hour
    new_trips["is_peak"]     = new_trips["hour_of_day"].isin([7,8,9,17,18,19])
    new_trips["eta_error"]   = new_trips["actual_eta"] - new_trips["predicted_eta"]

    # 3. Validate before writing
    assert new_trips["traffic_density"].between(0, 1).all()
    assert new_trips["actual_eta"].gt(0).all()

    # 4. Write to offline feature store
    feature_store.write_offline(
        data=new_trips,
        feature_group="route_features",
        timestamp_col="timestamp"
    )
```

### Stream Ingestion Pipeline

```python
# Kafka consumer — runs continuously
def stream_feature_ingestion(event):

    if event.type == "traffic_update":
        feature_store.write_online(
            key=f"traffic:{event.region}",
            value={
                "traffic_density": event.density,
                "incident_count": event.incidents,
                "updated_at": event.timestamp
            },
            ttl_seconds=300   # expire after 5 mins, always fresh
        )

    elif event.type == "weather_update":
        feature_store.write_online(
            key=f"weather:{event.region}",
            value={
                "condition": event.condition,
                "severity": event.severity,
                "updated_at": event.timestamp
            },
            ttl_seconds=600
        )
```

### Feature Versioning

Never overwrite old features. Always version:

```
feature_store/
├── route_features/
│   ├── v1/  2023-01-01 → 2023-12-31  (original training data)
│   ├── v2/  2024-01-01 → 2024-06-30  (after first drift event)
│   └── v3/  2024-07-01 → present     (current, includes storm data)
```

```python
training_data = feature_store.get_offline_features(
    feature_group="route_features",
    versions=["v1", "v2", "v3"],
    start="2023-01-01",
    end="2024-09-01"
)
```

### Data Quality Checks Before Ingestion

```python
def validate_new_data(df):

    checks = {
        "no nulls in key features":
            df[["traffic_density","actual_eta","weather"]].isnull().sum().sum() == 0,
        "eta is positive":
            df["actual_eta"].gt(0).all(),
        "traffic density in range":
            df["traffic_density"].between(0, 1).all(),
        "no future timestamps":
            df["timestamp"].lt(pd.Timestamp.now()).all(),
        "minimum sample size":
            len(df) >= 100
    }

    failed = [k for k, v in checks.items() if not v]
    if failed:
        raise ValueError(f"Data quality checks failed: {failed}")

    return True
```

If any check fails — don't ingest. Alert the data engineering team. Bad data in = bad model out.

### Point-in-Time Correctness

When building training data, use only features available **at the time of prediction** — not features learned later:

```
❌ Wrong: training row uses actual_traffic at trip END time
✅ Right: training row uses traffic_density at trip START time
```

```python
training_data = feature_store.get_point_in_time_features(
    entity_df=trips[["trip_id", "timestamp"]],
    features=["traffic_density", "weather", "road_incidents"],
    timestamp_col="timestamp"
)
```

### The Integrated Feature Store Flow

```
Completed trips (operational DB)
          ↓
    Nightly batch job
          ↓
    Data quality checks
          ↓
    Feature engineering
          ↓
    Offline feature store (versioned)
          ↓
    Available for next retrain

Live traffic / weather APIs
          ↓
    Kafka stream
          ↓
    Online feature store (TTL-based)
          ↓
    Available for real-time inference
```

---

*Guide covers Steps 1–7 of the ML lifecycle for a route prediction system. Step 8 (CI/CD retrain loop) is the natural next step.*
