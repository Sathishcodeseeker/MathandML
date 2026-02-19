# Flight Congestion ML System — Complete Blueprint
### Greenfield Architecture: Raw Data → Production → Self-Healing

---

## System Overview

```
S3 / ADLS (raw data)
        ↓
Databricks Bronze → Silver → Gold
        ↓
congestion_score computed
        ↓
XGBoost Pipeline trained + logged to MLflow
        ↓
FastAPI loads pipeline at startup
        ↓
Request → fetch Gold features → predict → top 3 routes
        ↓
Every prediction logged
        ↓
Nightly: outcomes joined + metrics computed + drift checked
        ↓
Trigger retrain if needed → new model → promote → FastAPI refreshes
```

---

## Build Order

```
Week 1-2:   Bronze/Silver/Gold pipeline in Databricks
Week 3-4:   Gold feature tables + congestion_score target
Week 5-6:   Train first model + register in MLflow
Week 7-8:   FastAPI serving layer
Week 9-10:  Real-time stream ingestion (Kafka)
Week 11-12: Monitoring + drift detection + retraining
```

---

## Phase 1: Data Foundation

### Architecture

```
Raw feeds
    ↓
S3 / ADLS (raw zone)   ← dump everything as-is
    ↓
Databricks Bronze      ← ingest raw, minimal cleaning
    ↓
Databricks Silver      ← join, clean, validate
    ↓
Databricks Gold        ← feature-level aggregations, model-ready
```

### Data Sources

**Historical (batch):**

```
Historical flights     → past routes, delays, outcomes
Weather history        → conditions at airports over time
ATC / Airspace data    → sector congestion historically
Airport data           → runways, gates, capacity
```

**Real-time (stream):**

```
Live weather API       → OpenWeather / NOAA
Live flight feed       → FlightAware / AviationStack
Live ATC feed          → OpenSky Network (free)
Airport ATIS feed      → runway in use, winds, visibility
```

---

## Phase 2: Feature Engineering & Congestion Score

### Congestion Score — Your Training Target

Congestion is a weighted combination of signals normalized to 0-1:

```
0.0  → perfectly clear route
0.5  → moderate congestion
1.0  → heavily congested, avoid
```

```python
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

def compute_congestion_score(df: pd.DataFrame) -> pd.Series:

    scaler = MinMaxScaler()

    df["norm_delay"]        = scaler.fit_transform(df[["actual_delay_mins"]])
    df["norm_airspace"]     = scaler.fit_transform(df[["airspace_sector_load"]])
    df["norm_weather"]      = scaler.fit_transform(df[["weather_severity"]])
    df["norm_gate"]         = scaler.fit_transform(df[["gate_occupancy_rate"]])
    df["norm_cancellation"] = scaler.fit_transform(df[["cancellation_rate"]])

    congestion_score = (
        0.35 * df["norm_delay"]        +
        0.25 * df["norm_airspace"]     +
        0.20 * df["norm_weather"]      +
        0.12 * df["norm_gate"]         +
        0.08 * df["norm_cancellation"]
    )

    return congestion_score.round(4)
```

### Feature Groups

**Group 1: Origin Airport Features**

```
Feature                      Source              Computation
────────────────────────────────────────────────────────────────────
runway_utilization           Historical flights  flights_per_hour / capacity
avg_departure_delay          Historical flights  mean(actual_delay) per bucket
gate_occupancy_rate          Historical flights  gates_in_use / total_gates
taxi_out_time_avg            Historical flights  mean(taxi_out_mins)
cancellation_rate            Historical flights  cancelled / scheduled
weather_severity             Weather API         0=clear,1=cloud,2=rain,3=storm
visibility_score             Weather API         normalized visibility km
wind_severity                Weather API         0=calm,1=moderate,2=strong
active_departures            Historical flights  count departing same hour
```

**Group 2: Destination Airport Features**

Same as origin but for destination:

```
runway_utilization_dest, avg_arrival_delay_dest,
gate_occupancy_rate_dest, weather_severity_dest,
visibility_score_dest, wind_severity_dest, active_arrivals_dest
```

**Group 3: Route-Level Features**

```
Feature                      Source              Computation
────────────────────────────────────────────────────────────────────
route_distance_km            Static IATA table   great circle distance
historical_route_delay       Historical flights  mean(delay) for O→via→D
route_on_time_rate           Historical flights  on_time / total
airspace_sector_load         ATC / OpenSky       flights / sector capacity
route_flight_frequency       Historical flights  flights per day on route
historical_cancellation      Historical flights  cancelled / total
```

**Group 4: Flight-Level Features**

```
Feature                      Source              Computation
────────────────────────────────────────────────────────────────────
aircraft_capacity            Static ref table    seats by aircraft type
aircraft_category            Static ref table    0=regional,1=narrow,2=wide
airline_otp_rate             Historical flights  on_time / total per airline
airline_cancellation_rate    Historical flights  cancelled / total per airline
```

**Group 5: Time Features (computed inline at inference)**

```
Feature          Computation
──────────────────────────────────────────────────────
hour_of_day      ts.hour
day_of_week      ts.weekday()
month            ts.month
season           0=winter,1=spring,2=monsoon,3=autumn
is_weekend       ts.weekday() >= 5
is_holiday       India holiday calendar
hour_sin         sin(2π × hour / 24)
hour_cos         cos(2π × hour / 24)
day_sin          sin(2π × day / 7)
day_cos          cos(2π × day / 7)
```

Cyclical encoding is critical — without it, model thinks hour 23 and hour 0 are far apart when they are actually adjacent.

### Databricks Gold Layer Build

**Airport feature table:**

```python
from pyspark.sql import functions as F

airport_features = flights_df.groupBy("origin", "time_bucket").agg(
    (F.count("flight_id") / F.lit(runway_capacity)).alias("runway_utilization"),
    F.mean("actual_delay_mins").alias("avg_departure_delay"),
    (F.sum(F.when(F.col("status") == "cancelled", 1).otherwise(0)) /
     F.count("flight_id")).alias("cancellation_rate"),
    F.mean("taxi_out_mins").alias("taxi_out_time_avg"),
    F.count("flight_id").alias("active_departures")
).join(
    weather_df.groupBy("airport_code", "time_bucket").agg(
        F.mean("severity_encoded").alias("weather_severity"),
        F.mean("visibility_km").alias("visibility_score"),
        F.mean("wind_speed_encoded").alias("wind_severity")
    ),
    on=[F.col("origin") == F.col("airport_code"),
        flights_df["time_bucket"] == weather_df["time_bucket"]]
)

airport_features.write.format("delta") \
    .mode("overwrite") \
    .saveAsTable("gold.airport_features")
```

**Route feature table:**

```python
route_features = flights_df.groupBy(
    "origin", "destination", "via_airport", "time_bucket"
).agg(
    F.mean("actual_delay_mins").alias("historical_route_delay"),
    (F.sum(F.when(F.col("status") == "on_time", 1).otherwise(0)) /
     F.count("flight_id")).alias("route_on_time_rate"),
    F.count("flight_id").alias("route_flight_frequency"),
    (F.sum(F.when(F.col("status") == "cancelled", 1).otherwise(0)) /
     F.count("flight_id")).alias("historical_cancellation")
).join(
    airspace_df.groupBy("sector_id", "time_bucket").agg(
        (F.count("flight_id") / F.lit(sector_capacity)).alias("airspace_sector_load")
    ),
    on=["sector_id", "time_bucket"]
)

route_features.write.format("delta") \
    .mode("overwrite") \
    .saveAsTable("gold.route_features")
```

### Final Training Table Structure

```
origin | dest | via  | aircraft | airline | time_bucket | runway_util | avg_delay | weather | ... | congestion_score
──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
BLR    | DEL  | AMD  | B737     | AI      | morning_peak| 0.87        | 12.4      | 2       | ... | 0.72
BLR    | DEL  | BOM  | B737     | AI      | morning_peak| 0.61        | 4.1       | 2       | ... | 0.28
BLR    | DEL  | NAG  | B737     | AI      | morning_peak| 0.72        | 8.2       | 2       | ... | 0.51
```

Every row = one historical route flown. `congestion_score` = target. Everything else = features. All numbers.

### Pre-Training Validation Checks

```python
# 1. Score distribution — should be spread across 0-1
df["congestion_score"].hist()

# 2. No nulls
print(df.isnull().sum())

# 3. Feature correlation — no two features >0.99 correlated
print(df.corr())

# 4. Enough rows per route
print(df.groupby(["origin","destination","via_airport"]).count())
# need at least 100 rows per route
```

---

## Phase 3: Model Training Pipeline

### Column Definitions

```python
categorical_cols = [
    "origin", "destination", "via_airport",
    "aircraft_type", "airline_code",
    "time_bucket", "aircraft_category"
]

numerical_cols = [
    "runway_utilization", "avg_departure_delay",
    "gate_occupancy_rate", "weather_severity",
    "visibility_score", "wind_severity",
    "airspace_sector_load", "route_distance_km",
    "route_on_time_rate", "historical_route_delay",
    "route_flight_frequency", "airline_otp_rate",
    "aircraft_capacity"
]

time_cols = [
    "hour_of_day", "day_of_week", "month",
    "is_weekend", "is_holiday", "season",
    "hour_sin", "hour_cos", "day_sin", "day_cos"
]
```

### Time-Based Split

```python
df = df.sort_values("departure_time")   # always sort first
n  = len(df)

train_df = df.iloc[:int(0.70 * n)]
val_df   = df.iloc[int(0.70 * n):int(0.85 * n)]
test_df  = df.iloc[int(0.85 * n):]

X_train, y_train = train_df.drop(columns=["congestion_score","departure_time"]), train_df["congestion_score"]
X_val,   y_val   = val_df.drop(columns=["congestion_score","departure_time"]),   val_df["congestion_score"]
X_test,  y_test  = test_df.drop(columns=["congestion_score","departure_time"]),  test_df["congestion_score"]
```

### Pipeline Definition

```python
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from xgboost import XGBRegressor

preprocessor = ColumnTransformer(transformers=[
    ("cat",  OrdinalEncoder(
                handle_unknown="use_encoded_value",
                unknown_value=-1
             ), categorical_cols),
    ("num",  StandardScaler(), numerical_cols),
    ("time", StandardScaler(), time_cols)
], remainder="drop")

pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", XGBRegressor(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        objective="reg:squarederror",
        early_stopping_rounds=30,
        random_state=42,
        n_jobs=-1
    ))
])
```

### Training With MLflow

```python
import mlflow
import mlflow.sklearn
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

mlflow.set_experiment("/flights/route_congestion")

with mlflow.start_run(run_name="xgboost_v1"):

    pipeline.fit(
        X_train, y_train,
        model__eval_set=[(
            pipeline["preprocessor"].fit_transform(X_val), y_val
        )],
        model__verbose=50
    )

    val_preds  = pipeline.predict(X_val)
    test_preds = pipeline.predict(X_test)

    mlflow.log_metrics({
        "val_rmse":  mean_squared_error(y_val,  val_preds,  squared=False),
        "val_mae":   mean_absolute_error(y_val,  val_preds),
        "val_r2":    r2_score(y_val,  val_preds),
        "test_rmse": mean_squared_error(y_test, test_preds, squared=False),
        "test_mae":  mean_absolute_error(y_test, test_preds),
        "test_r2":   r2_score(y_test, test_preds)
    })

    mlflow.sklearn.log_model(
        pipeline,
        artifact_path="route_ranker_pipeline",
        registered_model_name="route_congestion_ranker"
    )
```

### Good Metric Thresholds

```
metric    threshold    meaning
──────────────────────────────────────────────────────
RMSE      < 0.15       predictions within 0.15 of true score
MAE       < 0.10       average error under 0.10
R²        > 0.75       model explains 75%+ of variance
```

If metrics are poor — problem is almost always in Phase 2, not the model.

### Model Registry

```python
from mlflow.tracking import MlflowClient

client = MlflowClient()

if test_rmse < 0.15 and test_r2 > 0.75:
    client.transition_model_version_stage(
        name="route_congestion_ranker",
        version=1,
        stage="Production"
    )
```

Registry structure:

```
route_congestion_ranker
├── v1  → test_rmse=0.11, test_r2=0.82  stage=Production  ← current
└── v2  → test_rmse=0.13, test_r2=0.79  stage=Staging
```

---

## Phase 4: FastAPI Serving Layer

### Project Structure

```
flight_api/
├── main.py
├── config.py
├── routers/
│   └── predict.py
├── services/
│   ├── feature_service.py
│   ├── model_service.py
│   └── route_service.py
├── schemas/
│   └── request.py
└── requirements.txt
```

### config.py

```python
from pydantic import BaseSettings

class Settings(BaseSettings):
    databricks_host:      str
    databricks_token:     str
    databricks_http_path: str
    mlflow_tracking_uri:  str
    model_name:           str = "route_congestion_ranker"
    model_stage:          str = "Production"

    class Config:
        env_file = ".env"

settings = Settings()
```

### schemas/request.py

```python
from pydantic import BaseModel, validator
from datetime import datetime
from typing import List

class PredictRequest(BaseModel):
    flight_number:  str
    origin:         str
    destination:    str
    departure_time: datetime
    aircraft_type:  str

    @validator("origin", "destination")
    def must_be_iata(cls, v):
        if len(v) != 3 or not v.isupper():
            raise ValueError("Must be a valid 3-letter IATA code")
        return v

class RouteResult(BaseModel):
    route:            str
    via:              str
    congestion_score: float
    congestion_label: str   # LOW / MEDIUM / HIGH

class PredictResponse(BaseModel):
    origin:     str
    destination: str
    top_routes: List[RouteResult]
```

### services/feature_service.py

```python
from datetime import datetime
import numpy as np
import holidays

india_holidays = holidays.India()

def to_time_bucket(hour: int) -> str:
    if   hour in [5,6,7,8,9]:      return "morning_peak"
    elif hour in [10,11,12,13,14]: return "midday"
    elif hour in [15,16,17,18,19]: return "evening_peak"
    else:                          return "off_peak"

def compute_time_features(ts: datetime) -> dict:
    hour = ts.hour
    return {
        "hour_of_day": hour,
        "day_of_week": ts.weekday(),
        "month":       ts.month,
        "is_weekend":  int(ts.weekday() >= 5),
        "is_holiday":  int(ts.date() in india_holidays),
        "season":      (0 if ts.month in [12,1,2] else
                        1 if ts.month in [3,4,5]  else
                        2 if ts.month in [6,7,8]  else 3),
        "hour_sin":    np.sin(2 * np.pi * hour / 24),
        "hour_cos":    np.cos(2 * np.pi * hour / 24),
        "day_sin":     np.sin(2 * np.pi * ts.weekday() / 7),
        "day_cos":     np.cos(2 * np.pi * ts.weekday() / 7)
    }

def fetch_airport_features(airport: str, time_bucket: str, cursor) -> dict:
    cursor.execute(f"""
        SELECT runway_utilization, avg_departure_delay,
               gate_occupancy_rate, weather_severity,
               visibility_score, wind_severity,
               active_departures, cancellation_rate
        FROM gold.airport_features
        WHERE airport_code = '{airport}'
          AND time_bucket  = '{time_bucket}'
        LIMIT 1
    """)
    row = cursor.fetchone()
    if not row:
        raise ValueError(f"No Gold features for {airport} / {time_bucket}")
    return {
        "runway_utilization":  row[0],
        "avg_departure_delay": row[1],
        "gate_occupancy_rate": row[2],
        "weather_severity":    row[3],
        "visibility_score":    row[4],
        "wind_severity":       row[5],
        "active_departures":   row[6],
        "cancellation_rate":   row[7]
    }

def fetch_route_features(origin, destination, via, time_bucket, cursor) -> dict:
    cursor.execute(f"""
        SELECT route_distance_km, airspace_sector_load,
               historical_route_delay, route_on_time_rate,
               route_flight_frequency
        FROM gold.route_features
        WHERE origin      = '{origin}'
          AND destination = '{destination}'
          AND via_airport = '{via}'
          AND time_bucket = '{time_bucket}'
        LIMIT 1
    """)
    row = cursor.fetchone()
    if not row:
        raise ValueError(f"No route features for {origin}→{via}→{destination}")
    return {
        "route_distance_km":     row[0],
        "airspace_sector_load":  row[1],
        "historical_route_delay":row[2],
        "route_on_time_rate":    row[3],
        "route_flight_frequency":row[4]
    }
```

### services/model_service.py

```python
import mlflow.sklearn
import threading
import time
import pandas as pd
from config import settings

pipeline      = None
model_version = None

def load_model():
    global pipeline, model_version
    pipeline      = mlflow.sklearn.load_model(
                        f"models:/{settings.model_name}/{settings.model_stage}"
                    )
    model_version = settings.model_stage
    print(f"Model loaded: {model_version}")

def refresh_model_periodically(interval_seconds: int = 300):
    while True:
        time.sleep(interval_seconds)
        try:
            load_model()
        except Exception as e:
            print(f"Model refresh failed: {e}")

def predict_congestion_scores(feature_rows: list) -> list:
    feature_df = pd.DataFrame(feature_rows)
    return pipeline.predict(feature_df).tolist()

def score_to_label(score: float) -> str:
    if   score < 0.33: return "LOW"
    elif score < 0.66: return "MEDIUM"
    else:              return "HIGH"

# Load on startup + start background refresh
load_model()
threading.Thread(target=refresh_model_periodically, daemon=True).start()
```

### routers/predict.py

```python
from fastapi import APIRouter, HTTPException
from databricks import sql
from schemas.request import PredictRequest, PredictResponse, RouteResult
from services.feature_service import (
    to_time_bucket, compute_time_features,
    fetch_airport_features, fetch_route_features
)
from services.model_service import predict_congestion_scores, score_to_label
from config import settings

router = APIRouter()

@router.post("/predict/top-routes", response_model=PredictResponse)
def predict_top_routes(request: PredictRequest):

    ts          = request.departure_time
    time_bucket = to_time_bucket(ts.hour)
    origin      = request.origin
    destination = request.destination
    airline     = request.flight_number[:2]

    conn = sql.connect(
        server_hostname = settings.databricks_host,
        http_path       = settings.databricks_http_path,
        access_token    = settings.databricks_token
    )
    cursor = conn.cursor()

    try:
        # Fetch shared features once
        origin_feats = fetch_airport_features(origin, time_bucket, cursor)
        dest_feats   = fetch_airport_features(destination, time_bucket, cursor)
        time_feats   = compute_time_features(ts)

        # Get all possible via airports
        cursor.execute(f"""
            SELECT DISTINCT via_airport FROM gold.route_features
            WHERE origin='{origin}' AND destination='{destination}'
            ORDER BY route_flight_frequency DESC
        """)
        possible_vias = [row[0] for row in cursor.fetchall()]

        if not possible_vias:
            raise HTTPException(404, f"No routes found for {origin}→{destination}")

        # Build one feature row per route
        feature_rows, valid_vias = [], []

        for via in possible_vias:
            try:
                route_feats = fetch_route_features(origin, destination, via, time_bucket, cursor)
                feature_rows.append({
                    "origin":            origin,
                    "destination":       destination,
                    "via_airport":       via,
                    "aircraft_type":     request.aircraft_type,
                    "airline_code":      airline,
                    "time_bucket":       time_bucket,
                    **origin_feats,
                    **route_feats,
                    "runway_util_dest":  dest_feats["runway_utilization"],
                    "avg_delay_dest":    dest_feats["avg_departure_delay"],
                    "weather_dest":      dest_feats["weather_severity"],
                    **time_feats
                })
                valid_vias.append(via)
            except ValueError:
                continue

        # Predict all routes at once
        scores  = predict_congestion_scores(feature_rows)
        results = [
            RouteResult(
                route            = f"{origin} → {via} → {destination}",
                via              = via,
                congestion_score = round(scores[i], 3),
                congestion_label = score_to_label(scores[i])
            )
            for i, via in enumerate(valid_vias)
        ]

        # Sort ascending — lowest congestion = best
        results = sorted(results, key=lambda x: x.congestion_score)

        return PredictResponse(
            origin=origin, destination=destination, top_routes=results[:3]
        )

    finally:
        conn.close()
```

### Sample Response

```json
{
  "origin": "BLR",
  "destination": "DEL",
  "top_routes": [
    {"route": "BLR → BOM → DEL", "via": "BOM", "congestion_score": 0.28, "congestion_label": "LOW"},
    {"route": "BLR → AMD → DEL", "via": "AMD", "congestion_score": 0.41, "congestion_label": "MEDIUM"},
    {"route": "BLR → NAG → DEL", "via": "NAG", "congestion_score": 0.63, "congestion_label": "HIGH"}
  ]
}
```

### Install Dependencies

```bash
pip install fastapi uvicorn mlflow xgboost \
            databricks-sql-connector scikit-learn \
            pandas numpy holidays pydantic
```

---

## Phase 5: Monitoring, Drift Detection & Retraining

### Step 1: Prediction Logging Table

```sql
CREATE TABLE IF NOT EXISTS monitoring.prediction_logs (
    log_id              STRING,
    flight_number       STRING,
    origin              STRING,
    destination         STRING,
    departure_time      TIMESTAMP,
    aircraft_type       STRING,
    time_bucket         STRING,
    via_rank1           STRING,
    score_rank1         DOUBLE,
    via_rank2           STRING,
    score_rank2         DOUBLE,
    via_rank3           STRING,
    score_rank3         DOUBLE,
    model_version       STRING,
    predicted_at        TIMESTAMP,
    actual_via_chosen   STRING,
    actual_delay_mins   DOUBLE,
    customer_accepted   BOOLEAN,
    outcome_recorded_at TIMESTAMP
)
USING DELTA
PARTITIONED BY (DATE(predicted_at))
```

### Step 2: Nightly Outcome Join

```python
def update_outcomes():
    spark.sql("""
        MERGE INTO monitoring.prediction_logs logs
        USING (
            SELECT flight_number, departure_time,
                   actual_via_airport, actual_delay_mins
            FROM silver.completed_flights
            WHERE departure_time >= current_date - 2
        ) outcomes
        ON logs.flight_number  = outcomes.flight_number
       AND logs.departure_time = outcomes.departure_time
       AND logs.actual_via_chosen IS NULL
        WHEN MATCHED THEN UPDATE SET
            logs.actual_via_chosen = outcomes.actual_via_airport,
            logs.actual_delay_mins = outcomes.actual_delay_mins,
            logs.customer_accepted = (logs.via_rank1 = outcomes.actual_via_airport)
    """)
```

### Step 3: Performance Metrics

```python
from sklearn.metrics import mean_absolute_error

def compute_performance_metrics(days_back: int = 7):

    df = spark.sql(f"""
        SELECT score_rank1, actual_delay_mins, customer_accepted
        FROM monitoring.prediction_logs
        WHERE predicted_at >= current_date - {days_back}
          AND actual_delay_mins IS NOT NULL
    """).toPandas()

    mae        = mean_absolute_error(df["actual_delay_mins"], df["score_rank1"] * 60)
    acceptance = df["customer_accepted"].mean()

    baseline_mae        = 8.2
    baseline_acceptance = 0.72

    if mae > baseline_mae * 1.3:
        trigger_alert("MAE degraded >30%", mae, baseline_mae)

    if acceptance < baseline_acceptance * 0.85:
        trigger_alert("Acceptance rate dropped >15%", acceptance, baseline_acceptance)

    return {"mae": mae, "acceptance": acceptance}
```

### Step 4: Drift Detection

```python
from scipy.stats import ks_2samp, chisquare
import numpy as np

def compute_psi(reference, production, bins=10):
    ref_counts,  edges = np.histogram(reference, bins=bins)
    prod_counts, _     = np.histogram(production, bins=edges)
    ref_pct  = np.where(ref_counts  / len(reference)  == 0, 0.0001, ref_counts  / len(reference))
    prod_pct = np.where(prod_counts / len(production) == 0, 0.0001, prod_counts / len(production))
    return float(np.sum((prod_pct - ref_pct) * np.log(prod_pct / ref_pct)))

def run_drift_detection():

    reference  = spark.sql("SELECT * FROM gold.training_features WHERE departure_time < '2024-01-01'").toPandas()
    production = spark.sql("""
        SELECT a.runway_utilization, a.avg_departure_delay,
               a.weather_severity,   a.airspace_sector_load
        FROM monitoring.prediction_logs p
        JOIN gold.airport_features a
          ON a.airport_code = p.origin AND a.time_bucket = p.time_bucket
        WHERE p.predicted_at >= current_date - 7
    """).toPandas()

    drift_report = {}

    for feature in ["runway_utilization","avg_departure_delay","weather_severity","airspace_sector_load"]:
        psi          = compute_psi(reference[feature], production[feature])
        _, p_val     = ks_2samp(reference[feature], production[feature])
        status       = "OK" if psi < 0.1 else "WARNING" if psi < 0.2 else "ALERT"
        drift_report[feature] = {"psi": round(psi,4), "ks_p": round(p_val,4), "status": status}

    for feature in ["time_bucket","origin","destination"]:
        ref_counts  = reference[feature].value_counts()
        prod_counts = production[feature].value_counts().reindex(ref_counts.index, fill_value=0)
        _, p_val    = chisquare(prod_counts, ref_counts)
        drift_report[feature] = {"chi2_p": round(p_val,4), "status": "OK" if p_val > 0.05 else "ALERT"}

    return drift_report
```

**PSI Thresholds:**

```
PSI < 0.1   → no drift        ✅
PSI 0.1-0.2 → moderate drift  ⚠️
PSI > 0.2   → significant     🔴
```

### Step 5: Retrain Decision Logic

```python
def should_retrain(drift_report, performance_metrics) -> tuple:

    alert_count         = sum(1 for v in drift_report.values() if v.get("status") == "ALERT")
    mae_degraded        = performance_metrics["mae"]        > baseline_mae * 1.3
    acceptance_degraded = performance_metrics["acceptance"] < baseline_acceptance * 0.85

    if alert_count >= 3 and mae_degraded:
        return True, "Feature drift + performance degradation confirmed"
    if mae_degraded and acceptance_degraded:
        return True, "Performance degradation on both metrics"
    if alert_count >= 5:
        return True, "Widespread feature drift — preemptive retrain"

    return False, "Model healthy"
```

### Step 6: Automated Retraining

```python
def retrain_pipeline():

    df = spark.sql("""
        SELECT * FROM gold.training_features
        WHERE departure_time >= current_date - 180
        ORDER BY departure_time
    """).toPandas()

    df["congestion_score"] = compute_congestion_score(df)

    n       = len(df)
    X_train = df.iloc[:int(0.70*n)].drop(columns=["congestion_score","departure_time"])
    y_train = df.iloc[:int(0.70*n)]["congestion_score"]
    X_test  = df.iloc[int(0.85*n):].drop(columns=["congestion_score","departure_time"])
    y_test  = df.iloc[int(0.85*n):]["congestion_score"]

    pipeline.fit(X_train, y_train)
    test_rmse = mean_squared_error(y_test, pipeline.predict(X_test), squared=False)

    with mlflow.start_run(run_name=f"retrain_{pd.Timestamp.now().date()}"):
        mlflow.log_metric("test_rmse", test_rmse)
        mlflow.sklearn.log_model(pipeline, "route_ranker_pipeline",
                                 registered_model_name="route_congestion_ranker")

        if test_rmse < get_current_production_rmse():
            promote_to_production()
            print(f"New model promoted — RMSE improved to {test_rmse:.4f}")
        else:
            print(f"New model not promoted — RMSE worse at {test_rmse:.4f}")
```

### The Full Monitoring Loop

```
Every prediction
        ↓
Logged to monitoring.prediction_logs

Every night
        ↓
Outcomes joined from completed flights
        ↓
Performance metrics: MAE, acceptance rate
        ↓
Drift detection: PSI, KS test, Chi-square
        ↓
Decision: retrain?
        ↓
NO  → sleep, check tomorrow
YES → retrain on last 6 months
        ↓
New model better than current?
        ↓
NO  → keep current, alert team
YES → promote to Production
        ↓
FastAPI picks up new model within 5 mins
```

---

## Key Rules — Never Break These

```
1. Always time-based split — never random for time series data
2. Never touch test set more than once
3. Encoding maps must be identical between training and inference
4. Always version models — never overwrite
5. Always validate data quality before writing to Gold
6. Always log every prediction — you need it for monitoring
7. Pipeline bundles encoding + model — never encode separately
8. Point-in-time correctness — only use features known at prediction time
```

---

*Blueprint covers the complete greenfield ML lifecycle for flight congestion prediction: Data Foundation → Feature Engineering → Model Training → Serving → Monitoring & Retraining.*
