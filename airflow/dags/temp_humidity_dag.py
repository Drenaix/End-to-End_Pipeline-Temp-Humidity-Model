from airflow import DAG
import io
from airflow.decorators import dag, task
from datetime import datetime, timedelta
import polars as pl
import lightgbm as lgb
import connectorx as cx
from sqlalchemy import create_engine
import pandas as pd
from dotenv import load_dotenv
import os
# ---------------------------------------------------------------------------
# KONFIGURATION & VERBINDUNG
# ---------------------------------------------------------------------------
load_dotenv()

DB_URL = os.getenv("DATABASE_URL")
if not DB_URL:
    raise ValueError("FATAL: DATABASE_URL wurde nicht in den Umgebungsvariablen gefunden!")

# 2. SQLAlchemy-URL dynamisch ableiten (ersetzt den Präfix sauber)
SQL_ALCHEMY_URL = DB_URL.replace("postgresql://", "postgresql+psycopg2://", 1)

default_args = {
    'owner': 'admin',
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

@dag(
    dag_id='predict_room_climate_1h',
    default_args=default_args,
    start_date=datetime(2026, 7, 20),
    schedule_interval='0 6,18 * * *',
    catchup=False,
    tags=['mlops', 'homelab', 'timescaledb'],
)
def room_climate_ml_pipeline():

    @task()
    def extract_and_prepare_features() -> dict:
        query = """
            SELECT
                bucket_15min AS timestamp,
                clean_sensor_name,
                avg_value
            FROM mart_room_climate_15min
            WHERE bucket_15min > NOW() - INTERVAL '30 days'
            ORDER BY bucket_15min ASC
        """
        df_raw = cx.read_sql(DB_URL, query, return_type="polars")

        if df_raw.height == 0:
            raise ValueError("Keine Daten in mart_room_climate_15min gefunden!")

        df_indoor = (
            df_raw
            .pivot(values="avg_value", index="timestamp", on="clean_sensor_name")
            .with_columns(
                pl.col("timestamp").dt.replace_time_zone(None)
            )
            .fill_null(strategy="forward")
            .fill_null(strategy="backward")
        )

        return {
            "data_json": df_indoor.write_json(),
            "latest_timestamp": str(df_indoor["timestamp"].max())
        }

    @task()
    def train_predict_and_load(payload: dict):
        df = pl.read_json(io.StringIO(payload["data_json"]))

        temp_cols = [col for col in df.columns if "temperatur" in col.lower() or "temp" in col.lower()]
        if not temp_cols:
            raise ValueError(f"Keine Temperatur-Spalte in DataFrame gefunden. Verfügbar: {df.columns}")

        target_col = temp_cols[0]

        df_ml = df.with_columns(
            pl.col(target_col).shift(1).alias("temp_lag_1")
        ).drop_nulls()

        features = ["temp_lag_1"]
        X = df_ml.select(features).to_numpy()
        y = df_ml.select(target_col).to_numpy().ravel()

        model = lgb.LGBMRegressor(n_estimators=50, verbose=-1)
        model.fit(X, y)

        latest_row = df_ml.tail(1).select(features).to_numpy()
        pred_val = float(model.predict(latest_row)[0])

        target_time = datetime.fromisoformat(payload["latest_timestamp"]) + timedelta(minutes=15)

        result_df = pd.DataFrame([{
            "created_at": datetime.now(),
            "target_timestamp": target_time,
            "room_sensor": target_col,
            "predicted_temp": round(pred_val, 2),
            "predicted_hum": 0.0
        }])

        engine = create_engine(SQL_ALCHEMY_URL)
        with engine.begin() as conn:
            result_df.to_sql("ml_room_climate_forecast", con=conn, if_exists="append", index=False)

        print(f"Erfolgreich Prognose geschrieben für {target_col}: {round(pred_val, 2)}")

    data_payload = extract_and_prepare_features()
    train_predict_and_load(data_payload)

dag = room_climate_ml_pipeline()