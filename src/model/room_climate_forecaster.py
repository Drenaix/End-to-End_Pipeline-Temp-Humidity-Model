"""
Room Climate Forecaster & Feature Engineering Pipeline
------------------------------------------------------
Lädt historische Smart-Home-Sensordaten aus der TimescaleDB, reichert sie
mit Wetterdaten von Open-Meteo an und trainiert ein LightGBM-Modell zur
proaktiven Temperaturvorhersage (1 Stunde im Voraus).
"""

import os
import logging
from datetime import datetime, timezone
from typing import List, Tuple

import lightgbm as lgb
import openmeteo_requests
import pandas as pd
import polars as pl
import requests_cache
from dotenv import load_dotenv
from retry_requests import retry
from sklearn.metrics import mean_absolute_error, r2_score

# =====================================================================
# 1. KONFIGURATION & LOGGING
# =====================================================================
load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# Zugangsdaten und Parameter sicher aus der .env-Datei laden
DATABASE_URL: str = os.getenv("DATABASE_URL", "postgresql://user:pass@localhost:5432/climate_data")
LATITUDE: float = float(os.getenv("LATITUDE", "50.7753"))
LONGITUDE: float = float(os.getenv("LONGITUDE", "6.0839"))
WEATHER_PAST_DAYS: int = int(os.getenv("WEATHER_PAST_DAYS", "90"))
FORECAST_HORIZON_STEPS: int = 4  # 4 * 15min = 1 Stunde im Voraus


# =====================================================================
# 2. PHYSIKALISCHE FORMELN (POLARS EXPRESSIONS)
# =====================================================================
def calc_absolute_humidity(temp_col: str, rh_col: str, output_name: str) -> pl.Expr:
    """
    Berechnet die absolute Luftfeuchtigkeit in g/m³ als hochperformanten Polars-Ausdruck.
    Basiert auf der Magnus-Formel für den Sättigungsdampfdruck.
    """
    T = pl.col(temp_col)
    RH = pl.col(rh_col)
    
    # Sättigungsdampfdruck Es (hPa)
    Es = 6.112 * ((17.67 * T) / (243.5 + T)).exp()
    # Tatsächlicher Dampfdruck E (hPa)
    E = Es * (RH / 100.0)
    # Absolute Luftfeuchtigkeit AH (g/m³)
    AH = 216.7 * (E / (273.15 + T))
    
    return AH.alias(output_name)


# =====================================================================
# 3. DATA EXTRACTION (TIMESCALEDB & OPEN-METEO)
# =====================================================================
def load_indoor_sensor_data(db_url: str) -> pl.DataFrame:
    """Lädt die aggregierten 15-Minuten-Sensordaten und pivotiert sie im Speicher."""
    logging.info("Lade Sensordaten aus der TimescaleDB via ConnectorX...")
    query = """
        SELECT 
            bucket_15min AS timestamp,
            clean_sensor_name,
            avg_value
        FROM mart_room_climate_15min
        ORDER BY bucket_15min ASC
    """
    
    df_raw = pl.read_database_uri(query=query, uri=db_url, engine="connectorx")
    
    df_pivoted = (
        df_raw
        .pivot(values="avg_value", index="timestamp", on="clean_sensor_name")
        .with_columns(pl.col("timestamp").dt.replace_time_zone(None))
    )
    
    logging.info(f"✅ Sensordaten geladen: {len(df_pivoted)} Zeilen.")
    return df_pivoted


def load_outdoor_weather_data(lat: float, lon: float, past_days: int) -> pl.DataFrame:
    """Ruft historische Wetterdaten und Strahlungswerte von Open-Meteo ab."""
    logging.info(f"Lade Wetterdaten für Koordinaten ({lat}, {lon}) der letzten {past_days} Tage...")
    
    cache_session = requests_cache.CachedSession(".cache", expire_after=3600)
    retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
    openmeteo = openmeteo_requests.Client(session=retry_session)

    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "minutely_15": ["temperature_2m", "relative_humidity_2m", "direct_radiation"],
        "past_days": past_days,
        "forecast_days": 2,
    }
    
    response = openmeteo.weather_api(url, params=params)[0]
    minutely_15 = response.Minutely15()

    start_dt = datetime.fromtimestamp(minutely_15.Time(), tz=timezone.utc).replace(tzinfo=None)
    end_dt = datetime.fromtimestamp(minutely_15.TimeEnd(), tz=timezone.utc).replace(tzinfo=None)

    # closed="left" verhindert die zeitliche Überlappung des letzten Zeitstempels
    time_range = pl.datetime_range(
        start=start_dt, end=end_dt, interval="15m", closed="left", eager=True
    )

    df_weather = pl.DataFrame({
        "timestamp": time_range,
        "outdoor_temp": minutely_15.Variables(0).ValuesAsNumpy(),
        "outdoor_humidity": minutely_15.Variables(1).ValuesAsNumpy(),
        "solar_radiation": minutely_15.Variables(2).ValuesAsNumpy(),
    })
    
    logging.info(f"✅ Wetterdaten geladen: {len(df_weather)} Zeilen.")
    return df_weather


# =====================================================================
# 4. FEATURE ENGINEERING & DATA CLEANING
# =====================================================================
def build_ml_feature_matrix(
    df_indoor: pl.DataFrame, df_weather: pl.DataFrame
) -> Tuple[pl.DataFrame, str, str]:
    """
    Führt Joins durch, füllt Lücken via Forward-Fill (Zigbee-Fix), berechnet
    Thermodynamik-Ausdrücke, Lags und generiert das Zukunfts-Target.
    """
    logging.info("Starte Feature Engineering und Sensor-Bereinigung...")
    
    # Automatische Sensor-Erkennung
    temp_sensor = [col for col in df_indoor.columns if "temperatur" in col.lower()][0]
    hum_sensor = [col for col in df_indoor.columns if "feuchtigkeit" in col.lower()][0]
    logging.info(f"🎯 Hauptsensoren erkannt -> Temperatur: '{temp_sensor}', Feuchte: '{hum_sensor}'")

    df_ml = (
        df_indoor
        .join(df_weather, on="timestamp", how="inner")
        
        # Zigbee-Fix: Letzten bekannten Wert nach vorne auffüllen (und anfänglich nach hinten)
        .fill_null(strategy="forward")
        .fill_null(strategy="backward")
        
        # Physik & Lag-Features berechnen (Lag 4 = 1 Stunde historisch)
        .with_columns(
            calc_absolute_humidity(temp_sensor, hum_sensor, "indoor_abs_humidity"),
            calc_absolute_humidity("outdoor_temp", "outdoor_humidity", "outdoor_abs_humidity"),
            pl.col(temp_sensor).shift(4).alias("indoor_temp_lag_1h"),
            pl.col(hum_sensor).shift(4).alias("indoor_hum_lag_1h")
        )
        
        # Zukunfts-Targets (in exakt 1 Stunde = 4 * 15min in die Zukunft)
        .with_columns(
            pl.col(temp_sensor).shift(-FORECAST_HORIZON_STEPS).alias("target_temp_1h"),
            pl.col("indoor_abs_humidity").shift(-FORECAST_HORIZON_STEPS).alias("target_hum_1h")
        )
        
        # ML-Cleanup: Postgres Decimals in performante Float64 umwandeln
        .with_columns(pl.col(pl.Decimal).cast(pl.Float64))
        
        # Gezielt nur Nullwerte löschen, die durch die Lag-/Shift-Fenster entstanden sind
        .drop_nulls(subset=[temp_sensor, "target_temp_1h", "indoor_temp_lag_1h", "outdoor_temp"])
    )

    logging.info(f"✅ Feature Matrix fertiggestellt: {len(df_ml)} Zeilen bereit fürs Training.")
    return df_ml, temp_sensor, hum_sensor


# =====================================================================
# 5. MODELL-TRAINING & EVALUIERUNG
# =====================================================================
def train_and_evaluate_lightgbm(
    df_ml: pl.DataFrame, target_sensor: str, features: List[str]
) -> lgb.LGBMRegressor:
    """Trainiert den LightGBM-Regressor und gibt die Evaluierungsmetriken aus."""
    logging.info("Starte LightGBM Modell-Training (Time-Series Split: 80/20)...")
    
    split_idx = int(len(df_ml) * 0.8)
    df_train = df_ml.slice(0, split_idx)
    df_test = df_ml.slice(split_idx, len(df_ml) - split_idx)

    # Konvertierung für Scikit-Learn / LightGBM API
    X_train = df_train.select(features).to_pandas()
    X_test = df_test.select(features).to_pandas()
    y_train = df_train.select("target_temp_1h").to_pandas().values.ravel()
    y_test = df_test.select("target_temp_1h").to_pandas().values.ravel()

    # Modell-Initialisierung mit angepassten Hyperparametern für Smart-Home-Dynamik
    model = lgb.LGBMRegressor(
        n_estimators=50,
        learning_rate=0.05,
        min_child_samples=5,
        random_state=42,
        verbose=-1
    )
    model.fit(X_train, y_train)

    # Inferenz & Metrik-Berechnung
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    print("\n" + "=" * 60)
    print("🏆 ERGEBNIS DER PROAKTIVEN TEMPERATUR-VORHERSAGE (in 1h):")
    print("=" * 60)
    print(f"🌡️  Mittlerer absoluter Fehler (MAE) : {mae:.2f} °C")
    print(f"📊 Bestimmtheitsmaß (R²)            : {r2:.3f}")
    print("=" * 60 + "\n")
    
    return model


# =====================================================================
# 6. MAIN ORCHESTRATOR
# =====================================================================
def main():
    """Haupt-Pipeline für lokale Tests oder den Aufruf über CLI."""
    if not DATABASE_URL or "user:pass" in DATABASE_URL:
        logging.warning("⚠️ Keine gültige DATABASE_URL in der .env gefunden! Prüfe deine Konfiguration.")
    
    # 1. Daten laden
    df_indoor = load_indoor_sensor_data(DATABASE_URL)
    df_weather = load_outdoor_weather_data(LATITUDE, LONGITUDE, WEATHER_PAST_DAYS)
    
    # 2. Features bauen
    df_ml, temp_sensor, _ = build_ml_feature_matrix(df_indoor, df_weather)
    
    # 3. Modell trainieren & auswerten
    features = [
        temp_sensor, "indoor_abs_humidity",
        "outdoor_temp", "outdoor_abs_humidity", "solar_radiation",
        "indoor_temp_lag_1h", "indoor_hum_lag_1h"
    ]
    
    train_and_evaluate_lightgbm(df_ml, temp_sensor, features)


if __name__ == "__main__":
    main()