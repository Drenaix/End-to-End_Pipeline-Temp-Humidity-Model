# 🌡️ Proactive Indoor Climate & Mold Prevention (End-to-End ML Pipeline)

Instead of reactively responding to high humidity in a smart home (when it's already too late), this end-to-end project proactively forecasts room climate **for the next hour**.

By combining historical IoT sensor data (TimescaleDB), external weather forecasts (Open-Meteo API), and thermodynamic calculations, a **LightGBM model** delivers precise ventilation recommendations to prevent mold—especially in critical areas like cold basements during humid summer months.

---

## 🎯 Core Features & Functionality

1. **Automated Weather Ingestion:** Periodic retrieval of weather forecasts (temperature, relative humidity, direct solar radiation) via the Open-Meteo API, joined seamlessly with internal smart home telemetry.
2. **Thermodynamic Feature Engineering:** Calculation of **absolute humidity ($\text{g/m}^3$)** using the Magnus formula for saturation vapor pressure. This is essential for physically comparing moisture content between indoor and outdoor air environments.
3. **ML Time-Series Forecasting:** A LightGBM regressor predicts room temperature trends and humidity drift for a $+1$ hour horizon without active ventilation.
4. **Intelligent Ventilation Alerts:**
   - _Mold Prevention:_ Proactive alerts and ventilation recommendations whenever forecasted indoor humidity exceeds safety thresholds.
   - _Summer Condensation Shield:_ Automatic suppression of ventilation advice for cold basement or ground-floor rooms when warm, humid outdoor air would condense against cold masonry (dew point drop).

---

## 🏗️ System Architecture & Tech Stack

Built in alignment with modern Data Engineering and MLOps industry standards:

- **Event Broker & Ingestion:** [Redpanda](https://redpanda.com/) & Redpanda Connect for high-throughput streaming of IoT sensor events.
- **Database & Storage:** [TimescaleDB](https://www.timescale.com/) (PostgreSQL extension) for time-series hypertable management.
- **Orchestration:** [Apache Airflow](https://airflow.apache.org/) for scheduling cyclic pipeline runs (Ingestion -> Feature Engineering -> Training -> Inference).
- **High-Performance Processing:** [Polars](https://pola.rs/) for lightning-fast, memory-efficient DataFrame transformations, window lags, and data imputation (`Forward-Fill` for irregular Zigbee transmissions).
- **Machine Learning:** [LightGBM](https://lightgbm.readthedocs.io/) & Scikit-Learn for gradient boosting regression incorporating solar radiation and temporal lags.

```text
[Smart Home Sensors] -> (MQTT/Redpanda) -> [TimescaleDB]
                                                │
[Open-Meteo API] ------------------------------►├─► [Polars Feature Engineering]
                                                │            │
                                                ▼            ▼
                                    [Airflow DAGs] ◄─ [LightGBM Model] ─► [Ventilation Alerts]
```
