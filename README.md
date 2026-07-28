# 🌡️ Proaktive Raumklima- & Schimmel-Prävention (End-to-End ML Pipeline)

Anstatt im Smart Home reaktiv auf hohe Luftfeuchtigkeit zu reagieren (wenn es bereits zu spät ist), sagt dieses End-to-End-Projekt das Raumklima **proaktiv für die nächste Stunde voraus**.

Durch die Kombination von historischen IoT-Sensordaten (TimescaleDB), externen Wetterprognosen (Open-Meteo API) und physikalischen Berechnungen liefert ein **LightGBM-Modell** präzise Lüftungsempfehlungen zur Schimmelprävention – insbesondere für kritische Räume wie kalte Keller im Sommer.

---

## 🎯 Funktionalitäten

1. **Automatisierte Wetter-Ingestion:** Regelmäßiges Abrufen von Wetterprognosen (Temperatur, relative Feuchte, Sonnenstrahlung) über die Open-Meteo API und Verknüpfung mit internen Smart-Home-Daten.
2. **Physikalisches Feature Engineering:** Berechnung der **absoluten Luftfeuchtigkeit ($g/m^3$)** mittels der Magnus-Formel für Sättigungsdampfdruck. Dies ist entscheidend, um den tatsächlichen Wassergehalt der Luft physikalisch korrekt zwischen Innen- und Außenbereich zu vergleichen.
3. **ML-gestützte Zeitreihen-Prognose:** Ein LightGBM-Regressor prognostiziert die Temperaturentwicklung und Feuchtigkeitsdrift des Raumes für den Horizont von $+1$ Stunde ohne Lüftung.
4. **Intelligente Lüftungs-Empfehlungen:**
   - _Schimmel-Schutz:_ Warnung und Lüftungsempfehlung, sobald die prognostizierte Feuchtigkeit kritische Grenzwerte überschreitet.
   - _Sommer-Kondensationsschutz:_ Automatische Sperre der Lüftungsempfehlung für kalte Keller- oder Erdgeschossräume, wenn warme, feuchte Außenluft an kalten Wänden kondensieren würde (Taupunkt-Unterschreitung).

---

## 🏗️ System-Architektur & Tech Stack

Das Projekt setzt auf moderne Data-Engineering- und MLOps-Standards:

- **Event Broker & Ingestion:** [Redpanda](https://redpanda.com/) & Redpanda Connect für hochperformantes Streaming der Sensor-Events aus dem Smart Home.
- **Database & Storage:** [TimescaleDB](https://www.timescale.com/) (PostgreSQL-Erweiterung) für die Speicherung von Zeitreihendaten auf Hypertables.
- **Orchestration:** [Apache Airflow](https://airflow.apache.org/) zur zyklischen Steuerung der Data Pipeline (Ingestion -> Feature Engineering -> Training -> Vorhersage).
- **High-Performance Processing:** [Polars](https://pola.rs/) für blitzschnelles, speichereffizientes DataFrame-Handling, Window-Lags und Sensor-Bereinigungen (`Forward-Fill` für unregelmäßige Zigbee-Signale).
- **Machine Learning:** [LightGBM](https://lightgbm.readthedocs.io/) & Scikit-Learn für Gradient-Boosting-Regression unter Berücksichtigung historischer Lags und Solarstrahlung.

```text
[Smart Home Sensors] -> (MQTT/Redpanda) -> [TimescaleDB]
                                                │
[Open-Meteo API] ------------------------------►├─► [Polars Feature Engineering]
                                                │            │
                                                ▼            ▼
                                    [Airflow DAGs] ◄─ [LightGBM Model] ─► [Lüftungs-Alerts]
```
