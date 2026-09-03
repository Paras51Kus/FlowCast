# FlowCast

Traffic forecasting and mobility analytics system based on:

- traffic sensor observations;
- weather observations;
- calendar/event information.

## Dataset-specific schema

### Traffic

`traffic_sensor_log.csv`

Important fields:

- road_id
- road_name
- latitude
- longitude
- weather_station_id
- date
- time
- traffic_volume
- vehicle_count
- vehicle_type_dist
- avg_speed
- occupancy
- congestion_level
- travel_time
- accident_count
- signal_timing
- road_capacity

### Weather

`weather_observations.csv`

- station_id
- date
- time
- weather_condition
- temperature
- rainfall
- visibility

### Calendar

`calendar_events.csv`

- date
- public_holiday
- holiday_name
- event_flag
- event_name
- roadwork_flag

## Architecture

M1 Ingestion & Validation
→ M2 Cleaning & Wrangling
→ M3 Feature Engineering
→ M4 EDA & Reporting
→ M5 Classical ML
→ M6 Deep Learning
→ M7 Streamlit Dashboard

## Setup

Python 3.11 is recommended.

```powershell
py -3.11 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Place the three supplied CSV files in:

```text
data/raw/
```

Then run:

```powershell
python run_pipeline.py
```

Launch the dashboard:

```powershell
python -m streamlit run dashboard/app.py```

Run tests:

```powershell
pytest
```

## Important modelling decisions

### Traffic volume

Regression target: `traffic_volume`.

### Congestion

Classification target: `congestion_level`, using the four labels supplied by
the sensor data:

- Free-flow
- Moderate
- Heavy
- Severe

### Accident risk

Binary target:

```text
accident_risk = 1 if accident_count > 0 else 0
```

### Confidence

The dashboard uses an empirical 90% residual interval from the held-out
regression predictions. It is not an invented percentage.

### Time-aware evaluation

Training/test separation is chronological rather than random, preventing
future traffic observations from leaking into training.

## Folder layout

```text
flowcast/
├── data/
│   ├── raw/
│   ├── interim/
│   └── processed/
├── notebooks/
├── src/
├── models/
├── reports/
├── dashboard/
├── tests/
├── run_pipeline.py
├── config.yaml
├── requirements.txt
└── README.md
```
