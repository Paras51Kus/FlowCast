# FlowCast Data Quality Report

## Source datasets

FlowCast uses:

- `traffic_sensor_log.csv`
- `weather_observations.csv`
- `calendar_events.csv`

## Validation rules

### Traffic

- Timestamp must be parseable.
- Traffic volume cannot be negative.
- Average speed must be positive and within a road-traffic domain bound.
- Occupancy must be between 0 and 100%.
- Travel time must be positive.
- Accident count cannot be negative.
- Road capacity must be positive.

Invalid traffic records are quarantined instead of silently deleted.

### Weather

Weather observations are parsed using the supplied `DD/MM/YYYY` date format.
Weather observations are matched to traffic records using station ID and the
nearest observation within one hour.

### Calendar

Calendar information is joined by normalized calendar date.

## Cleaning

The cleaning pipeline:

1. removes duplicate road/timestamp observations;
2. handles invalid sensor values as missing;
3. interpolates numeric sensor values within road;
4. interpolates weather measurements within station;
5. harmonizes weather categories;
6. merges traffic, weather and calendar data.

## Feature engineering

Features include:

- lagged traffic volume;
- lagged speed, occupancy and travel time;
- rolling means and standard deviations;
- hour/day/month/week encodings;
- peak-hour flags;
- weekend/holiday/event/roadwork flags;
- rainfall and visibility flags;
- vehicle-type proportions;
- traffic-capacity ratio;
- weather × peak-hour interactions.

## Accident risk

`accident_risk = 1` when `accident_count > 0`; otherwise it is 0.

This is a supervised classification target derived directly from the supplied
traffic sensor accident observations.

## Congestion

`congestion_level` is used as supplied by the traffic sensor dataset and is
not artificially regenerated.

## Final verification

Run:

```bash
python run_pipeline.py
```

The generated reports and scoreboards should be used to replace any `TBD`
values in the final technical report.
