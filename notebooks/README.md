The main reproducible pipeline is implemented in `run_pipeline.py`.

Recommended notebook usage:

1. `01_eda.ipynb` — call `src.eda.run_eda()`, inspect distributions,
   correlations and peak-hour behaviour.
2. `02_cleaning.ipynb` — inspect M1 quarantine files and M2 merged data.
3. `03_modelling.ipynb` — load the generated scoreboards and compare
   classical and LSTM results.

The notebooks should contain the actual figures/results generated from the
supplied datasets rather than manually entered numbers.
