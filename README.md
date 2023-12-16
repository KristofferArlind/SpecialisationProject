# Specialisation Project
Code used for my specialisation project in Autumn 2023

Most of the scripts and notebooks wont work without the needed data. How I obtained the different data is described in the report.

## bear_weeks.csv and bull_weeks.csv
Contains the intervals defined as bear and bull periods in the project. The dates are all mondays.

## final_features.csv
Contains the features used in training for the project. Including dtype and categorical/not categorical flags

## Scripts
The script which are run on the IDUN cluster. Data processing and model training.

## Slurms
The slurm files associated with the different scripts to run on the IDUN cluster.

## Notebooks
The jupyter notebooks used to create plots and analyze the data and results. Sometimes ran on the IDUN cluster, sometimes locally.

after_cap_stats.ipynb plots stats on the market caps and number of securities present after the market cap and volume cutoff

after_cap_mc_currency.ipynb plots market caps and number of securities in stock exchanges and currencies in 2023 after the market cap and volume cutoff

bull_bear_plots.ipynb plots the bull and bear periods in the training and test data

handle_forex.ipynb converts the forex data from the US Federal Reserve to a simpler format

results.ipynb includes all the logic and plots used to analyze results from the models

