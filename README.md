# Earnings Predictor Model (EPM)

Implementation of enhanced earnings prediction methodology based on Herzberg, Guo, Brown (1999).

```text
.
├── data
│   ├── processed
│   │   ├── backtest_results_top100.csv
│   │   ├── backtest_results_top50.csv
│   │   ├── backtest_results.csv
│   │   ├── epm_features_top100.csv
│   │   ├── epm_features_top50.csv
│   │   ├── epm_features.csv
│   │   ├── epm_predictions_top100.csv
│   │   ├── epm_predictions_top50.csv
│   │   ├── epm_predictions.csv
│   │   ├── quintile_win_rate.png
│   │   └── sector_performance.png
│   └── raw
│       ├── earnings_data.csv
│       ├── earnings_estimates.csv
│       ├── epm_data.pkl
│       ├── epm_sp500_top100.pkl
│       └── epm_sp500_top50.pkl
├── docker-compose.yml
├── Dockerfile
├── notebooks
│   ├── 01_data_collection.ipynb
│   ├── 02_data_exloration.ipynb
│   └── 03_epm_analysis.ipynb
├── quintile_win_rate.png
├── README.md
├── requirements.txt
├── scripts
│   ├── create_plots.py
│   ├── debug_outliers.py
│   ├── run_collection.py
│   └── run_full_pipeline.py
├── sector_performance.png
├── src
│   ├── data
│   │   ├── collector.py
│   │   └── init.py
│   ├── features
│   │   ├── engineer.py
│   │   └── init.py
│   └── models
│       ├── __init__.py
│       ├── backtest.py
│       ├── epm_model.py
│       └── gradient_optimizer.py
├── structure.txt
└── temp.py
