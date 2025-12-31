"""
Debug outliers and model issues
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

import pandas as pd
import numpy as np

# Load backtest results
backtest = pd.read_csv(project_root / 'data' / 'processed' / 'backtest_results_top50.csv')

print("="*60)
print("OUTLIER ANALYSIS")
print("="*60)

# Find extreme outliers
outliers = backtest[backtest['error_reduction_pct'] < -50]

print(f"\nExtreme Negative Outliers (< -50% improvement):")
print(outliers[['ticker', 'sector', 'actual_eps', 'consensus_estimate', 
                'epm_prediction', 'error_reduction_pct']].to_string())

# Consumer Defensive deep dive
print("\n" + "="*60)
print("CONSUMER DEFENSIVE SECTOR ANALYSIS")
print("="*60)

cd_sector = backtest[backtest['sector'] == 'Consumer Defensive']
print(f"\nCount: {len(cd_sector)}")
print("\nAll Consumer Defensive tickers:")
print(cd_sector[['ticker', 'actual_eps', 'consensus_estimate', 'epm_prediction',
                 'consensus_error', 'epm_error', 'error_reduction_pct']].to_string())

# Check for data quality issues
print("\n" + "="*60)
print("DATA QUALITY CHECKS")
print("="*60)

# Zero or negative EPS
zero_eps = backtest[backtest['actual_eps'] == 0]
print(f"\nTickers with zero actual EPS: {len(zero_eps)}")
if len(zero_eps) > 0:
    print(zero_eps[['ticker', 'sector', 'actual_eps', 'consensus_estimate']].to_string())

# Very small EPS (< 0.1)
small_eps = backtest[backtest['actual_eps'].abs() < 0.1]
print(f"\nTickers with |EPS| < 0.1: {len(small_eps)}")
if len(small_eps) > 0:
    print(small_eps[['ticker', 'sector', 'actual_eps', 'consensus_estimate', 
                     'consensus_error_pct']].head(10).to_string())

# Huge percentage errors (> 100%)
huge_errors = backtest[backtest['consensus_error_pct'] > 100]
print(f"\nTickers with consensus error > 100%: {len(huge_errors)}")
if len(huge_errors) > 0:
    print(huge_errors[['ticker', 'sector', 'actual_eps', 'consensus_estimate',
                       'consensus_error_pct']].to_string())

# Summary by sector
print("\n" + "="*60)
print("SECTOR SUMMARY")
print("="*60)

sector_summary = backtest.groupby('sector').agg({
    'ticker': 'count',
    'epm_better': 'mean',
    'error_reduction_pct': ['mean', 'median', 'min', 'max'],
    'consensus_error_pct': 'mean',
    'epm_error_pct': 'mean'
}).round(2)

print(sector_summary)

