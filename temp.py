import pickle
import pandas as pd

# Load pickle data
with open('data/raw/epm_data.pkl', 'rb') as f:
    data = pickle.load(f)

print("="*60)
print("PICKLE DATA STRUCTURE")
print("="*60)
print(f"Keys: {list(data.keys())}")
print(f"Tickers loaded: {list(data['data'].keys())}")
print(f"\nMetadata: {data['metadata']}")

# Inspect AAPL in detail
aapl = data['data']['AAPL']
print("\n" + "="*60)
print("AAPL DATA STRUCTURE")
print("="*60)
print(f"Keys: {list(aapl.keys())}")

print("\n--- Earnings Estimate ---")
print(f"Type: {type(aapl['earnings_estimate'])}")
if aapl['earnings_estimate'] is not None:
    print(f"Shape: {aapl['earnings_estimate'].shape}")
    print(f"Columns: {list(aapl['earnings_estimate'].columns)}")
    print(f"Index: {list(aapl['earnings_estimate'].index)}")
    print(aapl['earnings_estimate'])

print("\n--- Earnings History ---")
print(f"Type: {type(aapl['earnings_history'])}")
if aapl['earnings_history'] is not None:
    print(f"Shape: {aapl['earnings_history'].shape}")
    print(f"Columns: {list(aapl['earnings_history'].columns)}")
    print(f"Records (last 5):")
    print(aapl['earnings_history'].tail())

print("\n--- Revenue Estimate ---")
if aapl['revenue_estimate'] is not None:
    print(f"Shape: {aapl['revenue_estimate'].shape}")
    print(aapl['revenue_estimate'])

print("\n--- EPS Trend ---")
if aapl['eps_trend'] is not None:
    print(f"Shape: {aapl['eps_trend'].shape}")
    print(aapl['eps_trend'])

print("\n--- Info ---")
print(aapl['info'])

print("\n" + "="*60)
print("DATA QUALITY CHECKS")
print("="*60)

for ticker in data['data'].keys():
    ticker_data = data['data'][ticker]
    has_est = ticker_data['earnings_estimate'] is not None and not ticker_data['earnings_estimate'].empty
    has_hist = ticker_data['earnings_history'] is not None and not ticker_data['earnings_history'].empty
    print(f"{ticker:6s} - Est: {has_est:5} | Hist: {has_hist:5} | Sector: {ticker_data['info']['sector']}")

