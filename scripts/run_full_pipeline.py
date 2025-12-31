#!/usr/bin/env python3
"""
Full EPM Pipeline: Data → Features → Model → Backtest
Production-ready end-to-end pipeline
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.data.collector import EPMDataCollector, SP500_TOP100
from src.features.engineer import EPMFeatureEngineer
from src.models.epm_model import EPMModel
from src.models.backtest import EPMBacktest
import argparse
import logging


def main():
    parser = argparse.ArgumentParser(description='Full EPM Pipeline')
    parser.add_argument('--tickers', type=int, default=50, help='Number of tickers to analyze')
    parser.add_argument('--rate-limit', type=float, default=0.5, help='Rate limit between API calls')
    parser.add_argument('--skip-collection', action='store_true', help='Skip data collection, use existing')
    
    args = parser.parse_args()
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    data_dir = project_root / 'data' / 'raw'
    processed_dir = project_root / 'data' / 'processed'
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Data Collection
    if not args.skip_collection:
        print("\n" + "="*60)
        print("STEP 1: DATA COLLECTION")
        print("="*60)
        
        tickers = SP500_TOP100[:args.tickers]
        collector = EPMDataCollector(tickers=tickers, data_dir=str(data_dir))
        data = collector.collect_all(rate_limit=args.rate_limit, verbose=True)
        
        collector.save(f'epm_sp500_top{args.tickers}.pkl')
        collector.save_csv()
        
        stats = collector.get_collection_stats()
        print("\nCollection Stats:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
    else:
        print("\n" + "="*60)
        print("STEP 1: LOADING EXISTING DATA")
        print("="*60)
        data, errors, metadata = EPMDataCollector.load(
            data_dir / f'epm_sp500_top{args.tickers}.pkl'
        )
        print(f"Loaded {len(data)} tickers")
    
    # Step 2: Feature Engineering
    print("\n" + "="*60)
    print("STEP 2: FEATURE ENGINEERING")
    print("="*60)
    
    engineer = EPMFeatureEngineer(data)
    features_df = engineer.engineer_all_features()
    
    print(f"Engineered features: {features_df.shape}")
    print(f"Columns: {len(features_df.columns)}")
    
    features_path = processed_dir / f'epm_features_top{args.tickers}.csv'
    engineer.save_features(features_path)
    
    # Step 3: EPM Model
    print("\n" + "="*60)
    print("STEP 3: EPM MODEL PREDICTIONS")
    print("="*60)
    
    model = EPMModel(features_df)
    predictions = model.generate_predictions()
    
    print(f"Generated predictions: {len(predictions)}")
    
    stats = model.get_summary_stats()
    print("\nModel Stats:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    predictions_path = processed_dir / f'epm_predictions_top{args.tickers}.csv'
    model.save_predictions(predictions_path)
    
    # Step 4: Backtesting
    print("\n" + "="*60)
    print("STEP 4: BACKTESTING")
    print("="*60)
    
    backtest = EPMBacktest(predictions, data)
    results = backtest.calculate_prediction_errors()
    
    print(backtest.generate_report())
    
    print("\n" + "="*60)
    print("QUINTILE PERFORMANCE")
    print("="*60)
    print(backtest.quintile_performance())
    
    print("\n" + "="*60)
    print("SECTOR PERFORMANCE")
    print("="*60)
    print(backtest.sector_performance())
    
    backtest_path = processed_dir / f'backtest_results_top{args.tickers}.csv'
    backtest.save_results(backtest_path)
    
    # Summary
    print("\n" + "="*60)
    print("PIPELINE COMPLETE")
    print("="*60)
    print(f"Data: {data_dir / f'epm_sp500_top{args.tickers}.pkl'}")
    print(f"Features: {features_path}")
    print(f"Predictions: {predictions_path}")
    print(f"Backtest: {backtest_path}")
    print("="*60)


if __name__ == "__main__":
    main()

