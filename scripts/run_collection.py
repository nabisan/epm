"""
Standalone script for running data collection.
Execute from project root: python scripts/run_collection.py
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.data.collector import EPMDataCollector, SP500_TOP100
import argparse
import logging


def main():
    parser = argparse.ArgumentParser(description='EPM Data Collection')
    parser.add_argument('--tickers', type=int, default=50, help='Number of tickers to collect')
    parser.add_argument('--rate-limit', type=float, default=0.5, help='Rate limit between requests')
    parser.add_argument('--output', type=str, default='epm_data.pkl', help='Output filename')
    
    args = parser.parse_args()
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    tickers = SP500_TOP100[:args.tickers]
    
    collector = EPMDataCollector(
        tickers=tickers,
        data_dir=str(project_root / 'data' / 'raw')
    )
    
    data = collector.collect_all(rate_limit=args.rate_limit, verbose=True)
    
    collector.save(args.output)
    collector.save_csv()
    
    stats = collector.get_collection_stats()
    
    print("\n" + "="*60)
    print("FINAL STATISTICS")
    print("="*60)
    for key, value in stats.items():
        print(f"{key:30s}: {value}")
    print("="*60)


if __name__ == "__main__":
    main()

