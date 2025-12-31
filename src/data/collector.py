"""
Data collection module for Earnings Predictor Model (EPM).
Fetches analyst estimates and historical earnings data from Yahoo Finance.
"""

import yfinance as yf
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
from pathlib import Path
from datetime import datetime
import time
import pickle
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# S&P 500 Top 100 by market cap (approximate, as of late 2024)
SP500_TOP100 = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'BRK-B', 'UNH', 'JNJ',
    'V', 'XOM', 'WMT', 'JPM', 'PG', 'MA', 'HD', 'CVX', 'MRK', 'ABBV',
    'LLY', 'AVGO', 'KO', 'PEP', 'COST', 'ADBE', 'TMO', 'MCD', 'CSCO', 'ACN',
    'ABT', 'NKE', 'DHR', 'VZ', 'INTC', 'CRM', 'ORCL', 'WFC', 'TXN', 'PM',
    'NEE', 'UPS', 'BMY', 'RTX', 'HON', 'UNP', 'QCOM', 'AMGN', 'LOW', 'BA',
    'MS', 'COP', 'SPGI', 'SBUX', 'CAT', 'GE', 'BLK', 'AMD', 'LMT', 'INTU',
    'AXP', 'ISRG', 'DE', 'PLD', 'GS', 'MDLZ', 'TJX', 'MMC', 'ADP', 'BKNG',
    'SYK', 'GILD', 'ADI', 'CVS', 'REGN', 'CI', 'ZTS', 'CME', 'CB', 'DUK',
    'SO', 'BDX', 'PNC', 'ITW', 'EOG', 'CL', 'BSX', 'NOC', 'MO', 'ETN',
    'APD', 'USB', 'MMM', 'TGT', 'CSX', 'FIS', 'SHW', 'ICE', 'AON', 'MCO'
]


class EPMDataCollector:
    """
    Collects analyst estimates and historical earnings data for EPM model.
    
    Attributes:
        tickers: List of stock ticker symbols to collect data for
        data_dir: Directory to store collected data
        data: Dictionary storing collected data for each ticker
        errors: Dictionary tracking failed data fetches
    """
    
    def __init__(self, tickers: List[str], data_dir: str = 'data/raw'):
        """
        Initialize the data collector.
        
        Args:
            tickers: List of stock ticker symbols
            data_dir: Directory path for storing data
        """
        self.tickers = tickers
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        self.data = {}
        self.errors = {}
        
        logger.info(f"Initialized EPM Data Collector for {len(tickers)} tickers")
    
    def fetch_analyst_data(self, ticker: str) -> Optional[Dict]:
        """
        Fetch comprehensive analyst data for a single ticker.
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            Dictionary containing estimates, actuals, and metadata, or None on failure
        """
        try:
            stock = yf.Ticker(ticker)
            
            # Get all analysis data using property access (not methods)
            data = {
                'ticker': ticker,
                'fetch_timestamp': datetime.now(),
                'earnings_estimate': stock.earnings_estimate,      
                'earnings_history': stock.earnings_history,        
                'revenue_estimate': stock.revenue_estimate,        
                'eps_trend': stock.eps_trend,                      
                'eps_revisions': stock.eps_revisions,              
                'growth_estimates': stock.growth_estimates,        
                'calendar': stock.calendar,                        
                'earnings_dates': stock.earnings_dates,            
                'analyst_price_targets': stock.analyst_price_targets,  
                'recommendations': stock.recommendations,          
                'upgrades_downgrades': stock.upgrades_downgrades,  
                'info': {
                    'sector': stock.info.get('sector'),
                    'industry': stock.info.get('industry'),
                    'marketCap': stock.info.get('marketCap'),
                    'fullTimeEmployees': stock.info.get('fullTimeEmployees'),
                    'country': stock.info.get('country'),
                }
            }
            
            return data
            
        except Exception as e:
            logger.error(f"Failed to fetch data for {ticker}: {str(e)}")
            self.errors[ticker] = {
                'error': str(e),
                'timestamp': datetime.now()
            }
            return None
    
    def collect_all(self, rate_limit: float = 0.5, verbose: bool = True) -> Dict:
        """
        Collect analyst data for all tickers.
        
        Args:
            rate_limit: Seconds to wait between requests to avoid rate limiting
            verbose: Whether to print progress updates
            
        Returns:
            Dictionary mapping tickers to their data
        """
        logger.info(f"Starting data collection for {len(self.tickers)} tickers")
        
        for i, ticker in enumerate(self.tickers, 1):
            if verbose:
                print(f"[{i:3d}/{len(self.tickers):3d}] {ticker:6s} ... ", end='', flush=True)
            
            data = self.fetch_analyst_data(ticker)
            
            if data is not None:
                self.data[ticker] = data
                if verbose:
                    print("OK")
            else:
                if verbose:
                    print("FAILED")
            
            # Rate limiting
            if i < len(self.tickers):
                time.sleep(rate_limit)
        
        success_rate = len(self.data) / len(self.tickers) * 100
        logger.info(f"Collection complete: {len(self.data)}/{len(self.tickers)} successful ({success_rate:.1f}%)")
        
        return self.data
    
    def get_collection_stats(self) -> Dict:
        """Return statistics about the data collection"""
        if not self.data:
            return {'status': 'No data collected'}
        
        stats = {
            'total_tickers': len(self.tickers),
            'successful_fetches': len(self.data),
            'failed_fetches': len(self.errors),
            'success_rate_pct': f"{len(self.data) / len(self.tickers) * 100:.1f}",
            'tickers_with_earnings_estimate': sum(
                1 for d in self.data.values() 
                if d.get('earnings_estimate') is not None and not d['earnings_estimate'].empty
            ),
            'tickers_with_earnings_history': sum(
                1 for d in self.data.values() 
                if d.get('earnings_history') is not None and not d['earnings_history'].empty
            ),
            'tickers_with_revenue_estimate': sum(
                1 for d in self.data.values() 
                if d.get('revenue_estimate') is not None and not d['revenue_estimate'].empty
            ),
        }
        
        return stats
    
    def save(self, filename: str = 'epm_data.pkl'):
        """
        Save collected data to pickle file.
        
        Args:
            filename: Name of the output pickle file
        """
        filepath = self.data_dir / filename
        
        save_data = {
            'data': self.data,
            'errors': self.errors,
            'metadata': {
                'tickers': self.tickers,
                'collection_timestamp': datetime.now(),
                'num_successful': len(self.data),
                'num_failed': len(self.errors),
            }
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(save_data, f)
        
        size_mb = filepath.stat().st_size / (1024 * 1024)
        logger.info(f"Data saved to {filepath} ({size_mb:.2f} MB)")
    
    def save_csv(self, earnings_file: str = 'earnings_data.csv', 
                 estimates_file: str = 'earnings_estimates.csv'):
        """
        Save data in CSV format for easier inspection.
        
        Args:
            earnings_file: Filename for earnings history data
            estimates_file: Filename for earnings estimates data
        """
        # Save earnings history
        history_list = []
        for ticker, ticker_data in self.data.items():
            if ticker_data.get('earnings_history') is not None and not ticker_data['earnings_history'].empty:
                hist_df = ticker_data['earnings_history'].copy()
                hist_df['ticker'] = ticker
                hist_df['fetch_date'] = ticker_data['fetch_timestamp']
                history_list.append(hist_df)
        
        if history_list:
            history_df = pd.concat(history_list, ignore_index=False)
            history_path = self.data_dir / earnings_file
            history_df.to_csv(history_path)
            logger.info(f"Earnings history saved to {history_path}")
        
        # Save earnings estimates
        estimate_list = []
        for ticker, ticker_data in self.data.items():
            if ticker_data.get('earnings_estimate') is not None and not ticker_data['earnings_estimate'].empty:
                est_df = ticker_data['earnings_estimate'].copy()
                est_df['ticker'] = ticker
                est_df['fetch_date'] = ticker_data['fetch_timestamp']
                estimate_list.append(est_df)
        
        if estimate_list:
            estimate_df = pd.concat(estimate_list, ignore_index=False)
            estimate_path = self.data_dir / estimates_file
            estimate_df.to_csv(estimate_path)
            logger.info(f"Earnings estimates saved to {estimate_path}")
    
    @staticmethod
    def load(filepath: str) -> Tuple[Dict, Dict, Dict]:
        """
        Load previously collected data from pickle file.
        
        Args:
            filepath: Path to the pickle file
            
        Returns:
            Tuple of (data, errors, metadata)
        """
        with open(filepath, 'rb') as f:
            saved = pickle.load(f)
        
        logger.info(f"Loaded data from {filepath}")
        logger.info(f"Contains {len(saved['data'])} tickers")
        
        return saved['data'], saved['errors'], saved['metadata']


if __name__ == "__main__":
    # Example usage
    collector = EPMDataCollector(
        tickers=SP500_TOP100[:10],
        data_dir='data/raw'
    )
    
    data = collector.collect_all(rate_limit=0.5, verbose=True)
    
    collector.save('test_epm_data.pkl')
    collector.save_csv()
    
    stats = collector.get_collection_stats()
    print("\nCollection Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")

