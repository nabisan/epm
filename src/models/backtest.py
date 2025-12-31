"""
Backtesting framework for EPM model.
Validates EPM predictions against actual earnings results.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class EPMBacktest:
    """
    Backtest EPM predictions against actual earnings.
    Measures accuracy improvement vs consensus.
    """
    
    def __init__(self, predictions_df: pd.DataFrame, actuals_data: Dict):
        """
        Initialize backtest with predictions and actual results.
        
        Args:
            predictions_df: DataFrame with EPM predictions
            actuals_data: Dictionary with actual earnings data from collector
        """
        self.predictions = predictions_df.copy()
        self.actuals_data = actuals_data
        self.results = None
        
        logger.info(f"Initialized EPM Backtest for {len(predictions_df)} tickers")
    
    def extract_actual_earnings(self, ticker: str) -> Optional[float]:
        """
        Extract most recent actual earnings for a ticker.
        Uses the most recent COMPLETED quarter from earnings_history.
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            Actual EPS or None if not available
        """
        if ticker not in self.actuals_data:
            return None
        
        earnings_history = self.actuals_data[ticker].get('earnings_history')
        
        if earnings_history is None or earnings_history.empty:
            return None
        
        # Get most recent actual EPS
        # This represents the LAST REPORTED quarter
        recent_actual = earnings_history['epsActual'].iloc[-1]
        
        return recent_actual if pd.notna(recent_actual) else None
    
    def calculate_prediction_errors(self) -> pd.DataFrame:
        """
        Calculate prediction errors for both consensus and EPM.
        NOTE: This is a pseudo-backtest using most recent historical earnings.
        """
        results = []
        
        for idx, row in self.predictions.iterrows():
            ticker = row['ticker']
            actual_eps = self.extract_actual_earnings(ticker)
            
            if actual_eps is None:
                logger.warning(f"No actual earnings found for {ticker}, skipping")
                continue
            
            consensus_estimate = row['consensus_estimate']
            epm_prediction = row['epm_prediction']
            
            # Data quality check: Skip if actual vs consensus differ by > 200%
            if abs(actual_eps) > 0:
                pct_diff = abs(actual_eps - consensus_estimate) / abs(actual_eps) * 100
                if pct_diff > 200:
                    logger.warning(f"Skipping {ticker}: Large mismatch (actual={actual_eps}, consensus={consensus_estimate}, diff={pct_diff:.1f}%)")
                    continue
            
            # Calculate errors
            consensus_error = abs(actual_eps - consensus_estimate)
            epm_error = abs(actual_eps - epm_prediction)
            
            # Skip if consensus error is extremely small (< 0.01)
            if consensus_error < 0.01:
                logger.info(f"Skipping {ticker}: Consensus nearly perfect (error={consensus_error:.4f})")
                continue
            
            # Calculate percentage errors
            if actual_eps != 0:
                consensus_error_pct = (consensus_error / abs(actual_eps)) * 100
                epm_error_pct = (epm_error / abs(actual_eps)) * 100
            else:
                consensus_error_pct = np.nan
                epm_error_pct = np.nan
            
            # Calculate improvement
            error_reduction = consensus_error - epm_error
            error_reduction_pct = ((consensus_error - epm_error) / consensus_error * 100)
            
            # Directional accuracy
            if pd.notna(row.get('adjustment_pct')):
                consensus_direction_correct = np.sign(actual_eps - consensus_estimate) == np.sign(row['adjustment_pct'])
            else:
                consensus_direction_correct = False
            
            results.append({
                'ticker': ticker,
                'sector': row['sector'],
                'actual_eps': actual_eps,
                'consensus_estimate': consensus_estimate,
                'epm_prediction': epm_prediction,
                'adjustment_pct': row.get('adjustment_pct', 0),
                'consensus_error': consensus_error,
                'epm_error': epm_error,
                'consensus_error_pct': consensus_error_pct,
                'epm_error_pct': epm_error_pct,
                'error_reduction': error_reduction,
                'error_reduction_pct': error_reduction_pct,
                'epm_better': epm_error < consensus_error,
                'direction_correct': consensus_direction_correct,
                'quintile': row.get('quintile', 'Unknown'),
            })
        
        results_df = pd.DataFrame(results)
        self.results = results_df
        
        logger.info(f"Calculated prediction errors for {len(results_df)} tickers")
        logger.warning(f"NOTE: Pseudo-backtest using most recent historical earnings. Not true out-of-sample test.")
        
        return results_df
    
    def calculate_accuracy_metrics(self) -> Dict:
        """
        Calculate comprehensive accuracy metrics.
        
        Returns:
            Dictionary with accuracy statistics
        """
        if self.results is None:
            self.calculate_prediction_errors()
        
        metrics = {
            # Basic counts
            'num_predictions': len(self.results),
            'num_epm_better': self.results['epm_better'].sum(),
            'num_consensus_better': (~self.results['epm_better']).sum(),
            
            # Win rates
            'epm_win_rate': self.results['epm_better'].mean() * 100,
            'direction_accuracy': self.results['direction_correct'].mean() * 100,
            
            # Mean absolute errors
            'mean_consensus_error': self.results['consensus_error'].mean(),
            'mean_epm_error': self.results['epm_error'].mean(),
            'mean_error_reduction': self.results['error_reduction'].mean(),
            
            # Percentage errors
            'mean_consensus_error_pct': self.results['consensus_error_pct'].mean(),
            'mean_epm_error_pct': self.results['epm_error_pct'].mean(),
            
            # Improvement metrics
            'mean_improvement_pct': self.results['error_reduction_pct'].mean(),
            'median_improvement_pct': self.results['error_reduction_pct'].median(),
            
            # By sector
            'best_sector': self.results.groupby('sector')['epm_better'].mean().idxmax(),
            'worst_sector': self.results.groupby('sector')['epm_better'].mean().idxmin(),
        }
        
        return metrics
    
    def quintile_performance(self) -> pd.DataFrame:
        """
        Analyze backtest performance by EPM quintile.
        Tests if extreme quintiles have better predictive power.
        
        Returns:
            DataFrame with quintile performance metrics
        """
        if self.results is None:
            self.calculate_prediction_errors()
        
        quintile_stats = self.results.groupby('quintile', observed=False).agg({
            'ticker': 'count',
            'epm_better': 'mean',
            'direction_correct': 'mean',
            'error_reduction': 'mean',
            'error_reduction_pct': 'mean',
            'consensus_error_pct': 'mean',
            'epm_error_pct': 'mean',
        }).round(4)
        
        quintile_stats.columns = [
            'count', 
            'epm_win_rate', 
            'direction_accuracy',
            'avg_error_reduction',
            'avg_improvement_pct',
            'consensus_error_pct',
            'epm_error_pct'
        ]
        
        return quintile_stats
    
    def sector_performance(self) -> pd.DataFrame:
        """
        Analyze backtest performance by sector.
        
        Returns:
            DataFrame with sector performance metrics
        """
        if self.results is None:
            self.calculate_prediction_errors()
        
        sector_stats = self.results.groupby('sector').agg({
            'ticker': 'count',
            'epm_better': 'mean',
            'direction_correct': 'mean',
            'error_reduction': 'mean',
            'error_reduction_pct': 'mean',
        }).round(4)
        
        sector_stats.columns = [
            'count',
            'epm_win_rate',
            'direction_accuracy', 
            'avg_error_reduction',
            'avg_improvement_pct'
        ]
        
        return sector_stats.sort_values('epm_win_rate', ascending=False)
    
    def get_best_predictions(self, top_n: int = 5) -> pd.DataFrame:
        """
        Get top N best EPM predictions (largest error reduction).
        
        Args:
            top_n: Number of top predictions to return
            
        Returns:
            DataFrame with best predictions
        """
        if self.results is None:
            self.calculate_prediction_errors()
        
        best = self.results.nlargest(top_n, 'error_reduction')
        
        return best[['ticker', 'sector', 'actual_eps', 'consensus_estimate', 
                     'epm_prediction', 'error_reduction', 'error_reduction_pct']]
    
    def get_worst_predictions(self, top_n: int = 5) -> pd.DataFrame:
        """
        Get top N worst EPM predictions (largest error increase).
        
        Args:
            top_n: Number of worst predictions to return
            
        Returns:
            DataFrame with worst predictions
        """
        if self.results is None:
            self.calculate_prediction_errors()
        
        worst = self.results.nsmallest(top_n, 'error_reduction')
        
        return worst[['ticker', 'sector', 'actual_eps', 'consensus_estimate',
                      'epm_prediction', 'error_reduction', 'error_reduction_pct']]
    
    def save_results(self, filepath: str):
        """Save backtest results to CSV"""
        if self.results is None:
            self.calculate_prediction_errors()
        
        self.results.to_csv(filepath, index=False)
        logger.info(f"Backtest results saved to {filepath}")
    
    def generate_report(self) -> str:
        """
        Generate comprehensive backtest report.
        
        Returns:
            Formatted report string
        """
        if self.results is None:
            self.calculate_prediction_errors()
        
        metrics = self.calculate_accuracy_metrics()
        
        report = []
        report.append("="*60)
        report.append("EPM BACKTEST REPORT")
        report.append("="*60)
        report.append(f"Total Predictions: {metrics['num_predictions']}")
        report.append(f"EPM Better: {metrics['num_epm_better']} ({metrics['epm_win_rate']:.1f}%)")
        report.append(f"Consensus Better: {metrics['num_consensus_better']}")
        report.append(f"Direction Accuracy: {metrics['direction_accuracy']:.1f}%")
        report.append("")
        report.append("ERROR METRICS")
        report.append("-"*60)
        report.append(f"Mean Consensus Error: {metrics['mean_consensus_error']:.4f}")
        report.append(f"Mean EPM Error: {metrics['mean_epm_error']:.4f}")
        report.append(f"Mean Error Reduction: {metrics['mean_error_reduction']:.4f}")
        report.append(f"Mean Improvement: {metrics['mean_improvement_pct']:.2f}%")
        report.append(f"Median Improvement: {metrics['median_improvement_pct']:.2f}%")
        report.append("")
        report.append("PERCENTAGE ERRORS")
        report.append("-"*60)
        report.append(f"Consensus Error %: {metrics['mean_consensus_error_pct']:.2f}%")
        report.append(f"EPM Error %: {metrics['mean_epm_error_pct']:.2f}%")
        report.append("")
        report.append("SECTOR PERFORMANCE")
        report.append("-"*60)
        report.append(f"Best Sector: {metrics['best_sector']}")
        report.append(f"Worst Sector: {metrics['worst_sector']}")
        report.append("")
        report.append("NOTE: Pseudo-backtest using most recent historical earnings.")
        report.append("Not a true out-of-sample test. Treat results as directional.")
        report.append("="*60)
        
        return "\n".join(report)


if __name__ == "__main__":
    # Test backtesting
    import sys
    from pathlib import Path
    
    project_root = Path(__file__).parent.parent.parent
    sys.path.insert(0, str(project_root))
    
    from src.data.collector import EPMDataCollector
    
    # Load data
    data_path = project_root / 'data' / 'raw' / 'epm_data.pkl'
    data, errors, metadata = EPMDataCollector.load(data_path)
    
    # Load predictions
    predictions_path = project_root / 'data' / 'processed' / 'epm_predictions.csv'
    predictions_df = pd.read_csv(predictions_path)
    
    # Run backtest
    backtest = EPMBacktest(predictions_df, data)
    results = backtest.calculate_prediction_errors()
    
    # Print report
    print(backtest.generate_report())
    
    print("\n" + "="*60)
    print("QUINTILE PERFORMANCE")
    print("="*60)
    print(backtest.quintile_performance())
    
    print("\n" + "="*60)
    print("SECTOR PERFORMANCE")
    print("="*60)
    print(backtest.sector_performance())
    
    print("\n" + "="*60)
    print("TOP 5 BEST PREDICTIONS")
    print("="*60)
    print(backtest.get_best_predictions(5).to_string())
    
    print("\n" + "="*60)
    print("TOP 5 WORST PREDICTIONS")
    print("="*60)
    print(backtest.get_worst_predictions(5).to_string())
    
    # Save results
    output_path = project_root / 'data' / 'processed' / 'backtest_results.csv'
    backtest.save_results(output_path)

