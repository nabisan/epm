"""
Feature engineering module for EPM (Earnings Predictor Model).
Implements Herzberg et al. (1999) methodology.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class EPMFeatureEngineer:
    """
    Feature engineering for earnings prediction based on:
    - Forecast recency weighting
    - Analyst track record
    - Estimate clustering
    - Revision frequency and direction
    """
    
    def __init__(self, data: Dict):
        """
        Initialize feature engineer with collected data.
        
        Args:
            data: Dictionary from EPMDataCollector containing ticker data
        """
        self.data = data
        self.features = {}
        logger.info(f"Initialized EPM Feature Engineer for {len(data)} tickers")
    
    def calculate_recency_weights(self, eps_trend: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate recency-weighted consensus estimate.
        More recent estimates get higher weight.
        
        Args:
            eps_trend: DataFrame with columns [current, 7daysAgo, 30daysAgo, 60daysAgo, 90daysAgo]
            
        Returns:
            Dictionary with recency-weighted estimates for each period
        """
        if eps_trend is None or eps_trend.empty:
            return {}
        
        # Exponential decay weights (more recent = higher weight)
        weights = {
            'current': 0.40,
            '7daysAgo': 0.25,
            '30daysAgo': 0.20,
            '60daysAgo': 0.10,
            '90daysAgo': 0.05
        }
        
        recency_estimates = {}
        
        for period in eps_trend.index:
            weighted_sum = 0
            total_weight = 0
            
            for col, weight in weights.items():
                if col in eps_trend.columns:
                    val = eps_trend.loc[period, col]
                    if pd.notna(val):
                        weighted_sum += val * weight
                        total_weight += weight
            
            if total_weight > 0:
                recency_estimates[period] = weighted_sum / total_weight
            else:
                recency_estimates[period] = eps_trend.loc[period, 'current']
        
        return recency_estimates
    
    def calculate_revision_momentum(self, eps_trend: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate estimate revision momentum (current vs historical).
        Positive = upward revisions, Negative = downward revisions
        
        Args:
            eps_trend: DataFrame with estimate trends
            
        Returns:
            Dictionary with revision momentum for each period
        """
        if eps_trend is None or eps_trend.empty:
            return {}
        
        momentum = {}
        
        for period in eps_trend.index:
            current = eps_trend.loc[period, 'current']
            days_90 = eps_trend.loc[period, '90daysAgo']
            
            if pd.notna(current) and pd.notna(days_90) and days_90 != 0:
                momentum[period] = (current - days_90) / abs(days_90)
            else:
                momentum[period] = 0.0
        
        return momentum
    
    def calculate_estimate_clustering(self, earnings_estimate: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate estimate clustering metric.
        Lower spread = higher analyst agreement = more reliable
        
        Args:
            earnings_estimate: DataFrame with avg, low, high estimates
            
        Returns:
            Dictionary with clustering scores (0-1, higher = more clustered)
        """
        if earnings_estimate is None or earnings_estimate.empty:
            return {}
        
        clustering = {}
        
        for period in earnings_estimate.index:
            avg = earnings_estimate.loc[period, 'avg']
            low = earnings_estimate.loc[period, 'low']
            high = earnings_estimate.loc[period, 'high']
            
            if pd.notna(avg) and pd.notna(low) and pd.notna(high) and avg != 0:
                spread = (high - low) / abs(avg)
                # Convert spread to clustering score (inverse relationship)
                clustering[period] = 1 / (1 + spread)
            else:
                clustering[period] = 0.5  # neutral
        
        return clustering
    
    def calculate_analyst_coverage_score(self, earnings_estimate: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate analyst coverage quality score.
        More analysts = more reliable (with diminishing returns)
        
        Args:
            earnings_estimate: DataFrame with numberOfAnalysts
            
        Returns:
            Dictionary with coverage scores (0-1)
        """
        if earnings_estimate is None or earnings_estimate.empty:
            return {}
        
        coverage = {}
        
        for period in earnings_estimate.index:
            num_analysts = earnings_estimate.loc[period, 'numberOfAnalysts']
            
            if pd.notna(num_analysts):
                # Logarithmic scaling: diminishing returns after ~20 analysts
                coverage[period] = min(1.0, np.log1p(num_analysts) / np.log1p(40))
            else:
                coverage[period] = 0.0
        
        return coverage
    
    def calculate_historical_accuracy(self, earnings_history: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate historical forecast accuracy metrics.
        
        Args:
            earnings_history: DataFrame with epsActual, epsEstimate, surprisePercent
            
        Returns:
            Dictionary with accuracy metrics
        """
        if earnings_history is None or earnings_history.empty:
            return {
                'mean_surprise_pct': 0.0,
                'abs_mean_surprise_pct': 0.0,
                'surprise_std': 0.0,
                'beat_rate': 0.5,
                'recent_surprise_pct': 0.0
            }
        
        surprises = earnings_history['surprisePercent'].dropna()
        
        if len(surprises) == 0:
            return {
                'mean_surprise_pct': 0.0,
                'abs_mean_surprise_pct': 0.0,
                'surprise_std': 0.0,
                'beat_rate': 0.5,
                'recent_surprise_pct': 0.0
            }
        
        # Calculate metrics
        mean_surprise = surprises.mean()
        abs_mean_surprise = surprises.abs().mean()
        surprise_std = surprises.std()
        beat_rate = (surprises > 0).sum() / len(surprises)
        
        # Most recent surprise (last quarter)
        recent_surprise = surprises.iloc[-1] if len(surprises) > 0 else 0.0
        
        return {
            'mean_surprise_pct': mean_surprise,
            'abs_mean_surprise_pct': abs_mean_surprise,
            'surprise_std': surprise_std if pd.notna(surprise_std) else 0.0,
            'beat_rate': beat_rate,
            'recent_surprise_pct': recent_surprise
        }
    
    def calculate_growth_consistency(self, earnings_estimate: pd.DataFrame, 
                                    earnings_history: pd.DataFrame) -> float:
        """
        Calculate growth estimate consistency with historical performance.
        
        Args:
            earnings_estimate: DataFrame with growth estimates
            earnings_history: DataFrame with historical earnings
            
        Returns:
            Consistency score (0-1)
        """
        if earnings_estimate is None or earnings_history is None:
            return 0.5
        
        if earnings_estimate.empty or earnings_history.empty:
            return 0.5
        
        # Get estimated growth for current year
        if '0y' in earnings_estimate.index and 'growth' in earnings_estimate.columns:
            estimated_growth = earnings_estimate.loc['0y', 'growth']
        else:
            return 0.5
        
        # Calculate historical growth rate
        if len(earnings_history) >= 4:
            recent_eps = earnings_history['epsActual'].iloc[-4:].values
            if len(recent_eps) >= 2 and recent_eps[0] != 0:
                historical_growth = (recent_eps[-1] - recent_eps[0]) / abs(recent_eps[0])
                
                # Compare estimated vs historical growth
                if pd.notna(estimated_growth) and pd.notna(historical_growth):
                    growth_diff = abs(estimated_growth - historical_growth)
                    # Consistency score: lower difference = higher score
                    consistency = 1 / (1 + growth_diff)
                    return consistency
        
        return 0.5
    
    def engineer_ticker_features(self, ticker: str) -> Dict:
        """
        Engineer all features for a single ticker.
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            Dictionary containing all engineered features
        """
        ticker_data = self.data[ticker]
        
        # Extract raw data
        earnings_estimate = ticker_data.get('earnings_estimate')
        earnings_history = ticker_data.get('earnings_history')
        eps_trend = ticker_data.get('eps_trend')
        revenue_estimate = ticker_data.get('revenue_estimate')
        
        # Calculate features
        recency_weights = self.calculate_recency_weights(eps_trend)
        revision_momentum = self.calculate_revision_momentum(eps_trend)
        clustering = self.calculate_estimate_clustering(earnings_estimate)
        coverage = self.calculate_analyst_coverage_score(earnings_estimate)
        accuracy_metrics = self.calculate_historical_accuracy(earnings_history)
        growth_consistency = self.calculate_growth_consistency(earnings_estimate, earnings_history)
        
        # Combine into feature dictionary
        features = {
            'ticker': ticker,
            'sector': ticker_data['info']['sector'],
            'industry': ticker_data['info']['industry'],
            'market_cap': ticker_data['info']['marketCap'],
            
            # Current quarter (0q) features
            'consensus_estimate_0q': earnings_estimate.loc['0q', 'avg'] if earnings_estimate is not None and '0q' in earnings_estimate.index else None,
            'recency_weighted_0q': recency_weights.get('0q'),
            'revision_momentum_0q': revision_momentum.get('0q'),
            'clustering_score_0q': clustering.get('0q'),
            'coverage_score_0q': coverage.get('0q'),
            'num_analysts_0q': earnings_estimate.loc['0q', 'numberOfAnalysts'] if earnings_estimate is not None and '0q' in earnings_estimate.index else None,
            
            # Next quarter (+1q) features
            'consensus_estimate_1q': earnings_estimate.loc['+1q', 'avg'] if earnings_estimate is not None and '+1q' in earnings_estimate.index else None,
            'recency_weighted_1q': recency_weights.get('+1q'),
            'revision_momentum_1q': revision_momentum.get('+1q'),
            'clustering_score_1q': clustering.get('+1q'),
            'coverage_score_1q': coverage.get('+1q'),
            
            # Historical accuracy
            'mean_surprise_pct': accuracy_metrics['mean_surprise_pct'],
            'abs_mean_surprise_pct': accuracy_metrics['abs_mean_surprise_pct'],
            'surprise_std': accuracy_metrics['surprise_std'],
            'beat_rate': accuracy_metrics['beat_rate'],
            'recent_surprise_pct': accuracy_metrics['recent_surprise_pct'],
            
            # Growth and consistency
            'growth_estimate_0y': earnings_estimate.loc['0y', 'growth'] if earnings_estimate is not None and '0y' in earnings_estimate.index else None,
            'growth_consistency': growth_consistency,
            
            'timestamp': datetime.now()
        }
        
        return features
    
    def engineer_all_features(self) -> pd.DataFrame:
        """
        Engineer features for all tickers.
        
        Returns:
            DataFrame containing features for all tickers
        """
        logger.info(f"Engineering features for {len(self.data)} tickers")
        
        feature_list = []
        
        for ticker in self.data.keys():
            try:
                features = self.engineer_ticker_features(ticker)
                feature_list.append(features)
                self.features[ticker] = features
            except Exception as e:
                logger.error(f"Failed to engineer features for {ticker}: {str(e)}")
        
        features_df = pd.DataFrame(feature_list)
        
        logger.info(f"Feature engineering complete: {len(features_df)} tickers, {len(features_df.columns)} features")
        
        return features_df
    
    def save_features(self, filepath: str):
        """Save engineered features to CSV"""
        features_df = pd.DataFrame(list(self.features.values()))
        features_df.to_csv(filepath, index=False)
        logger.info(f"Features saved to {filepath}")


if __name__ == "__main__":
    # Test feature engineering
    import sys
    from pathlib import Path
    
    # Add project root to path
    project_root = Path(__file__).parent.parent.parent
    sys.path.insert(0, str(project_root))
    
    from src.data.collector import EPMDataCollector
    
    # Load data
    data_path = project_root / 'data' / 'raw' / 'epm_data.pkl'
    data, errors, metadata = EPMDataCollector.load(data_path)
    
    # Engineer features
    engineer = EPMFeatureEngineer(data)
    features_df = engineer.engineer_all_features()
    
    print("\n" + "="*60)
    print("FEATURE ENGINEERING RESULTS")
    print("="*60)
    print(f"Shape: {features_df.shape}")
    print(f"\nColumns:\n{list(features_df.columns)}")
    print(f"\nSample features (AAPL):")
    aapl_features = features_df[features_df['ticker'] == 'AAPL'].T
    print(aapl_features)
    
    print("\n" + "="*60)
    print("FEATURE STATISTICS")
    print("="*60)
    print(features_df[['recency_weighted_0q', 'revision_momentum_0q', 
                       'clustering_score_0q', 'coverage_score_0q', 
                       'beat_rate', 'growth_consistency']].describe())
    
    # Save
    output_path = project_root / 'data' / 'processed'
    output_path.mkdir(parents=True, exist_ok=True)
    engineer.save_features(output_path / 'epm_features.csv')

