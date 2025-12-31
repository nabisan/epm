"""
Enhanced Earnings Predictor Model (EPM)
Implementation based on Herzberg, Guo, Brown (1999)
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class EPMModel:
    """
    Enhanced Earnings Predictor Model
    
    Combines multiple signals to improve earnings forecast accuracy:
    - Recency-weighted consensus (40%)
    - Revision momentum (20%)
    - Analyst clustering (20%)
    - Historical accuracy (20%)
    """
    
    def __init__(self, features_df: pd.DataFrame):
        """
        Initialize EPM model with engineered features.
        
        Args:
            features_df: DataFrame containing engineered features
        """
        self.features_df = features_df.copy()
        self.predictions = None
        
        # Model weights (Herzberg et al. 1999 based)
        self.weights = {
            'recency': 0.40,
            'momentum': 0.20,
            'clustering': 0.20,
            'accuracy': 0.20
        }
        
        logger.info(f"Initialized EPM Model for {len(features_df)} tickers")
    
    def normalize_signal(self, series: pd.Series, 
                        clip_lower: float = -3, 
                        clip_upper: float = 3) -> pd.Series:
        """
        Normalize signal to z-scores with clipping.
        
        Args:
            series: Input series to normalize
            clip_lower: Lower bound for clipping
            clip_upper: Upper bound for clipping
            
        Returns:
            Normalized series
        """
        mean = series.mean()
        std = series.std()
        
        if std == 0 or pd.isna(std):
            return pd.Series(0, index=series.index)
        
        z_scores = (series - mean) / std
        return z_scores.clip(clip_lower, clip_upper)
    
    def calculate_recency_signal(self) -> pd.Series:
        """
        Calculate recency adjustment signal.
        Positive = recency-weighted > consensus (bullish)
        Negative = recency-weighted < consensus (bearish)
        
        Returns:
            Normalized recency signal
        """
        recency_diff = (
            self.features_df['recency_weighted_0q'] - 
            self.features_df['consensus_estimate_0q']
        ) / self.features_df['consensus_estimate_0q'].abs()
        
        return self.normalize_signal(recency_diff)
    
    def calculate_momentum_signal(self) -> pd.Series:
        """
        Calculate revision momentum signal.
        Positive momentum = upward revisions (bullish)
        Negative momentum = downward revisions (bearish)
        
        Returns:
            Normalized momentum signal
        """
        momentum = self.features_df['revision_momentum_0q'].copy()
        return self.normalize_signal(momentum)
    
    def calculate_clustering_signal(self) -> pd.Series:
        """
        Calculate analyst clustering confidence signal.
        High clustering = high confidence in consensus
        Low clustering = low confidence, expect deviation
        
        Returns:
            Clustering confidence score (0-1)
        """
        clustering = self.features_df['clustering_score_0q'].copy()
        # Already 0-1 scaled, no normalization needed
        return clustering
    
    def calculate_accuracy_signal(self) -> pd.Series:
        """
        Calculate historical accuracy adjustment signal.
        Positive beat rate + recent surprise = bullish adjustment
        
        Returns:
            Normalized accuracy signal
        """
        # Combine beat rate and recent surprise
        accuracy_score = (
            self.features_df['beat_rate'] * 0.5 + 
            self.features_df['recent_surprise_pct'].clip(-0.5, 0.5) * 0.5
        )
        
        return self.normalize_signal(accuracy_score)
    
    def calculate_composite_adjustment(self) -> pd.Series:
        """
        Calculate composite EPM adjustment factor.
        Combines all signals with model weights.
        
        Returns:
            Composite adjustment factor (percentage)
        """
        # Calculate individual signals
        recency_signal = self.calculate_recency_signal()
        momentum_signal = self.calculate_momentum_signal()
        clustering_signal = self.calculate_clustering_signal()
        accuracy_signal = self.calculate_accuracy_signal()
        
        # Store signals for analysis
        self.features_df['signal_recency'] = recency_signal
        self.features_df['signal_momentum'] = momentum_signal
        self.features_df['signal_clustering'] = clustering_signal
        self.features_df['signal_accuracy'] = accuracy_signal
        
        # Composite adjustment
        # Recency and momentum are directional (-/+)
        # Clustering modulates confidence (0-1)
        # Accuracy provides bias adjustment
        
        directional_component = (
            recency_signal * self.weights['recency'] +
            momentum_signal * self.weights['momentum'] +
            accuracy_signal * self.weights['accuracy']
        )
        
        # Clustering acts as confidence multiplier
        confidence_multiplier = clustering_signal
        
        # Final adjustment (scaled to reasonable range)
        composite_adjustment = directional_component * confidence_multiplier * 0.05
        
        return composite_adjustment
    
    def generate_predictions(self) -> pd.DataFrame:
        """
        Generate EPM earnings predictions.
        
        Returns:
            DataFrame with consensus, EPM predictions, and adjustments
        """
        logger.info("Generating EPM predictions")
        
        # Calculate composite adjustment
        adjustment_pct = self.calculate_composite_adjustment()
        
        # Apply adjustment to consensus
        consensus = self.features_df['consensus_estimate_0q']
        epm_prediction = consensus * (1 + adjustment_pct)
        
        # Create predictions DataFrame
        predictions = pd.DataFrame({
            'ticker': self.features_df['ticker'],
            'sector': self.features_df['sector'],
            'consensus_estimate': consensus,
            'epm_prediction': epm_prediction,
            'adjustment_pct': adjustment_pct * 100,  # Convert to percentage
            'adjustment_amount': epm_prediction - consensus,
            
            # Signal components
            'recency_signal': self.features_df['signal_recency'],
            'momentum_signal': self.features_df['signal_momentum'],
            'clustering_score': self.features_df['signal_clustering'],
            'accuracy_signal': self.features_df['signal_accuracy'],
            
            # Metadata
            'num_analysts': self.features_df['num_analysts_0q'],
            'coverage_score': self.features_df['coverage_score_0q'],
            'beat_rate': self.features_df['beat_rate'],
            'recent_surprise': self.features_df['recent_surprise_pct'] * 100,
        })
        
        # Assign quintiles with proper handling
        try:
            predictions['quintile'] = pd.qcut(
                predictions['adjustment_pct'], 
                q=5, 
                labels=['Q1 (Most Bearish)', 'Q2', 'Q3', 'Q4', 'Q5 (Most Bullish)'],
                duplicates='drop'  # Handle duplicates
            )
        except ValueError as e:
            logger.warning(f"Could not create quintiles: {e}. Using quantiles instead.")
            predictions['quintile'] = pd.cut(
                predictions['adjustment_pct'],
                bins=5,
                labels=['Q1 (Most Bearish)', 'Q2', 'Q3', 'Q4', 'Q5 (Most Bullish)']
            )
        
        self.predictions = predictions
        
        logger.info(f"Generated predictions for {len(predictions)} tickers")
        logger.info(f"Mean adjustment: {adjustment_pct.mean()*100:.2f}%")
        logger.info(f"Adjustment range: [{adjustment_pct.min()*100:.2f}%, {adjustment_pct.max()*100:.2f}%]")
        
        return predictions
    
    def rank_predictions(self, metric: str = 'adjustment_pct') -> pd.DataFrame:
        """
        Rank tickers by EPM signal strength.
        
        Args:
            metric: Metric to rank by ('adjustment_pct', 'epm_prediction', etc.)
            
        Returns:
            Sorted DataFrame with rankings
        """
        if self.predictions is None:
            self.generate_predictions()
        
        ranked = self.predictions.copy()
        ranked['rank'] = ranked[metric].rank(ascending=False, method='dense').astype(int)
        ranked = ranked.sort_values('rank')
        
        return ranked
    
    def get_quintile_analysis(self) -> pd.DataFrame:
        """
        Analyze predictions by quintiles.
        Used to evaluate EPM spread and signal strength.
        
        Returns:
            DataFrame with quintile statistics
        """
        if self.predictions is None:
            self.generate_predictions()
        
        quintile_stats = self.predictions.groupby('quintile', observed=False).agg({
            'ticker': 'count',
            'adjustment_pct': ['mean', 'std'],
            'epm_prediction': 'mean',
            'consensus_estimate': 'mean',
            'beat_rate': 'mean',
            'clustering_score': 'mean',
        })
        
        return quintile_stats
    
    def save_predictions(self, filepath: str):
        """Save predictions to CSV"""
        if self.predictions is None:
            self.generate_predictions()
        
        self.predictions.to_csv(filepath, index=False)
        logger.info(f"Predictions saved to {filepath}")
    
    def get_summary_stats(self) -> Dict:
        """
        Get summary statistics of EPM predictions.
        
        Returns:
            Dictionary with summary metrics
        """
        if self.predictions is None:
            self.generate_predictions()
        
        stats = {
            'num_tickers': len(self.predictions),
            'mean_adjustment_pct': self.predictions['adjustment_pct'].mean(),
            'std_adjustment_pct': self.predictions['adjustment_pct'].std(),
            'max_bullish_adjustment': self.predictions['adjustment_pct'].max(),
            'max_bearish_adjustment': self.predictions['adjustment_pct'].min(),
            'num_bullish': (self.predictions['adjustment_pct'] > 0).sum(),
            'num_bearish': (self.predictions['adjustment_pct'] < 0).sum(),
            'num_neutral': (self.predictions['adjustment_pct'] == 0).sum(),
            'mean_consensus': self.predictions['consensus_estimate'].mean(),
            'mean_epm_prediction': self.predictions['epm_prediction'].mean(),
            'mean_clustering_score': self.predictions['clustering_score'].mean(),
            'mean_beat_rate': self.predictions['beat_rate'].mean(),
        }
        
        return stats


if __name__ == "__main__":
    # Test EPM model
    import sys
    from pathlib import Path
    
    project_root = Path(__file__).parent.parent.parent
    sys.path.insert(0, str(project_root))
    
    # Load features
    features_path = project_root / 'data' / 'processed' / 'epm_features.csv'
    features_df = pd.read_csv(features_path)
    
    # Initialize and run model
    model = EPMModel(features_df)
    predictions = model.generate_predictions()
    
    print("\n" + "="*60)
    print("EPM PREDICTIONS")
    print("="*60)
    print(predictions[['ticker', 'consensus_estimate', 'epm_prediction', 
                       'adjustment_pct', 'adjustment_amount']].to_string())
    
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    stats = model.get_summary_stats()
    for key, value in stats.items():
        print(f"{key:30s}: {value}")
    
    print("\n" + "="*60)
    print("QUINTILE ANALYSIS")
    print("="*60)
    quintile_stats = model.get_quintile_analysis()
    print(quintile_stats)
    
    print("\n" + "="*60)
    print("TOP 5 BULLISH (EPM > Consensus)")
    print("="*60)
    ranked = model.rank_predictions('adjustment_pct')
    print(ranked[['ticker', 'sector', 'consensus_estimate', 'epm_prediction', 
                  'adjustment_pct']].head().to_string())
    
    print("\n" + "="*60)
    print("TOP 5 BEARISH (EPM < Consensus)")
    print("="*60)
    print(ranked[['ticker', 'sector', 'consensus_estimate', 'epm_prediction', 
                  'adjustment_pct']].tail().to_string())
    
    # Save
    output_path = project_root / 'data' / 'processed' / 'epm_predictions.csv'
    model.save_predictions(output_path)

