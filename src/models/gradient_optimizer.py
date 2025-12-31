"""
Gradient-based weight optimization for EPM model.
Optimizes signal weights to maximize backtest Sharpe ratio.
"""

import pandas as pd
import numpy as np
from typing import List, Dict
import logging
from scipy.optimize import minimize
from src.models.epm_model import EPMModel

logger = logging.getLogger(__name__)


class EPMGradientOptimizer:
    """
    Gradient descent optimizer for EPM signal weights.
    Objective: Maximize quintile spread / backtest Sharpe ratio.
    """
    
    def __init__(self, features_df: pd.DataFrame, backtest_data: Dict):
        self.features_df = features_df
        self.backtest_data = backtest_data
        self.optimal_weights = None
        
    def objective_function(self, weights: np.ndarray) -> float:
        """
        Objective to minimize: Negative Sharpe ratio of quintile returns.
        """
        # Create temporary model with test weights
        temp_model = EPMModel(self.features_df)
        temp_model.weights = {
            'recency': abs(weights[0]),
            'momentum': abs(weights[1]),
            'clustering': abs(weights[2]),
            'accuracy': abs(weights[3]),
            'analyst': abs(weights[4])
        }
        
        # Generate predictions
        predictions = temp_model.generate_predictions()
        
        # Backtest
        backtester = EPMBacktest(predictions, self.backtest_data)
        results = backtester.calculate_prediction_errors()
        
        if len(results) == 0:
            return 1000  # Penalty for no results
        
        # Calculate quintile spread (Q5 - Q1 win rate)
        q5_win = results[results['quintile'] == 'Q5 (Most Bullish)']['epm_better'].mean()
        q1_win = results[results['quintile'] == 'Q1 (Most Bearish)']['epm_better'].mean()
        
        quintile_spread = q5_win - q1_win
        
        # Overall improvement
        mean_improvement = results['error_reduction_pct'].mean()
        
        # Combined score (higher = better)
        score = quintile_spread * 0.7 + mean_improvement * 0.3
        
        # Return negative for minimization
        return -score
    
    def optimize_weights(self, initial_weights: List[float] = None) -> Dict:
        """
        Run gradient descent optimization.
        """
        if initial_weights is None:
            initial_weights = [0.4, 0.2, 0.2, 0.2, 0.0]  # Equal start
        
        # Bounds: weights between 0 and 1, sum to 1
        bounds = [(0, 1)] * len(initial_weights)
        constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
        
        logger.info("Starting gradient optimization...")
        result = minimize(
            self.objective_function,
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'disp': True, 'maxiter': 100}
        )
        
        optimal_weights = {
            'recency': result.x[0],
            'momentum': result.x[1],
            'clustering': result.x[2],
            'accuracy': result.x[3],
            'analyst': result.x[4]
        }
        
        self.optimal_weights = optimal_weights
        logger.info(f"Optimization complete! Score: {-result.fun:.4f}")
        
        return optimal_weights


if __name__ == "__main__":
    # Test optimizer
    from src.data.collector import EPMDataCollector
    
    # Load data
    data, errors, metadata = EPMDataCollector.load('data/raw/epm_sp500_top50.pkl')
    features = pd.read_csv('data/processed/epm_features_top50.csv')
    
    optimizer = EPMGradientOptimizer(features, data)
    optimal = optimizer.optimize_weights()
    
    print("\nOptimal Weights:")
    for key, value in optimal.items():
        print(f"  {key:12s}: {value:.4f}")

