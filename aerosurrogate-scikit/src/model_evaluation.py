"""
Model evaluation utilities with aerodynamic-specific validation.

This module provides comprehensive evaluation tools for aerodynamic surrogate models,
including physics-informed validation, uncertainty quantification, and performance
assessment specifically designed for fluid mechanics applications.

Authors: ML Engineering Team
Date: 2025
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any, Optional, Union
import warnings
from pathlib import Path
import logging

# Scientific computing
from scipy import stats
from scipy.stats import pearsonr, spearmanr, normaltest

# Scikit-learn imports
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    mean_absolute_percentage_error, explained_variance_score,
    max_error
)
from sklearn.model_selection import (
    cross_val_score, cross_validate, learning_curve,
    validation_curve, permutation_test_score
)
from sklearn.preprocessing import StandardScaler
from sklearn.base import clone

# Statistical tests
from scipy.stats import (
    jarque_bera, shapiro, anderson, kstest,
    levene, bartlett, chi2_contingency
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings('ignore', category=FutureWarning)


class AerodynamicModelEvaluator:
    """
    Comprehensive evaluation framework for aerodynamic surrogate models.
    
    Features:
    - Standard regression metrics with aerodynamic interpretation
    - Physics-informed validation against fluid mechanics principles
    - Uncertainty quantification and confidence intervals
    - Residual analysis and diagnostic plots
    - Cross-validation with domain-aware stratification
    - Feature importance and sensitivity analysis
    - Model comparison and ranking
    """
    
    def __init__(self, random_state: int = 42):
        """
        Initialize the aerodynamic model evaluator.
        
        Args:
            random_state: Random seed for reproducibility
        """
        self.random_state = random_state
        self.evaluation_results = {}
        self.physics_validation_results = {}
        self.uncertainty_results = {}
        
        # Define aerodynamic parameter bounds and expected relationships
        self.physical_bounds = {
            'cd': (0.1, 1.0),      # Drag coefficient reasonable range
            'cl': (-2.0, 2.0),     # Lift coefficient reasonable range
            'cs': (-0.5, 0.5),     # Side force coefficient
            'cmy': (-0.5, 0.5),    # Pitching moment coefficient
        }
        
        self.expected_relationships = {
            ('frontal_area', 'cd'): 'positive',    # Larger area → higher drag
            ('clearance', 'cl'): 'negative',       # Lower clearance → more downforce
            ('ratio_height_fast_back', 'cd'): 'positive',  # Steeper fastback → higher drag
            ('bottom_taper_angle', 'cl'): 'negative',      # Larger diffuser angle → more downforce
        }
        
        logger.info("Initialized AerodynamicModelEvaluator")
    
    def calculate_comprehensive_metrics(self, 
                                      y_true: np.ndarray, 
                                      y_pred: np.ndarray,
                                      target_names: Optional[List[str]] = None) -> Dict[str, float]:
        """
        Calculate comprehensive regression metrics for aerodynamic predictions.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            target_names: Names of target variables
            
        Returns:
            Dictionary of comprehensive metrics
        """
        metrics = {}
        
        # Handle multi-target case
        if y_true.ndim > 1 and y_true.shape[1] > 1:
            if target_names is None:
                target_names = [f'target_{i}' for i in range(y_true.shape[1])]
            
            for i, target_name in enumerate(target_names):
                y_true_single = y_true[:, i]
                y_pred_single = y_pred[:, i]
                
                # Standard metrics
                metrics[f'{target_name}_r2'] = r2_score(y_true_single, y_pred_single)
                metrics[f'{target_name}_rmse'] = np.sqrt(mean_squared_error(y_true_single, y_pred_single))
                metrics[f'{target_name}_mae'] = mean_absolute_error(y_true_single, y_pred_single)
                metrics[f'{target_name}_mape'] = mean_absolute_percentage_error(y_true_single, y_pred_single)
                metrics[f'{target_name}_explained_var'] = explained_variance_score(y_true_single, y_pred_single)
                metrics[f'{target_name}_max_error'] = max_error(y_true_single, y_pred_single)
                
                # Aerodynamic-specific metrics
                metrics[f'{target_name}_mean_error'] = np.mean(y_pred_single - y_true_single)
                metrics[f'{target_name}_std_error'] = np.std(y_pred_single - y_true_single)
                
                # Relative metrics
                y_range = np.max(y_true_single) - np.min(y_true_single)
                metrics[f'{target_name}_normalized_rmse'] = metrics[f'{target_name}_rmse'] / y_range
                metrics[f'{target_name}_normalized_mae'] = metrics[f'{target_name}_mae'] / y_range
                
                # Physical validity (percentage of predictions within expected bounds)
                if target_name.lower() in self.physical_bounds:
                    bounds = self.physical_bounds[target_name.lower()]
                    within_bounds = np.sum((y_pred_single >= bounds[0]) & (y_pred_single <= bounds[1]))
                    metrics[f'{target_name}_physical_validity'] = within_bounds / len(y_pred_single)
                
                # Correlation between true and predicted
                metrics[f'{target_name}_correlation'] = pearsonr(y_true_single, y_pred_single)[0]
                
            # Overall metrics for multi-target
            metrics['overall_r2'] = np.mean([metrics[f'{name}_r2'] for name in target_names])
            metrics['overall_rmse'] = np.mean([metrics[f'{name}_rmse'] for name in target_names])
            
        else:
            # Single target
            if y_true.ndim > 1:
                y_true = y_true.ravel()
                y_pred = y_pred.ravel()
            
            target_name = target_names[0] if target_names else 'target'
            
            # Standard metrics
            metrics['r2'] = r2_score(y_true, y_pred)
            metrics['rmse'] = np.sqrt(mean_squared_error(y_true, y_pred))
            metrics['mae'] = mean_absolute_error(y_true, y_pred)
            metrics['mape'] = mean_absolute_percentage_error(y_true, y_pred)
            metrics['explained_var'] = explained_variance_score(y_true, y_pred)
            metrics['max_error'] = max_error(y_true, y_pred)
            
            # Aerodynamic-specific metrics
            metrics['mean_error'] = np.mean(y_pred - y_true)
            metrics['std_error'] = np.std(y_pred - y_true)
            
            # Relative metrics
            y_range = np.max(y_true) - np.min(y_true)
            metrics['normalized_rmse'] = metrics['rmse'] / y_range
            metrics['normalized_mae'] = metrics['mae'] / y_range
            
            # Physical validity
            if target_name.lower() in self.physical_bounds:
                bounds = self.physical_bounds[target_name.lower()]
                within_bounds = np.sum((y_pred >= bounds[0]) & (y_pred <= bounds[1]))
                metrics['physical_validity'] = within_bounds / len(y_pred)
            
            # Correlation
            metrics['correlation'] = pearsonr(y_true, y_pred)[0]
        
        return metrics
    
    def validate_aerodynamic_physics(self, 
                                   model,
                                   X: np.ndarray,
                                   y_true: np.ndarray,
                                   feature_names: Optional[List[str]] = None,
                                   target_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Validate model predictions against aerodynamic physics principles.
        
        Args:
            model: Trained model
            X: Feature matrix
            y_true: True target values
            feature_names: Names of features
            target_names: Names of targets
            
        Returns:
            Dictionary of physics validation results
        """
        y_pred = model.predict(X)
        
        validation = {
            'monotonicity_tests': {},
            'correlation_tests': {},
            'physical_bounds_tests': {},
            'consistency_tests': {},
            'summary': {}
        }
        
        # Convert to DataFrame for easier manipulation
        if feature_names is None:
            feature_names = [f'feature_{i}' for i in range(X.shape[1])]
        
        if hasattr(X, 'columns'):
            X_df = X
        else:
            X_df = pd.DataFrame(X, columns=feature_names)
        
        if target_names is None:
            if y_true.ndim > 1 and y_true.shape[1] > 1:
                target_names = [f'target_{i}' for i in range(y_true.shape[1])]
            else:
                target_names = ['target']
        
        # Test monotonicity relationships
        monotonicity_score = 0
        total_monotonicity_tests = 0
        
        for (feature, target), expected_direction in self.expected_relationships.items():
            if feature in X_df.columns:
                # Find corresponding target
                target_idx = None
                if target in target_names:
                    target_idx = target_names.index(target)
                elif target.lower() in [name.lower() for name in target_names]:
                    target_idx = [name.lower() for name in target_names].index(target.lower())
                
                if target_idx is not None:
                    if y_pred.ndim > 1:
                        pred_values = y_pred[:, target_idx]
                    else:
                        pred_values = y_pred
                    
                    corr = pearsonr(X_df[feature], pred_values)[0]
                    expected_positive = expected_direction == 'positive'
                    is_correct = (corr > 0) == expected_positive
                    
                    validation['monotonicity_tests'][f'{feature}_{target}'] = {
                        'correlation': corr,
                        'expected_direction': expected_direction,
                        'is_physically_correct': is_correct,
                        'confidence': abs(corr)
                    }
                    
                    if is_correct:
                        monotonicity_score += 1
                    total_monotonicity_tests += 1
        
        # Test physical bounds
        bounds_score = 0
        total_bounds_tests = 0
        
        for i, target_name in enumerate(target_names):
            if target_name.lower() in self.physical_bounds:
                bounds = self.physical_bounds[target_name.lower()]
                
                if y_pred.ndim > 1:
                    pred_values = y_pred[:, i]
                else:
                    pred_values = y_pred
                
                within_bounds = np.sum((pred_values >= bounds[0]) & (pred_values <= bounds[1]))
                validity_ratio = within_bounds / len(pred_values)
                
                validation['physical_bounds_tests'][target_name] = {
                    'bounds': bounds,
                    'min_predicted': pred_values.min(),
                    'max_predicted': pred_values.max(),
                    'within_bounds_count': within_bounds,
                    'total_predictions': len(pred_values),
                    'validity_ratio': validity_ratio,
                    'is_acceptable': validity_ratio > 0.9  # 90% should be within bounds
                }
                
                if validity_ratio > 0.9:
                    bounds_score += 1
                total_bounds_tests += 1
        
        # Test prediction consistency (variance should be reasonable)
        consistency_score = 0
        total_consistency_tests = 0
        
        for i, target_name in enumerate(target_names):
            if y_pred.ndim > 1:
                pred_values = y_pred[:, i]
                true_values = y_true[:, i]
            else:
                pred_values = y_pred
                true_values = y_true
            
            pred_cv = np.std(pred_values) / np.mean(np.abs(pred_values))
            true_cv = np.std(true_values) / np.mean(np.abs(true_values))
            
            cv_ratio = pred_cv / true_cv if true_cv > 0 else float('inf')
            is_consistent = 0.5 <= cv_ratio <= 2.0  # Predicted variance within 2x of true
            
            validation['consistency_tests'][target_name] = {
                'predicted_cv': pred_cv,
                'true_cv': true_cv,
                'cv_ratio': cv_ratio,
                'is_consistent': is_consistent
            }
            
            if is_consistent:
                consistency_score += 1
            total_consistency_tests += 1
        
        # Calculate overall physics compliance score
        total_tests = total_monotonicity_tests + total_bounds_tests + total_consistency_tests
        total_passed = monotonicity_score + bounds_score + consistency_score
        
        validation['summary'] = {
            'monotonicity_score': monotonicity_score / max(total_monotonicity_tests, 1),
            'bounds_score': bounds_score / max(total_bounds_tests, 1),
            'consistency_score': consistency_score / max(total_consistency_tests, 1),
            'overall_physics_score': total_passed / max(total_tests, 1),
            'total_tests': total_tests,
            'tests_passed': total_passed
        }
        
        return validation
    
    def perform_residual_analysis(self, 
                                y_true: np.ndarray, 
                                y_pred: np.ndarray,
                                feature_matrix: Optional[np.ndarray] = None,
                                target_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Perform comprehensive residual analysis.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            feature_matrix: Feature matrix for dependency analysis
            target_names: Names of targets
            
        Returns:
            Dictionary of residual analysis results
        """
        analysis = {}
        
        # Handle multi-target case
        if y_true.ndim > 1 and y_true.shape[1] > 1:
            if target_names is None:
                target_names = [f'target_{i}' for i in range(y_true.shape[1])]
            
            for i, target_name in enumerate(target_names):
                y_true_single = y_true[:, i]
                y_pred_single = y_pred[:, i]
                residuals = y_pred_single - y_true_single
                
                analysis[target_name] = self._analyze_single_target_residuals(
                    residuals, y_pred_single, feature_matrix
                )
        else:
            # Single target
            if y_true.ndim > 1:
                y_true = y_true.ravel()
                y_pred = y_pred.ravel()
            
            residuals = y_pred - y_true
            target_name = target_names[0] if target_names else 'target'
            
            analysis[target_name] = self._analyze_single_target_residuals(
                residuals, y_pred, feature_matrix
            )
        
        return analysis
    
    def _analyze_single_target_residuals(self, 
                                       residuals: np.ndarray,
                                       predictions: np.ndarray,
                                       feature_matrix: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Analyze residuals for a single target variable.
        """
        analysis = {}
        
        # Basic statistics
        analysis['statistics'] = {
            'mean': np.mean(residuals),
            'std': np.std(residuals),
            'min': np.min(residuals),
            'max': np.max(residuals),
            'median': np.median(residuals),
            'q25': np.percentile(residuals, 25),
            'q75': np.percentile(residuals, 75),
            'skewness': stats.skew(residuals),
            'kurtosis': stats.kurtosis(residuals)
        }
        
        # Normality tests
        analysis['normality_tests'] = {}
        
        if len(residuals) >= 8:  # Minimum for Shapiro-Wilk
            shapiro_stat, shapiro_p = shapiro(residuals)
            analysis['normality_tests']['shapiro_wilk'] = {
                'statistic': shapiro_stat,
                'p_value': shapiro_p,
                'is_normal': shapiro_p > 0.05
            }
        
        if len(residuals) >= 20:  # Minimum for Jarque-Bera
            jb_stat, jb_p = jarque_bera(residuals)
            analysis['normality_tests']['jarque_bera'] = {
                'statistic': jb_stat,
                'p_value': jb_p,
                'is_normal': jb_p > 0.05
            }
        
        # Heteroscedasticity analysis
        analysis['heteroscedasticity'] = {}
        
        # Correlation between absolute residuals and predictions
        abs_residuals = np.abs(residuals)
        het_corr = pearsonr(predictions, abs_residuals)[0]
        analysis['heteroscedasticity']['correlation_with_predictions'] = het_corr
        analysis['heteroscedasticity']['has_heteroscedasticity'] = abs(het_corr) > 0.2
        
        # Outlier detection
        analysis['outliers'] = {}
        z_scores = np.abs(stats.zscore(residuals))
        outliers_z = np.sum(z_scores > 3)
        
        q1, q3 = np.percentile(residuals, [25, 75])
        iqr = q3 - q1
        outliers_iqr = np.sum((residuals < (q1 - 1.5 * iqr)) | (residuals > (q3 + 1.5 * iqr)))
        
        analysis['outliers'] = {
            'z_score_outliers': outliers_z,
            'iqr_outliers': outliers_iqr,
            'outlier_percentage_z': outliers_z / len(residuals) * 100,
            'outlier_percentage_iqr': outliers_iqr / len(residuals) * 100
        }
        
        # Independence test (Durbin-Watson if ordered data)
        if len(residuals) > 10:
            # Simple runs test for randomness
            median_residual = np.median(residuals)
            runs, n1, n2 = self._runs_test(residuals > median_residual)
            expected_runs = (2 * n1 * n2) / (n1 + n2) + 1
            variance_runs = (2 * n1 * n2 * (2 * n1 * n2 - n1 - n2)) / ((n1 + n2) ** 2 * (n1 + n2 - 1))
            
            if variance_runs > 0:
                z_runs = (runs - expected_runs) / np.sqrt(variance_runs)
                p_runs = 2 * (1 - stats.norm.cdf(abs(z_runs)))
                
                analysis['independence'] = {
                    'runs_test_statistic': z_runs,
                    'runs_test_p_value': p_runs,
                    'is_independent': p_runs > 0.05
                }
        
        return analysis
    
    def _runs_test(self, binary_sequence: np.ndarray) -> Tuple[int, int, int]:
        """
        Perform runs test for independence.
        
        Returns:
            (number_of_runs, n1, n2)
        """
        runs = 1
        n1 = np.sum(binary_sequence)
        n2 = len(binary_sequence) - n1
        
        for i in range(1, len(binary_sequence)):
            if binary_sequence[i] != binary_sequence[i-1]:
                runs += 1
        
        return runs, n1, n2
    
    def quantify_uncertainty(self, 
                           model,
                           X_train: np.ndarray,
                           y_train: np.ndarray,
                           X_test: np.ndarray,
                           y_test: np.ndarray,
                           n_bootstrap: int = 100,
                           confidence_level: float = 0.95) -> Dict[str, Any]:
        """
        Quantify model uncertainty using bootstrap sampling.
        
        Args:
            model: Trained model
            X_train: Training features
            y_train: Training targets
            X_test: Test features
            y_test: Test targets
            n_bootstrap: Number of bootstrap samples
            confidence_level: Confidence level for intervals
            
        Returns:
            Dictionary of uncertainty quantification results
        """
        logger.info(f"Quantifying uncertainty with {n_bootstrap} bootstrap samples...")
        
        n_samples = len(X_train)
        predictions_bootstrap = []
        metrics_bootstrap = []
        
        # Bootstrap sampling
        for i in range(n_bootstrap):
            # Sample with replacement
            bootstrap_indices = np.random.choice(n_samples, size=n_samples, replace=True)
            X_boot = X_train[bootstrap_indices]
            y_boot = y_train[bootstrap_indices]
            
            # Train model on bootstrap sample
            model_boot = clone(model)
            
            # Handle different target formats
            if y_boot.ndim > 1 and y_boot.shape[1] == 1:
                y_boot_fit = y_boot.ravel()
            else:
                y_boot_fit = y_boot
            
            try:
                model_boot.fit(X_boot, y_boot_fit)
                
                # Make predictions
                y_pred_boot = model_boot.predict(X_test)
                predictions_bootstrap.append(y_pred_boot)
                
                # Calculate metrics
                if y_test.ndim > 1 and y_test.shape[1] == 1:
                    y_test_eval = y_test.ravel()
                else:
                    y_test_eval = y_test
                
                metrics = self.calculate_comprehensive_metrics(y_test_eval, y_pred_boot)
                metrics_bootstrap.append(metrics)
                
            except Exception as e:
                logger.warning(f"Bootstrap iteration {i} failed: {str(e)}")
                continue
        
        if not predictions_bootstrap:
            logger.error("All bootstrap iterations failed")
            return {}
        
        # Convert to arrays
        predictions_bootstrap = np.array(predictions_bootstrap)
        
        # Calculate confidence intervals
        alpha = 1 - confidence_level
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100
        
        uncertainty_results = {
            'confidence_level': confidence_level,
            'n_bootstrap_samples': len(predictions_bootstrap),
            'prediction_intervals': {},
            'metric_intervals': {}
        }
        
        # Prediction intervals
        pred_mean = np.mean(predictions_bootstrap, axis=0)
        pred_std = np.std(predictions_bootstrap, axis=0)
        pred_lower = np.percentile(predictions_bootstrap, lower_percentile, axis=0)
        pred_upper = np.percentile(predictions_bootstrap, upper_percentile, axis=0)
        
        uncertainty_results['prediction_intervals'] = {
            'mean': pred_mean,
            'std': pred_std,
            'lower_bound': pred_lower,
            'upper_bound': pred_upper,
            'interval_width': pred_upper - pred_lower
        }
        
        # Metric intervals
        if metrics_bootstrap:
            metrics_df = pd.DataFrame(metrics_bootstrap)
            
            for metric in metrics_df.columns:
                values = metrics_df[metric].dropna().values
                if len(values) > 0:
                    uncertainty_results['metric_intervals'][metric] = {
                        'mean': np.mean(values),
                        'std': np.std(values),
                        'lower_bound': np.percentile(values, lower_percentile),
                        'upper_bound': np.percentile(values, upper_percentile),
                        'interval_width': np.percentile(values, upper_percentile) - np.percentile(values, lower_percentile)
                    }
        
        return uncertainty_results
    
    def create_diagnostic_plots(self, 
                              model,
                              X_test: np.ndarray,
                              y_test: np.ndarray,
                              target_names: Optional[List[str]] = None,
                              save_path: Optional[str] = None) -> None:
        """
        Create comprehensive diagnostic plots for model evaluation.
        
        Args:
            model: Trained model
            X_test: Test features
            y_test: Test targets
            target_names: Names of targets
            save_path: Path to save plots
        """
        y_pred = model.predict(X_test)
        
        # Determine number of targets
        if y_test.ndim > 1 and y_test.shape[1] > 1:
            n_targets = y_test.shape[1]
            if target_names is None:
                target_names = [f'Target_{i+1}' for i in range(n_targets)]
        else:
            n_targets = 1
            if y_test.ndim > 1:
                y_test = y_test.ravel()
                y_pred = y_pred.ravel()
            if target_names is None:
                target_names = ['Target']
        
        # Create subplots
        fig_rows = 2 if n_targets == 1 else 3
        fig_cols = 2 if n_targets == 1 else n_targets
        fig_size = (6 * fig_cols, 5 * fig_rows)
        
        fig, axes = plt.subplots(fig_rows, fig_cols, figsize=fig_size)
        if n_targets == 1:
            axes = axes.flatten()
        
        for i in range(n_targets):
            target_name = target_names[i]
            
            if n_targets > 1:
                y_true_single = y_test[:, i]
                y_pred_single = y_pred[:, i]
                ax_offset = i
            else:
                y_true_single = y_test
                y_pred_single = y_pred
                ax_offset = 0
            
            residuals = y_pred_single - y_true_single
            
            # Plot 1: Prediction vs Actual
            if n_targets == 1:
                ax1 = axes[0]
            else:
                ax1 = axes[0, ax_offset]
            
            ax1.scatter(y_true_single, y_pred_single, alpha=0.6, s=30)
            ax1.plot([y_true_single.min(), y_true_single.max()], 
                    [y_true_single.min(), y_true_single.max()], 'r--', lw=2)
            
            r2 = r2_score(y_true_single, y_pred_single)
            rmse = np.sqrt(mean_squared_error(y_true_single, y_pred_single))
            
            ax1.set_xlabel(f'True {target_name}')
            ax1.set_ylabel(f'Predicted {target_name}')
            ax1.set_title(f'{target_name} - Prediction vs Actual\\nR² = {r2:.4f}, RMSE = {rmse:.4f}')
            ax1.grid(True, alpha=0.3)
            
            # Add error bands
            perfect_line = np.linspace(y_true_single.min(), y_true_single.max(), 100)
            error_margin = 0.1  # 10% error margin
            ax1.fill_between(perfect_line, perfect_line*(1-error_margin), perfect_line*(1+error_margin), 
                           alpha=0.2, color='gray', label='±10% Error Band')
            ax1.legend()
            
            # Plot 2: Residuals vs Predicted
            if n_targets == 1:
                ax2 = axes[1]
            else:
                ax2 = axes[1, ax_offset]
            
            ax2.scatter(y_pred_single, residuals, alpha=0.6, s=30)
            ax2.axhline(y=0, color='r', linestyle='--')
            ax2.set_xlabel(f'Predicted {target_name}')
            ax2.set_ylabel('Residuals')
            ax2.set_title(f'{target_name} - Residual Analysis')
            ax2.grid(True, alpha=0.3)
            
            # Add statistical info
            mean_residual = np.mean(residuals)
            std_residual = np.std(residuals)
            ax2.text(0.05, 0.95, f'Mean: {mean_residual:.4f}\\nStd: {std_residual:.4f}', 
                    transform=ax2.transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            # Plot 3: Residual distribution (only for multi-target)
            if n_targets > 1:
                ax3 = axes[2, ax_offset]
                ax3.hist(residuals, bins=20, alpha=0.7, density=True, edgecolor='black')
                ax3.axvline(0, color='r', linestyle='--', label='Zero Error')
                
                # Overlay normal distribution
                x_norm = np.linspace(residuals.min(), residuals.max(), 100)
                normal_dist = stats.norm.pdf(x_norm, mean_residual, std_residual)
                ax3.plot(x_norm, normal_dist, 'r-', linewidth=2, label='Normal Fit')
                
                ax3.set_xlabel('Residuals')
                ax3.set_ylabel('Density')
                ax3.set_title(f'{target_name} - Residual Distribution')
                ax3.legend()
                ax3.grid(True, alpha=0.3)
        
        # Additional plots for single target
        if n_targets == 1:
            # Q-Q plot
            ax3 = axes[2]
            stats.probplot(residuals, dist="norm", plot=ax3)
            ax3.set_title('Q-Q Plot (Normality Test)')
            ax3.grid(True, alpha=0.3)
            
            # Residual distribution
            ax4 = axes[3]
            ax4.hist(residuals, bins=20, alpha=0.7, density=True, edgecolor='black')
            ax4.axvline(0, color='r', linestyle='--', label='Zero Error')
            
            # Overlay normal distribution
            x_norm = np.linspace(residuals.min(), residuals.max(), 100)
            normal_dist = stats.norm.pdf(x_norm, mean_residual, std_residual)
            ax4.plot(x_norm, normal_dist, 'r-', linewidth=2, label='Normal Fit')
            
            ax4.set_xlabel('Residuals')
            ax4.set_ylabel('Density')
            ax4.set_title('Residual Distribution')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.suptitle('Model Diagnostic Plots', y=0.98, fontsize=16, fontweight='bold')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Diagnostic plots saved to {save_path}")
        
        plt.show()
    
    def compare_models(self, 
                      models_dict: Dict[str, Any],
                      X_test: np.ndarray,
                      y_test: np.ndarray,
                      target_names: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Compare multiple models using comprehensive metrics.
        
        Args:
            models_dict: Dictionary of {model_name: model} pairs
            X_test: Test features
            y_test: Test targets
            target_names: Names of targets
            
        Returns:
            DataFrame with model comparison results
        """
        logger.info(f"Comparing {len(models_dict)} models...")
        
        comparison_results = []
        
        for model_name, model in models_dict.items():
            logger.info(f"  Evaluating {model_name}...")
            
            try:
                # Make predictions
                y_pred = model.predict(X_test)
                
                # Calculate metrics
                metrics = self.calculate_comprehensive_metrics(y_test, y_pred, target_names)
                
                # Add model name
                metrics['Model'] = model_name
                
                # Physics validation
                physics_validation = self.validate_aerodynamic_physics(
                    model, X_test, y_test, target_names=target_names
                )
                
                # Add physics score
                physics_score = physics_validation['summary']['overall_physics_score']
                metrics['Physics_Score'] = physics_score
                
                comparison_results.append(metrics)
                
            except Exception as e:
                logger.error(f"Error evaluating {model_name}: {str(e)}")
                continue
        
        if not comparison_results:
            logger.error("No models evaluated successfully")
            return pd.DataFrame()
        
        # Create comparison DataFrame
        comparison_df = pd.DataFrame(comparison_results)
        
        # Move Model column to front
        cols = ['Model'] + [col for col in comparison_df.columns if col != 'Model']
        comparison_df = comparison_df[cols]
        
        # Sort by primary metric (R² or overall R²)
        if 'overall_r2' in comparison_df.columns:
            sort_column = 'overall_r2'
        elif 'r2' in comparison_df.columns:
            sort_column = 'r2'
        else:
            # Find first R² column
            r2_columns = [col for col in comparison_df.columns if 'r2' in col.lower()]
            sort_column = r2_columns[0] if r2_columns else comparison_df.columns[1]
        
        comparison_df = comparison_df.sort_values(sort_column, ascending=False)
        
        logger.info(f"Model comparison complete. Best model: {comparison_df.iloc[0]['Model']}")
        
        return comparison_df
    
    def generate_evaluation_report(self, 
                                 model,
                                 X_test: np.ndarray,
                                 y_test: np.ndarray,
                                 model_name: str = "Model",
                                 target_names: Optional[List[str]] = None) -> str:
        """
        Generate comprehensive evaluation report.
        
        Args:
            model: Trained model
            X_test: Test features
            y_test: Test targets
            model_name: Name of the model
            target_names: Names of targets
            
        Returns:
            Formatted evaluation report
        """
        logger.info(f"Generating evaluation report for {model_name}...")
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        metrics = self.calculate_comprehensive_metrics(y_test, y_pred, target_names)
        
        # Physics validation
        physics_validation = self.validate_aerodynamic_physics(
            model, X_test, y_test, target_names=target_names
        )
        
        # Residual analysis
        residual_analysis = self.perform_residual_analysis(y_test, y_pred, target_names=target_names)
        
        # Generate report
        report = []
        report.append("=" * 80)
        report.append(f"AERODYNAMIC MODEL EVALUATION REPORT")
        report.append(f"Model: {model_name}")
        report.append("=" * 80)
        
        # Model performance
        report.append(f"\\n📊 MODEL PERFORMANCE:")
        
        if 'overall_r2' in metrics:
            report.append(f"  Overall R²: {metrics['overall_r2']:.4f}")
            report.append(f"  Overall RMSE: {metrics['overall_rmse']:.4f}")
        elif 'r2' in metrics:
            report.append(f"  R²: {metrics['r2']:.4f}")
            report.append(f"  RMSE: {metrics['rmse']:.4f}")
            report.append(f"  MAE: {metrics['mae']:.4f}")
            report.append(f"  MAPE: {metrics['mape']:.2f}%")
        
        # Target-specific metrics
        if target_names and len(target_names) > 1:
            report.append(f"\\n🎯 TARGET-SPECIFIC PERFORMANCE:")
            for target in target_names:
                if f'{target}_r2' in metrics:
                    report.append(f"  {target.upper()}:")
                    report.append(f"    R²: {metrics[f'{target}_r2']:.4f}")
                    report.append(f"    RMSE: {metrics[f'{target}_rmse']:.4f}")
                    report.append(f"    MAE: {metrics[f'{target}_mae']:.4f}")
        
        # Physics validation
        report.append(f"\\n🔬 PHYSICS VALIDATION:")
        physics_summary = physics_validation['summary']
        report.append(f"  Overall Physics Score: {physics_summary['overall_physics_score']:.3f}")
        report.append(f"  Monotonicity Score: {physics_summary['monotonicity_score']:.3f}")
        report.append(f"  Physical Bounds Score: {physics_summary['bounds_score']:.3f}")
        report.append(f"  Consistency Score: {physics_summary['consistency_score']:.3f}")
        report.append(f"  Tests Passed: {physics_summary['tests_passed']}/{physics_summary['total_tests']}")
        
        # Detailed physics tests
        report.append(f"\\n  Monotonicity Tests:")
        for test_name, result in physics_validation['monotonicity_tests'].items():
            status = "✅" if result['is_physically_correct'] else "⚠️"
            report.append(f"    {status} {test_name}: r={result['correlation']:.3f} ({result['expected_direction']})")
        
        # Physical bounds
        if physics_validation['physical_bounds_tests']:
            report.append(f"\\n  Physical Bounds Tests:")
            for target, result in physics_validation['physical_bounds_tests'].items():
                status = "✅" if result['is_acceptable'] else "⚠️"
                report.append(f"    {status} {target}: {result['validity_ratio']:.1%} within bounds {result['bounds']}")
        
        # Residual analysis summary
        report.append(f"\\n📈 RESIDUAL ANALYSIS:")
        
        for target_name, analysis in residual_analysis.items():
            if len(residual_analysis) > 1:
                report.append(f"\\n  {target_name.upper()}:")
                prefix = "    "
            else:
                prefix = "  "
            
            stats_data = analysis['statistics']
            report.append(f"{prefix}Mean Residual: {stats_data['mean']:.4f}")
            report.append(f"{prefix}Residual Std: {stats_data['std']:.4f}")
            report.append(f"{prefix}Residual Range: [{stats_data['min']:.4f}, {stats_data['max']:.4f}]")
            
            # Normality
            if 'normality_tests' in analysis and analysis['normality_tests']:
                normality_results = []
                for test_name, result in analysis['normality_tests'].items():
                    if result['is_normal']:
                        normality_results.append(f"{test_name}: ✅")
                    else:
                        normality_results.append(f"{test_name}: ❌")
                
                report.append(f"{prefix}Normality Tests: {', '.join(normality_results)}")
            
            # Heteroscedasticity
            if 'heteroscedasticity' in analysis:
                het_status = "⚠️ Detected" if analysis['heteroscedasticity']['has_heteroscedasticity'] else "✅ Not detected"
                report.append(f"{prefix}Heteroscedasticity: {het_status}")
            
            # Outliers
            if 'outliers' in analysis:
                outlier_pct = analysis['outliers']['outlier_percentage_z']
                report.append(f"{prefix}Outliers (Z-score): {outlier_pct:.1f}%")
        
        # Model quality assessment
        report.append(f"\\n💡 MODEL QUALITY ASSESSMENT:")
        
        # Overall assessment
        overall_score = 0
        assessment_count = 0
        
        if 'overall_r2' in metrics:
            r2_score = metrics['overall_r2']
        elif 'r2' in metrics:
            r2_score = metrics['r2']
        else:
            r2_score = 0
        
        if r2_score > 0.95:
            report.append(f"  🌟 Excellent predictive performance (R² > 0.95)")
            overall_score += 3
        elif r2_score > 0.90:
            report.append(f"  ✅ Good predictive performance (R² > 0.90)")
            overall_score += 2
        elif r2_score > 0.80:
            report.append(f"  🔶 Acceptable predictive performance (R² > 0.80)")
            overall_score += 1
        else:
            report.append(f"  ⚠️ Performance needs improvement (R² ≤ 0.80)")
        assessment_count += 1
        
        # Physics compliance
        physics_score = physics_summary['overall_physics_score']
        if physics_score > 0.8:
            report.append(f"  ✅ Excellent physics compliance")
            overall_score += 2
        elif physics_score > 0.6:
            report.append(f"  🔶 Good physics compliance")
            overall_score += 1
        else:
            report.append(f"  ⚠️ Physics compliance needs attention")
        assessment_count += 1
        
        # Overall recommendation
        avg_score = overall_score / assessment_count if assessment_count > 0 else 0
        
        report.append(f"\\n🎯 RECOMMENDATION:")
        if avg_score >= 2.5:
            report.append(f"  🚀 Model is ready for production deployment")
            report.append(f"  ✅ Excellent performance and physics compliance")
        elif avg_score >= 1.5:
            report.append(f"  📈 Model shows good performance")
            report.append(f"  🔧 Consider minor refinements before deployment")
        elif avg_score >= 0.5:
            report.append(f"  🔶 Model needs improvement")
            report.append(f"  📊 Consider additional training data or feature engineering")
        else:
            report.append(f"  ⚠️ Model requires significant improvement")
            report.append(f"  🔄 Review model architecture and training approach")
        
        report.append(f"\\n" + "=" * 80)
        report.append(f"Report generated for aerodynamic surrogate model evaluation")
        report.append("=" * 80)
        
        return "\\n".join(report)


# Convenience functions for quick evaluation
def quick_model_evaluation(model, X_test, y_test, model_name="Model", target_names=None):
    """
    Quickly evaluate a model with standard metrics and plots.
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test targets
        model_name: Name of the model
        target_names: Names of targets
        
    Returns:
        Dictionary with evaluation results
    """
    evaluator = AerodynamicModelEvaluator()
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    metrics = evaluator.calculate_comprehensive_metrics(y_test, y_pred, target_names)
    
    # Physics validation
    physics_validation = evaluator.validate_aerodynamic_physics(
        model, X_test, y_test, target_names=target_names
    )
    
    # Create diagnostic plots
    evaluator.create_diagnostic_plots(model, X_test, y_test, target_names)
    
    # Generate report
    report = evaluator.generate_evaluation_report(
        model, X_test, y_test, model_name, target_names
    )
    
    print(report)
    
    return {
        'metrics': metrics,
        'physics_validation': physics_validation,
        'report': report
    }


def compare_aerodynamic_models(models_dict, X_test, y_test, target_names=None):
    """
    Compare multiple aerodynamic models.
    
    Args:
        models_dict: Dictionary of {model_name: model} pairs
        X_test: Test features
        y_test: Test targets
        target_names: Names of targets
        
    Returns:
        DataFrame with comparison results
    """
    evaluator = AerodynamicModelEvaluator()
    comparison_df = evaluator.compare_models(models_dict, X_test, y_test, target_names)
    
    print("\\n" + "="*80)
    print("MODEL COMPARISON RESULTS")
    print("="*80)
    print(comparison_df.round(4))
    
    return comparison_df


if __name__ == "__main__":
    # Example usage
    print("AerodynamicModelEvaluator - Comprehensive evaluation tools for aerodynamic surrogate models")
    print("\\nFeatures:")
    print("  ✅ Standard and aerodynamic-specific metrics")
    print("  ✅ Physics-informed validation")
    print("  ✅ Uncertainty quantification")
    print("  ✅ Comprehensive residual analysis")
    print("  ✅ Diagnostic plots and visualizations")
    print("  ✅ Model comparison framework")
    print("  ✅ Detailed evaluation reports")