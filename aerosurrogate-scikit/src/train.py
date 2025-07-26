"""
Comprehensive Machine Learning Training Script for Aerodynamic Surrogate Models.

This script implements a complete ML pipeline for predicting drag (Cd) and lift (Cl) 
coefficients using the Windsor body aerodynamic dataset. It includes multiple regression
algorithms, hyperparameter optimization, and aerodynamic-specific validation.

Authors: ML Engineering Team
Date: 2025
"""

import numpy as np
import pandas as pd
import pickle
import json
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Union
import logging
from datetime import datetime
import argparse

# Scientific computing
from scipy import stats
from scipy.stats import randint, uniform

# Scikit-learn core
from sklearn.model_selection import (
    train_test_split, cross_val_score, GridSearchCV, 
    RandomizedSearchCV, StratifiedKFold, KFold
)
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    mean_absolute_percentage_error, explained_variance_score
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.base import TransformerMixin

# Regression algorithms
from sklearn.linear_model import (
    LinearRegression, Ridge, Lasso, ElasticNet, 
    BayesianRidge, HuberRegressor
)
from sklearn.ensemble import (
    RandomForestRegressor, GradientBoostingRegressor,
    ExtraTreesRegressor, AdaBoostRegressor, BaggingRegressor
)
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor

# XGBoost and LightGBM (if available)
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    
try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

# Plotting
import matplotlib.pyplot as plt
import seaborn as sns

# Project imports
from data_processing import (
    WindsorDataLoader, AerodynamicPreprocessor,
    create_drag_preprocessor, create_lift_preprocessor,
    create_multi_target_preprocessor
)
from config import PROJECT_ROOT, MODELS_DIR, RANDOM_STATE, CV_FOLDS

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(Path(PROJECT_ROOT) / 'training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)


class AerodynamicModelTrainer:
    """
    Comprehensive ML trainer for aerodynamic surrogate models.
    
    Features:
    - Multiple regression algorithms with proper hyperparameter tuning
    - Aerodynamic-specific evaluation metrics and validation
    - Physics-informed model assessment
    - Model persistence and comparison framework
    - Cross-validation with proper stratification
    - Feature importance analysis
    """
    
    def __init__(self, 
                 target_type: str = 'both',
                 cv_folds: int = 5,
                 test_size: float = 0.2,
                 random_state: int = RANDOM_STATE,
                 models_dir: str = MODELS_DIR,
                 enable_advanced_models: bool = True):
        """
        Initialize the aerodynamic model trainer.
        
        Args:
            target_type: 'drag', 'lift', or 'both'
            cv_folds: Number of cross-validation folds
            test_size: Test set proportion
            random_state: Random seed for reproducibility
            models_dir: Directory to save trained models
            enable_advanced_models: Whether to include XGBoost/LightGBM if available
        """
        self.target_type = target_type
        self.cv_folds = cv_folds
        self.test_size = test_size
        self.random_state = random_state
        self.models_dir = Path(models_dir)
        self.enable_advanced_models = enable_advanced_models
        
        # Create models directory
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize storage for results
        self.models = {}
        self.preprocessors = {}
        self.results = {}
        self.cv_results = {}
        self.feature_importance = {}
        
        # Data storage
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.feature_names = None
        
        logger.info(f"Initialized AerodynamicModelTrainer for {target_type} prediction")
        
    def define_model_configurations(self) -> Dict[str, Dict]:
        """
        Define model configurations with hyperparameter search spaces.
        
        Returns:
            Dictionary of model configurations
        """
        configs = {
            # Linear Models
            'linear_regression': {
                'model': LinearRegression(),
                'params': {},
                'search_type': 'grid'
            },
            
            'ridge': {
                'model': Ridge(random_state=self.random_state),
                'params': {
                    'alpha': [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
                },
                'search_type': 'grid'
            },
            
            'lasso': {
                'model': Lasso(random_state=self.random_state, max_iter=2000),
                'params': {
                    'alpha': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
                },
                'search_type': 'grid'
            },
            
            'elastic_net': {
                'model': ElasticNet(random_state=self.random_state, max_iter=2000),
                'params': {
                    'alpha': [0.001, 0.01, 0.1, 1.0, 10.0],
                    'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9]
                },
                'search_type': 'grid'
            },
            
            'bayesian_ridge': {
                'model': BayesianRidge(),
                'params': {
                    'alpha_1': [1e-6, 1e-5, 1e-4],
                    'alpha_2': [1e-6, 1e-5, 1e-4],
                    'lambda_1': [1e-6, 1e-5, 1e-4],
                    'lambda_2': [1e-6, 1e-5, 1e-4]
                },
                'search_type': 'random',
                'n_iter': 20
            },
            
            # Tree-based Models
            'random_forest': {
                'model': RandomForestRegressor(random_state=self.random_state),
                'params': {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [3, 5, 10, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4],
                    'max_features': ['sqrt', 'log2', None]
                },
                'search_type': 'random',
                'n_iter': 50
            },
            
            'extra_trees': {
                'model': ExtraTreesRegressor(random_state=self.random_state),
                'params': {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [3, 5, 10, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                },
                'search_type': 'random',
                'n_iter': 30
            },
            
            'gradient_boosting': {
                'model': GradientBoostingRegressor(random_state=self.random_state),
                'params': {
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'max_depth': [3, 5, 7],
                    'subsample': [0.8, 0.9, 1.0],
                    'min_samples_split': [2, 5, 10]
                },
                'search_type': 'random',
                'n_iter': 30
            },
            
            # Support Vector Regression
            'svr_rbf': {
                'model': SVR(kernel='rbf'),
                'params': {
                    'C': [0.1, 1, 10, 100],
                    'gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
                    'epsilon': [0.01, 0.1, 0.2]
                },
                'search_type': 'random',
                'n_iter': 25
            },
            
            'svr_linear': {
                'model': SVR(kernel='linear'),
                'params': {
                    'C': [0.1, 1, 10, 100],
                    'epsilon': [0.01, 0.1, 0.2]
                },
                'search_type': 'grid'
            },
            
            # Neural Networks
            'mlp_regressor': {
                'model': MLPRegressor(
                    random_state=self.random_state,
                    max_iter=1000,
                    early_stopping=True,
                    validation_fraction=0.2
                ),
                'params': {
                    'hidden_layer_sizes': [
                        (50,), (100,), (50, 25), (100, 50), 
                        (100, 50, 25), (150, 75)
                    ],
                    'activation': ['relu', 'tanh'],
                    'alpha': [0.0001, 0.001, 0.01],
                    'learning_rate': ['adaptive', 'constant']
                },
                'search_type': 'random',
                'n_iter': 20
            },
            
            # Other algorithms
            'knn': {
                'model': KNeighborsRegressor(),
                'params': {
                    'n_neighbors': [3, 5, 7, 10, 15],
                    'weights': ['uniform', 'distance'],
                    'metric': ['euclidean', 'manhattan', 'minkowski']
                },
                'search_type': 'grid'
            },
            
            'decision_tree': {
                'model': DecisionTreeRegressor(random_state=self.random_state),
                'params': {
                    'max_depth': [3, 5, 10, 15, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4],
                    'max_features': ['sqrt', 'log2', None]
                },
                'search_type': 'grid'
            }
        }
        
        # Add advanced models if available and enabled
        if self.enable_advanced_models:
            if XGBOOST_AVAILABLE:
                configs['xgboost'] = {
                    'model': xgb.XGBRegressor(
                        random_state=self.random_state,
                        objective='reg:squarederror'
                    ),
                    'params': {
                        'n_estimators': [50, 100, 200],
                        'learning_rate': [0.01, 0.1, 0.2],
                        'max_depth': [3, 5, 7],
                        'subsample': [0.8, 0.9, 1.0],
                        'colsample_bytree': [0.8, 0.9, 1.0]
                    },
                    'search_type': 'random',
                    'n_iter': 30
                }
                
            if LIGHTGBM_AVAILABLE:
                configs['lightgbm'] = {
                    'model': lgb.LGBMRegressor(
                        random_state=self.random_state,
                        objective='regression',
                        verbosity=-1
                    ),
                    'params': {
                        'n_estimators': [50, 100, 200],
                        'learning_rate': [0.01, 0.1, 0.2],
                        'max_depth': [3, 5, 7, -1],
                        'num_leaves': [31, 50, 100],
                        'subsample': [0.8, 0.9, 1.0],
                        'colsample_bytree': [0.8, 0.9, 1.0]
                    },
                    'search_type': 'random',
                    'n_iter': 30
                }
        
        return configs
    
    def load_and_preprocess_data(self, data_dir: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load and preprocess the Windsor body dataset.
        
        Args:
            data_dir: Data directory path (optional)
            
        Returns:
            Tuple of (X_processed, y_processed)
        """
        logger.info("Loading and preprocessing Windsor body dataset...")
        
        # Load data
        loader = WindsorDataLoader(data_dir)
        features, targets = loader.get_feature_target_split()
        
        # Select targets based on type
        if self.target_type == 'drag':
            y = targets[['cd']] if 'cd' in targets.columns else targets.iloc[:, [0]]
            preprocessor = create_drag_preprocessor(
                n_features=12,
                random_state=self.random_state,
                test_size=self.test_size
            )
        elif self.target_type == 'lift':
            y = targets[['cl']] if 'cl' in targets.columns else targets.iloc[:, [1]]
            preprocessor = create_lift_preprocessor(
                n_features=10,
                random_state=self.random_state,
                test_size=self.test_size
            )
        else:  # both
            y = targets
            preprocessor = create_multi_target_preprocessor(
                n_features=15,
                random_state=self.random_state,
                test_size=self.test_size
            )
        
        # Fit and transform
        X_processed, y_processed = preprocessor.fit_transform(features, y)
        
        # Store preprocessor
        self.preprocessors[self.target_type] = preprocessor
        
        # Create train-test split
        self.X_train, self.X_test, self.y_train, self.y_test = preprocessor.train_test_split(
            X_processed, y_processed
        )
        
        # Store feature names
        if hasattr(X_processed, 'columns'):
            self.feature_names = X_processed.columns.tolist()
        else:
            self.feature_names = [f'feature_{i}' for i in range(X_processed.shape[1])]
        
        logger.info(f"Data preprocessing complete:")
        logger.info(f"  Training set: {self.X_train.shape}")
        logger.info(f"  Test set: {self.X_test.shape}")
        logger.info(f"  Target type: {self.target_type}")
        logger.info(f"  Feature engineering: {preprocessor.feature_engineering}")
        logger.info(f"  Feature selection: {preprocessor.feature_selection}")
        
        return X_processed, y_processed
    
    def calculate_aerodynamic_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Calculate comprehensive aerodynamic performance metrics.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            
        Returns:
            Dictionary of aerodynamic metrics
        """
        metrics = {}
        
        # Handle multi-target case
        if y_true.ndim > 1 and y_true.shape[1] > 1:
            for i, target in enumerate(['cd', 'cl'] if y_true.shape[1] == 2 else [f'target_{i}' for i in range(y_true.shape[1])]):
                y_true_single = y_true[:, i]
                y_pred_single = y_pred[:, i]
                
                metrics[f'{target}_r2'] = r2_score(y_true_single, y_pred_single)
                metrics[f'{target}_rmse'] = np.sqrt(mean_squared_error(y_true_single, y_pred_single))
                metrics[f'{target}_mae'] = mean_absolute_error(y_true_single, y_pred_single)
                metrics[f'{target}_mape'] = mean_absolute_percentage_error(y_true_single, y_pred_single)
                metrics[f'{target}_explained_var'] = explained_variance_score(y_true_single, y_pred_single)
                
                # Aerodynamic-specific metrics
                metrics[f'{target}_mean_error'] = np.mean(y_pred_single - y_true_single)
                metrics[f'{target}_max_error'] = np.max(np.abs(y_pred_single - y_true_single))
                
        else:
            # Single target
            if y_true.ndim > 1:
                y_true = y_true.ravel()
                y_pred = y_pred.ravel()
                
            metrics['r2'] = r2_score(y_true, y_pred)
            metrics['rmse'] = np.sqrt(mean_squared_error(y_true, y_pred))
            metrics['mae'] = mean_absolute_error(y_true, y_pred)
            metrics['mape'] = mean_absolute_percentage_error(y_true, y_pred)
            metrics['explained_var'] = explained_variance_score(y_true, y_pred)
            
            # Aerodynamic-specific metrics
            metrics['mean_error'] = np.mean(y_pred - y_true)
            metrics['max_error'] = np.max(np.abs(y_pred - y_true))
            
            # Physics validation metrics
            if self.target_type == 'drag':
                # Check if predictions maintain physical bounds for drag
                metrics['physical_validity'] = np.mean((y_pred >= 0.1) & (y_pred <= 1.0))
            elif self.target_type == 'lift':
                # Lift can be positive or negative, but should be reasonable
                metrics['physical_validity'] = np.mean((y_pred >= -2.0) & (y_pred <= 2.0))
        
        return metrics
    
    def validate_aerodynamic_physics(self, model, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """
        Validate model predictions against aerodynamic physics principles.
        
        Args:
            model: Trained model
            X_test: Test features
            y_test: Test targets
            
        Returns:
            Dictionary of physics validation results
        """
        y_pred = model.predict(X_test)
        
        validation = {
            'monotonicity_tests': {},
            'correlation_tests': {},
            'physics_compliance': {}
        }
        
        # Convert to DataFrame for easier manipulation
        if hasattr(X_test, 'columns'):
            X_df = X_test
        else:
            X_df = pd.DataFrame(X_test, columns=self.feature_names)
        
        # Test monotonicity relationships (where expected)
        expected_relationships = {
            'frontal_area': 'positive',  # Larger area should increase drag
            'clearance': 'negative'      # Lower clearance should increase downforce (negative lift)
        }
        
        for feature, expected_direction in expected_relationships.items():
            if feature in X_df.columns:
                # Calculate correlation between feature and prediction
                if y_pred.ndim > 1:
                    # For multi-target, check drag coefficient
                    corr = np.corrcoef(X_df[feature], y_pred[:, 0])[0, 1]
                    target_name = 'cd'
                else:
                    corr = np.corrcoef(X_df[feature], y_pred)[0, 1]
                    target_name = self.target_type
                
                expected_positive = expected_direction == 'positive'
                is_correct = (corr > 0) == expected_positive
                
                validation['monotonicity_tests'][f'{feature}_{target_name}'] = {
                    'correlation': corr,
                    'expected_direction': expected_direction,
                    'is_physically_correct': is_correct
                }
        
        # Test prediction ranges
        if y_pred.ndim > 1:
            for i, target in enumerate(['cd', 'cl']):
                pred_range = (y_pred[:, i].min(), y_pred[:, i].max())
                true_range = (y_test[:, i].min(), y_test[:, i].max())
                
                validation['physics_compliance'][target] = {
                    'prediction_range': pred_range,
                    'true_range': true_range,
                    'range_similarity': 1 - abs(
                        (pred_range[1] - pred_range[0]) - (true_range[1] - true_range[0])
                    ) / (true_range[1] - true_range[0])
                }
        else:
            pred_range = (y_pred.min(), y_pred.max())
            true_range = (y_test.min(), y_test.max())
            
            validation['physics_compliance'][self.target_type] = {
                'prediction_range': pred_range,
                'true_range': true_range,
                'range_similarity': 1 - abs(
                    (pred_range[1] - pred_range[0]) - (true_range[1] - true_range[0])
                ) / (true_range[1] - true_range[0])
            }
        
        return validation
    
    def train_single_model(self, 
                          model_name: str, 
                          config: Dict,
                          optimize_hyperparameters: bool = True) -> Dict[str, Any]:
        """
        Train a single model with optional hyperparameter optimization.
        
        Args:
            model_name: Name of the model
            config: Model configuration
            optimize_hyperparameters: Whether to optimize hyperparameters
            
        Returns:
            Dictionary containing trained model and results
        """
        logger.info(f"Training {model_name}...")
        
        model = config['model']
        params = config['params']
        
        # Convert target to appropriate format
        if self.y_train.ndim > 1 and self.y_train.shape[1] == 1:
            y_train_formatted = self.y_train.ravel()
            y_test_formatted = self.y_test.ravel()
        else:
            y_train_formatted = self.y_train
            y_test_formatted = self.y_test
        
        # Hyperparameter optimization
        if optimize_hyperparameters and params:
            cv = KFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
            
            if config['search_type'] == 'grid':
                search = GridSearchCV(
                    model, params, cv=cv, scoring='neg_mean_squared_error',
                    n_jobs=-1, verbose=0
                )
            else:  # random search
                n_iter = config.get('n_iter', 20)
                search = RandomizedSearchCV(
                    model, params, n_iter=n_iter, cv=cv, scoring='neg_mean_squared_error',
                    n_jobs=-1, verbose=0, random_state=self.random_state
                )
            
            search.fit(self.X_train, y_train_formatted)
            best_model = search.best_estimator_
            best_params = search.best_params_
            best_cv_score = -search.best_score_
            
        else:
            # Train with default parameters
            best_model = model
            best_params = {}
            best_cv_score = None
            best_model.fit(self.X_train, y_train_formatted)
        
        # Cross-validation scoring
        cv_scores = cross_val_score(
            best_model, self.X_train, y_train_formatted,
            cv=self.cv_folds, scoring='neg_mean_squared_error'
        )
        cv_rmse_scores = np.sqrt(-cv_scores)
        
        # Make predictions
        y_train_pred = best_model.predict(self.X_train)
        y_test_pred = best_model.predict(self.X_test)
        
        # Calculate metrics
        train_metrics = self.calculate_aerodynamic_metrics(y_train_formatted, y_train_pred)
        test_metrics = self.calculate_aerodynamic_metrics(y_test_formatted, y_test_pred)
        
        # Physics validation
        physics_validation = self.validate_aerodynamic_physics(
            best_model, self.X_test, y_test_formatted
        )
        
        # Feature importance (if available)
        feature_importance = None
        if hasattr(best_model, 'feature_importances_'):
            feature_importance = dict(zip(self.feature_names, best_model.feature_importances_))
        elif hasattr(best_model, 'coef_'):
            # For linear models, use absolute coefficients
            coef = best_model.coef_
            if coef.ndim > 1:
                coef = np.mean(np.abs(coef), axis=0)
            else:
                coef = np.abs(coef)
            feature_importance = dict(zip(self.feature_names, coef))
        
        # Compile results
        results = {
            'model': best_model,
            'model_name': model_name,
            'best_params': best_params,
            'train_metrics': train_metrics,
            'test_metrics': test_metrics,
            'cv_rmse_mean': cv_rmse_scores.mean(),
            'cv_rmse_std': cv_rmse_scores.std(),
            'cv_scores': cv_rmse_scores,
            'physics_validation': physics_validation,
            'feature_importance': feature_importance,
            'predictions': {
                'y_train_true': y_train_formatted,
                'y_train_pred': y_train_pred,
                'y_test_true': y_test_formatted,
                'y_test_pred': y_test_pred
            }
        }
        
        # Store results
        self.models[model_name] = best_model
        self.results[model_name] = results
        
        logger.info(f"  {model_name} training complete:")
        logger.info(f"    Test R²: {test_metrics.get('r2', 'N/A'):.4f}")
        logger.info(f"    Test RMSE: {test_metrics.get('rmse', 'N/A'):.4f}")
        logger.info(f"    CV RMSE: {cv_rmse_scores.mean():.4f} ± {cv_rmse_scores.std():.4f}")
        
        return results
    
    def train_all_models(self, 
                        selected_models: Optional[List[str]] = None,
                        optimize_hyperparameters: bool = True) -> Dict[str, Any]:
        """
        Train all or selected models.
        
        Args:
            selected_models: List of model names to train (None for all)
            optimize_hyperparameters: Whether to optimize hyperparameters
            
        Returns:
            Dictionary of all training results
        """
        logger.info("Starting comprehensive model training...")
        
        # Get model configurations
        model_configs = self.define_model_configurations()
        
        # Filter models if specified
        if selected_models:
            model_configs = {name: config for name, config in model_configs.items() 
                           if name in selected_models}
        
        logger.info(f"Training {len(model_configs)} models: {list(model_configs.keys())}")
        
        # Train each model
        all_results = {}
        for model_name, config in model_configs.items():
            try:
                results = self.train_single_model(
                    model_name, config, optimize_hyperparameters
                )
                all_results[model_name] = results
                
            except Exception as e:
                logger.error(f"Error training {model_name}: {str(e)}")
                continue
        
        logger.info(f"Model training complete. Trained {len(all_results)} models successfully.")
        return all_results
    
    def evaluate_models(self) -> pd.DataFrame:
        """
        Create comprehensive model evaluation and comparison.
        
        Returns:
            DataFrame with model comparison results
        """
        if not self.results:
            raise ValueError("No trained models found. Run train_all_models() first.")
        
        # Compile evaluation metrics
        evaluation_data = []
        
        for model_name, results in self.results.items():
            test_metrics = results['test_metrics']
            
            row = {
                'Model': model_name,
                'Test_R2': test_metrics.get('r2', np.nan),
                'Test_RMSE': test_metrics.get('rmse', np.nan),
                'Test_MAE': test_metrics.get('mae', np.nan),
                'Test_MAPE': test_metrics.get('mape', np.nan),
                'CV_RMSE_Mean': results['cv_rmse_mean'],
                'CV_RMSE_Std': results['cv_rmse_std'],
                'Physical_Validity': test_metrics.get('physical_validity', np.nan),
                'Max_Error': test_metrics.get('max_error', np.nan),
                'Mean_Error': test_metrics.get('mean_error', np.nan)
            }
            
            # Add multi-target metrics if available
            if 'cd_r2' in test_metrics:
                row.update({
                    'Cd_R2': test_metrics['cd_r2'],
                    'Cd_RMSE': test_metrics['cd_rmse'],
                    'Cl_R2': test_metrics.get('cl_r2', np.nan),
                    'Cl_RMSE': test_metrics.get('cl_rmse', np.nan)
                })
            
            evaluation_data.append(row)
        
        # Create DataFrame and sort by performance
        eval_df = pd.DataFrame(evaluation_data)
        
        # Sort by R² (descending) and RMSE (ascending)
        primary_r2_col = 'Test_R2' if 'Test_R2' in eval_df.columns else 'Cd_R2'
        primary_rmse_col = 'Test_RMSE' if 'Test_RMSE' in eval_df.columns else 'Cd_RMSE'
        
        eval_df = eval_df.sort_values([primary_r2_col, primary_rmse_col], 
                                     ascending=[False, True])
        
        logger.info("Model evaluation complete:")
        logger.info(f"  Best model (R²): {eval_df.iloc[0]['Model']} ({eval_df.iloc[0][primary_r2_col]:.4f})")
        logger.info(f"  Best model (RMSE): {eval_df.iloc[0]['Model']} ({eval_df.iloc[0][primary_rmse_col]:.4f})")
        
        return eval_df
    
    def analyze_feature_importance(self) -> pd.DataFrame:
        """
        Analyze and compare feature importance across models.
        
        Returns:
            DataFrame with feature importance analysis
        """
        if not self.results:
            raise ValueError("No trained models found.")
        
        # Collect feature importances
        importance_data = {}
        
        for model_name, results in self.results.items():
            if results['feature_importance'] is not None:
                importance_data[model_name] = results['feature_importance']
        
        if not importance_data:
            logger.warning("No feature importance data available from trained models.")
            return pd.DataFrame()
        
        # Create DataFrame
        importance_df = pd.DataFrame(importance_data).fillna(0)
        
        # Add summary statistics
        importance_df['Mean_Importance'] = importance_df.mean(axis=1)
        importance_df['Std_Importance'] = importance_df.std(axis=1)
        importance_df['Max_Importance'] = importance_df.max(axis=1)
        
        # Sort by mean importance
        importance_df = importance_df.sort_values('Mean_Importance', ascending=False)
        
        logger.info("Feature importance analysis complete:")
        logger.info(f"  Most important feature: {importance_df.index[0]}")
        logger.info(f"  Top 5 features: {importance_df.index[:5].tolist()}")
        
        return importance_df
    
    def create_prediction_plots(self, save_path: Optional[str] = None):
        """
        Create comprehensive prediction visualization plots.
        
        Args:
            save_path: Path to save plots (optional)
        """
        if not self.results:
            raise ValueError("No trained models found.")
        
        # Select top 3 models for plotting
        eval_df = self.evaluate_models()
        top_models = eval_df.head(3)['Model'].tolist()
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()
        
        # Plot 1: Prediction vs Actual for best model
        best_model = top_models[0]
        best_results = self.results[best_model]
        
        ax = axes[0]
        y_true = best_results['predictions']['y_test_true']
        y_pred = best_results['predictions']['y_test_pred']
        
        if y_true.ndim > 1:
            # Multi-target: plot drag coefficient
            y_true = y_true[:, 0] if y_true.shape[1] > 1 else y_true.ravel()
            y_pred = y_pred[:, 0] if y_pred.shape[1] > 1 else y_pred.ravel()
        
        ax.scatter(y_true, y_pred, alpha=0.6, s=30)
        ax.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
        
        r2 = r2_score(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        
        ax.set_xlabel('True Values')
        ax.set_ylabel('Predicted Values')
        ax.set_title(f'{best_model} - Prediction vs Actual\nR² = {r2:.4f}, RMSE = {rmse:.4f}')
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Residuals for best model
        ax = axes[1]
        residuals = y_pred - y_true
        ax.scatter(y_pred, residuals, alpha=0.6, s=30)
        ax.axhline(y=0, color='r', linestyle='--')
        ax.set_xlabel('Predicted Values')
        ax.set_ylabel('Residuals')
        ax.set_title(f'{best_model} - Residual Analysis')
        ax.grid(True, alpha=0.3)
        
        # Plot 3: Model comparison (R² scores)
        ax = axes[2]
        model_names = [name.replace('_', ' ').title() for name in top_models]
        r2_scores = [self.results[name]['test_metrics'].get('r2', 0) for name in top_models]
        
        bars = ax.bar(model_names, r2_scores, color=['gold', 'silver', 'bronze'])
        ax.set_ylabel('R² Score')
        ax.set_title('Top 3 Models - R² Comparison')
        ax.set_ylim(0, 1)
        
        # Add value labels on bars
        for bar, score in zip(bars, r2_scores):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{score:.3f}', ha='center', va='bottom')
        
        # Plot 4: Cross-validation scores
        ax = axes[3]
        cv_means = [self.results[name]['cv_rmse_mean'] for name in top_models]
        cv_stds = [self.results[name]['cv_rmse_std'] for name in top_models]
        
        bars = ax.bar(model_names, cv_means, yerr=cv_stds, capsize=5, 
                     color=['gold', 'silver', 'bronze'], alpha=0.7)
        ax.set_ylabel('CV RMSE')
        ax.set_title('Top 3 Models - Cross-Validation RMSE')
        
        plt.tight_layout()
        plt.suptitle(f'Aerodynamic Model Performance Analysis\nTarget: {self.target_type.title()}',
                    y=0.98, fontsize=16, fontweight='bold')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Prediction plots saved to {save_path}")
        
        plt.show()
    
    def save_models_and_results(self, base_path: Optional[str] = None):
        """
        Save trained models, preprocessors, and results.
        
        Args:
            base_path: Base path for saving (optional)
        """
        if base_path is None:
            base_path = self.models_dir
        else:
            base_path = Path(base_path)
            base_path.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save models
        models_path = base_path / f"models_{self.target_type}_{timestamp}.pkl"
        with open(models_path, 'wb') as f:
            pickle.dump(self.models, f)
        logger.info(f"Models saved to {models_path}")
        
        # Save preprocessors
        preprocessors_path = base_path / f"preprocessors_{self.target_type}_{timestamp}.pkl"
        with open(preprocessors_path, 'wb') as f:
            pickle.dump(self.preprocessors, f)
        logger.info(f"Preprocessors saved to {preprocessors_path}")
        
        # Save results as JSON (excluding non-serializable objects)
        results_serializable = {}
        for model_name, results in self.results.items():
            results_serializable[model_name] = {
                'model_name': results['model_name'],
                'best_params': results['best_params'],
                'train_metrics': results['train_metrics'],
                'test_metrics': results['test_metrics'],
                'cv_rmse_mean': results['cv_rmse_mean'],
                'cv_rmse_std': results['cv_rmse_std'],
                'physics_validation': results['physics_validation'],
                'feature_importance': results['feature_importance']
            }
        
        results_path = base_path / f"results_{self.target_type}_{timestamp}.json"
        with open(results_path, 'w') as f:
            json.dump(results_serializable, f, indent=2, default=str)
        logger.info(f"Results saved to {results_path}")
        
        # Save evaluation DataFrame
        eval_df = self.evaluate_models()
        eval_path = base_path / f"evaluation_{self.target_type}_{timestamp}.csv"
        eval_df.to_csv(eval_path, index=False)
        logger.info(f"Evaluation results saved to {eval_path}")
        
        # Save feature importance
        if self.feature_importance or any(r.get('feature_importance') for r in self.results.values()):
            importance_df = self.analyze_feature_importance()
            if not importance_df.empty:
                importance_path = base_path / f"feature_importance_{self.target_type}_{timestamp}.csv"
                importance_df.to_csv(importance_path)
                logger.info(f"Feature importance saved to {importance_path}")
        
        return {
            'models_path': models_path,
            'preprocessors_path': preprocessors_path,
            'results_path': results_path,
            'evaluation_path': eval_path
        }
    
    def generate_training_report(self) -> str:
        """
        Generate comprehensive training report.
        
        Returns:
            Formatted training report string
        """
        if not self.results:
            return "No training results available."
        
        report = []
        report.append("=" * 80)
        report.append("AERODYNAMIC SURROGATE MODEL TRAINING REPORT")
        report.append("=" * 80)
        
        # Training configuration
        report.append(f"\n🎯 TRAINING CONFIGURATION:")
        report.append(f"  Target Type: {self.target_type.upper()}")
        report.append(f"  Models Trained: {len(self.results)}")
        report.append(f"  Cross-Validation Folds: {self.cv_folds}")
        report.append(f"  Test Set Size: {self.test_size:.1%}")
        report.append(f"  Training Set Shape: {self.X_train.shape}")
        report.append(f"  Test Set Shape: {self.X_test.shape}")
        
        # Model performance summary
        eval_df = self.evaluate_models()
        report.append(f"\n📊 MODEL PERFORMANCE SUMMARY:")
        report.append(f"  Best Model: {eval_df.iloc[0]['Model']}")
        
        if 'Test_R2' in eval_df.columns:
            report.append(f"  Best R²: {eval_df.iloc[0]['Test_R2']:.4f}")
            report.append(f"  Best RMSE: {eval_df.iloc[0]['Test_RMSE']:.4f}")
        else:
            report.append(f"  Best Cd R²: {eval_df.iloc[0]['Cd_R2']:.4f}")
            report.append(f"  Best Cd RMSE: {eval_df.iloc[0]['Cd_RMSE']:.4f}")
        
        # Top 5 models
        report.append(f"\n🏆 TOP 5 MODELS:")
        for i, (_, row) in enumerate(eval_df.head(5).iterrows(), 1):
            model_name = row['Model']
            if 'Test_R2' in row:
                r2, rmse = row['Test_R2'], row['Test_RMSE']
            else:
                r2, rmse = row['Cd_R2'], row['Cd_RMSE']
            report.append(f"  {i}. {model_name:20s}: R² = {r2:.4f}, RMSE = {rmse:.4f}")
        
        # Feature importance
        importance_df = self.analyze_feature_importance()
        if not importance_df.empty:
            report.append(f"\n🎛️ TOP 10 MOST IMPORTANT FEATURES:")
            for i, (feature, row) in enumerate(importance_df.head(10).iterrows(), 1):
                report.append(f"  {i:2d}. {feature:25s}: {row['Mean_Importance']:.4f}")
        
        # Physics validation
        report.append(f"\n🔬 PHYSICS VALIDATION SUMMARY:")
        physics_valid_count = 0
        total_tests = 0
        
        for model_name, results in self.results.items():
            physics = results['physics_validation']
            monotonicity = physics.get('monotonicity_tests', {})
            
            for test_name, test_result in monotonicity.items():
                total_tests += 1
                if test_result['is_physically_correct']:
                    physics_valid_count += 1
        
        if total_tests > 0:
            physics_score = physics_valid_count / total_tests
            report.append(f"  Physics Compliance Rate: {physics_score:.1%} ({physics_valid_count}/{total_tests} tests)")
        
        # Advanced model availability
        report.append(f"\n🚀 ADVANCED MODELS:")
        report.append(f"  XGBoost Available: {'✅' if XGBOOST_AVAILABLE else '❌'}")
        report.append(f"  LightGBM Available: {'✅' if LIGHTGBM_AVAILABLE else '❌'}")
        
        # Training recommendations
        report.append(f"\n💡 RECOMMENDATIONS:")
        best_model_name = eval_df.iloc[0]['Model']
        best_model_results = self.results[best_model_name]
        
        if 'Test_R2' in eval_df.columns:
            best_r2 = eval_df.iloc[0]['Test_R2']
        else:
            best_r2 = eval_df.iloc[0]['Cd_R2']
        
        if best_r2 > 0.95:
            report.append(f"  🎉 Excellent model performance achieved!")
            report.append(f"  ✅ {best_model_name} is ready for production use")
        elif best_r2 > 0.90:
            report.append(f"  ✅ Good model performance achieved")
            report.append(f"  📈 Consider ensemble methods for further improvement")
        else:
            report.append(f"  ⚠️  Model performance could be improved")
            report.append(f"  📊 Consider additional feature engineering or data collection")
        
        # Feature engineering summary
        if self.preprocessors:
            preprocessor = list(self.preprocessors.values())[0]
            report.append(f"\n🛠️ PREPROCESSING SUMMARY:")
            report.append(f"  Feature Engineering: {'✅' if preprocessor.feature_engineering else '❌'}")
            report.append(f"  Feature Selection: {'✅' if preprocessor.feature_selection else '❌'}")
            report.append(f"  Outlier Handling: {'✅' if preprocessor.outlier_handling else '❌'}")
            report.append(f"  Scaling Strategy: {preprocessor.scaling_strategy}")
        
        report.append(f"\n" + "=" * 80)
        report.append(f"Report generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("=" * 80)
        
        return "\n".join(report)


def main():
    """
    Main training function with command-line interface.
    """
    parser = argparse.ArgumentParser(description='Train aerodynamic surrogate models')
    parser.add_argument('--target', choices=['drag', 'lift', 'both'], default='both',
                       help='Target variable(s) to predict')
    parser.add_argument('--models', nargs='+', default=None,
                       help='Specific models to train (default: all)')
    parser.add_argument('--data-dir', type=str, default=None,
                       help='Data directory path')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory for models and results')
    parser.add_argument('--no-hyperopt', action='store_true',
                       help='Disable hyperparameter optimization')
    parser.add_argument('--cv-folds', type=int, default=5,
                       help='Number of cross-validation folds')
    parser.add_argument('--test-size', type=float, default=0.2,
                       help='Test set proportion')
    parser.add_argument('--quick', action='store_true',
                       help='Quick training with basic models only')
    
    args = parser.parse_args()
    
    # Configure trainer
    trainer = AerodynamicModelTrainer(
        target_type=args.target,
        cv_folds=args.cv_folds,
        test_size=args.test_size,
        models_dir=args.output_dir or MODELS_DIR,
        enable_advanced_models=not args.quick
    )
    
    try:
        # Load and preprocess data
        trainer.load_and_preprocess_data(args.data_dir)
        
        # Train models
        selected_models = args.models
        if args.quick and selected_models is None:
            # Quick training with basic models
            selected_models = ['linear_regression', 'ridge', 'random_forest', 'gradient_boosting']
        
        trainer.train_all_models(
            selected_models=selected_models,
            optimize_hyperparameters=not args.no_hyperopt
        )
        
        # Generate evaluation and report
        eval_df = trainer.evaluate_models()
        print("\n" + "="*60)
        print("MODEL EVALUATION RESULTS")
        print("="*60)
        print(eval_df.round(4))
        
        # Create plots
        trainer.create_prediction_plots()
        
        # Save results
        saved_paths = trainer.save_models_and_results()
        
        # Generate and print report
        report = trainer.generate_training_report()
        print("\n" + report)
        
        # Save report
        if args.output_dir:
            report_path = Path(args.output_dir) / f"training_report_{args.target}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        else:
            report_path = Path(MODELS_DIR) / f"training_report_{args.target}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        
        with open(report_path, 'w') as f:
            f.write(report)
        
        print(f"\nTraining complete! Results saved to:")
        for key, path in saved_paths.items():
            print(f"  {key}: {path}")
        print(f"  Report: {report_path}")
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()