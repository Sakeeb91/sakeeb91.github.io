"""
Data processing module for Windsor body CFD dataset.

This module provides functions to load, validate, and preprocess the 
Windsor body aerodynamics dataset for machine learning applications.
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path
from typing import Tuple, Dict, Optional, List, Union, Any
import logging
import pickle
import warnings
from abc import ABC, abstractmethod

# Scikit-learn imports for preprocessing
from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, RobustScaler, PowerTransformer,
    PolynomialFeatures, KBinsDiscretizer, FunctionTransformer
)
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.feature_selection import (
    SelectKBest, SelectPercentile, RFE, SelectFromModel,
    mutual_info_regression, f_regression
)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LassoCV
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted
from scipy import stats
from scipy.stats import zscore

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)


class WindsorDataLoader:
    """
    Data loader for the Windsor body CFD dataset.
    
    This class handles loading the high-fidelity CFD dataset containing
    geometric parameters and aerodynamic force coefficients for automotive
    aerodynamics analysis.
    """
    
    def __init__(self, data_dir: str = None):
        """
        Initialize the data loader.
        
        Args:
            data_dir: Path to the raw data directory. If None, uses default path.
        """
        if data_dir is None:
            # Default path relative to project root
            project_root = Path(__file__).parent.parent
            self.data_dir = project_root / "data" / "raw" / "windsorml"
        else:
            self.data_dir = Path(data_dir)
            
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")
            
        logger.info(f"Initialized WindsorDataLoader with data directory: {self.data_dir}")
    
    def load_dataset(self, include_varref: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load the complete Windsor body dataset.
        
        Args:
            include_varref: Whether to include variable reference area force coefficients
            
        Returns:
            Tuple of (geometric_parameters, force_coefficients) DataFrames
        """
        logger.info("Loading Windsor body dataset...")
        
        # Load geometric parameters
        geo_file = self.data_dir / "geo_parameters_all.csv"
        if not geo_file.exists():
            raise FileNotFoundError(f"Geometric parameters file not found: {geo_file}")
        
        geo_data = pd.read_csv(geo_file)
        logger.info(f"Loaded geometric parameters: {geo_data.shape}")
        
        # Load force coefficients
        if include_varref:
            force_file = self.data_dir / "force_mom_varref_all.csv"
        else:
            force_file = self.data_dir / "force_mom_all.csv"
            
        if not force_file.exists():
            raise FileNotFoundError(f"Force coefficients file not found: {force_file}")
            
        force_data = pd.read_csv(force_file)
        
        # Clean column names by stripping whitespace
        force_data.columns = force_data.columns.str.strip()
        
        logger.info(f"Loaded force coefficients: {force_data.shape}")
        
        return geo_data, force_data
    
    def load_combined_dataset(self, include_varref: bool = False) -> pd.DataFrame:
        """
        Load and combine geometric parameters with force coefficients.
        
        Args:
            include_varref: Whether to include variable reference area force coefficients
            
        Returns:
            Combined DataFrame with both geometric parameters and force coefficients
        """
        geo_data, force_data = self.load_dataset(include_varref=include_varref)
        
        # Merge on run number
        combined_data = pd.merge(geo_data, force_data, on='run', how='inner')
        
        logger.info(f"Combined dataset shape: {combined_data.shape}")
        return combined_data
    
    def get_feature_target_split(self, 
                               include_varref: bool = False,
                               target_columns: Optional[list] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Get features and targets ready for machine learning.
        
        Args:
            include_varref: Whether to include variable reference area force coefficients
            target_columns: List of target column names. If None, uses default [cd, cl]
            
        Returns:
            Tuple of (features, targets) DataFrames
        """
        if target_columns is None:
            target_columns = ['cd', 'cl']  # Drag and Lift coefficients
            
        combined_data = self.load_combined_dataset(include_varref=include_varref)
        
        # Define feature columns (geometric parameters, excluding run number)
        feature_columns = [
            'ratio_length_back_fast',
            'ratio_height_nose_windshield', 
            'ratio_height_fast_back',
            'side_taper',
            'clearance',
            'bottom_taper_angle',
            'frontal_area'
        ]
        
        # Validate that all required columns exist
        missing_features = set(feature_columns) - set(combined_data.columns)
        missing_targets = set(target_columns) - set(combined_data.columns)
        
        if missing_features:
            raise ValueError(f"Missing feature columns: {missing_features}")
        if missing_targets:
            raise ValueError(f"Missing target columns: {missing_targets}")
        
        features = combined_data[feature_columns].copy()
        targets = combined_data[target_columns].copy()
        
        logger.info(f"Features shape: {features.shape}")
        logger.info(f"Targets shape: {targets.shape}")
        
        return features, targets


class AerodynamicFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Domain-specific feature engineering for aerodynamic data.
    
    Creates aerodynamically meaningful features based on fluid mechanics principles:
    - Geometric ratios and dimensionless parameters
    - Interaction terms for complex flow phenomena
    - Derived features based on aerodynamic theory
    """
    
    def __init__(self, 
                 create_ratios: bool = True,
                 create_interactions: bool = True,
                 create_polynomial: bool = True,
                 polynomial_degree: int = 2,
                 include_angle_transforms: bool = True):
        """
        Initialize the aerodynamic feature engineer.
        
        Args:
            create_ratios: Create aerodynamically meaningful ratios
            create_interactions: Create interaction terms for complex flow effects
            create_polynomial: Create polynomial features for non-linear relationships
            polynomial_degree: Degree for polynomial features
            include_angle_transforms: Apply trigonometric transforms to angular parameters
        """
        self.create_ratios = create_ratios
        self.create_interactions = create_interactions
        self.create_polynomial = create_polynomial
        self.polynomial_degree = polynomial_degree
        self.include_angle_transforms = include_angle_transforms
        self.feature_names_ = None
        self.original_features_ = None
    
    def fit(self, X, y=None):
        """
        Fit the feature engineer.
        
        Args:
            X: Input features DataFrame
            y: Target values (ignored)
            
        Returns:
            self
        """
        if isinstance(X, pd.DataFrame):
            self.original_features_ = X.columns.tolist()
        else:
            self.original_features_ = [f'feature_{i}' for i in range(X.shape[1])]
            
        return self
    
    def transform(self, X):
        """
        Transform features using aerodynamic domain knowledge.
        
        Args:
            X: Input features
            
        Returns:
            Transformed features DataFrame
        """
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=self.original_features_)
        
        X_transformed = X.copy()
        
        # Create aerodynamically meaningful ratios
        if self.create_ratios:
            X_transformed = self._create_aerodynamic_ratios(X_transformed)
        
        # Create interaction terms for complex flow phenomena
        if self.create_interactions:
            X_transformed = self._create_aerodynamic_interactions(X_transformed)
        
        # Apply angle transforms for angular parameters
        if self.include_angle_transforms:
            X_transformed = self._apply_angle_transforms(X_transformed)
        
        # Create polynomial features for non-linear relationships
        if self.create_polynomial:
            X_transformed = self._create_polynomial_features(X_transformed)
        
        self.feature_names_ = X_transformed.columns.tolist()
        logger.info(f"Feature engineering complete: {len(self.original_features_)} → {len(self.feature_names_)} features")
        
        return X_transformed
    
    def _create_aerodynamic_ratios(self, X):
        """
        Create aerodynamically meaningful ratio features.
        """
        X_ratios = X.copy()
        
        # Aspect ratio approximations
        if 'ratio_length_back_fast' in X.columns and 'ratio_height_fast_back' in X.columns:
            X_ratios['fastback_aspect_ratio'] = X['ratio_length_back_fast'] / (X['ratio_height_fast_back'] + 1e-8)
        
        # Slenderness ratio
        if 'ratio_height_nose_windshield' in X.columns and 'frontal_area' in X.columns:
            X_ratios['vehicle_slenderness'] = X['ratio_height_nose_windshield'] / np.sqrt(X['frontal_area'])
        
        # Ground clearance ratio (important for ground effect)
        if 'clearance' in X.columns and 'frontal_area' in X.columns:
            vehicle_height_approx = np.sqrt(X['frontal_area'])  # Approximation
            X_ratios['clearance_ratio'] = X['clearance'] / vehicle_height_approx
        
        # Blockage ratio indicator
        if 'frontal_area' in X.columns:
            # Normalize by typical test section area
            X_ratios['blockage_indicator'] = X['frontal_area'] / 0.12  # Approximate normalization
        
        return X_ratios
    
    def _create_aerodynamic_interactions(self, X):
        """
        Create interaction terms for complex aerodynamic phenomena.
        """
        X_interactions = X.copy()
        
        # Ground effect interactions
        if 'clearance' in X.columns and 'bottom_taper_angle' in X.columns:
            X_interactions['ground_effect_intensity'] = (1 / (X['clearance'] + 1)) * X['bottom_taper_angle']
        
        # Fastback pressure recovery interaction
        if 'ratio_length_back_fast' in X.columns and 'ratio_height_fast_back' in X.columns:
            X_interactions['fastback_pressure_recovery'] = X['ratio_length_back_fast'] * (1 - X['ratio_height_fast_back'])
        
        # Three-dimensional flow interaction
        if 'side_taper' in X.columns and 'frontal_area' in X.columns:
            X_interactions['crossflow_effect'] = X['side_taper'] * np.sqrt(X['frontal_area'])
        
        # Windshield-fastback transition
        if 'ratio_height_nose_windshield' in X.columns and 'ratio_height_fast_back' in X.columns:
            X_interactions['windshield_fastback_transition'] = abs(X['ratio_height_nose_windshield'] - X['ratio_height_fast_back'])
        
        # Combined geometric complexity
        geometric_cols = ['ratio_length_back_fast', 'ratio_height_nose_windshield', 
                         'ratio_height_fast_back', 'side_taper']
        available_cols = [col for col in geometric_cols if col in X.columns]
        if len(available_cols) >= 3:
            X_interactions['geometric_complexity'] = X[available_cols].std(axis=1)
        
        return X_interactions
    
    def _apply_angle_transforms(self, X):
        """
        Apply trigonometric transforms to angular parameters.
        """
        X_angles = X.copy()
        
        # Angular parameters that benefit from trigonometric transforms
        angle_params = ['side_taper', 'bottom_taper_angle']
        
        for param in angle_params:
            if param in X.columns:
                # Convert to radians (assuming input is in degrees)
                radians = np.deg2rad(X[param])
                
                # Sine and cosine for periodicity
                X_angles[f'{param}_sin'] = np.sin(radians)
                X_angles[f'{param}_cos'] = np.cos(radians)
                
                # Tangent for slope effects (with clipping for stability)
                X_angles[f'{param}_tan'] = np.clip(np.tan(radians), -10, 10)
        
        return X_angles
    
    def _create_polynomial_features(self, X):
        """
        Create polynomial features for capturing non-linear relationships.
        """
        if self.polynomial_degree < 2:
            return X
        
        # Select key parameters for polynomial expansion (avoid explosion)
        key_params = ['ratio_length_back_fast', 'ratio_height_fast_back', 
                     'clearance', 'frontal_area']
        available_params = [col for col in key_params if col in X.columns]
        
        if not available_params:
            return X
        
        X_poly = X.copy()
        
        # Create polynomial features only for key parameters
        for param in available_params[:3]:  # Limit to avoid explosion
            if param in X.columns:
                X_poly[f'{param}_squared'] = X[param] ** 2
                if self.polynomial_degree >= 3:
                    X_poly[f'{param}_cubed'] = X[param] ** 3
        
        return X_poly
    
    def get_feature_names_out(self, input_features=None):
        """
        Get output feature names.
        """
        if self.feature_names_ is None:
            raise ValueError("Feature engineer has not been fitted yet.")
        return self.feature_names_


class AerodynamicScaler(BaseEstimator, TransformerMixin):
    """
    Domain-aware scaling for aerodynamic parameters.
    
    Different parameter types require different scaling approaches:
    - Geometric ratios: StandardScaler (normally distributed)
    - Angular parameters: RobustScaler (may have outliers)
    - Force coefficients: PowerTransformer for normalization
    - Clearance: Log transformation + StandardScaler
    """
    
    def __init__(self, scaling_strategy: str = 'mixed'):
        """
        Initialize aerodynamic scaler.
        
        Args:
            scaling_strategy: 'mixed', 'standard', 'robust', 'minmax', 'power'
        """
        self.scaling_strategy = scaling_strategy
        self.scalers_ = {}
        self.feature_groups_ = {}
        self.is_fitted_ = False
    
    def fit(self, X, y=None):
        """
        Fit scalers for different parameter types.
        """
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)
        
        self._categorize_features(X)
        
        for group_name, features in self.feature_groups_.items():
            if features:
                scaler = self._get_scaler_for_group(group_name)
                scaler.fit(X[features])
                self.scalers_[group_name] = scaler
        
        self.is_fitted_ = True
        return self
    
    def transform(self, X):
        """
        Transform features using appropriate scalers.
        """
        check_is_fitted(self, 'is_fitted_')
        
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)
        
        X_scaled = X.copy()
        
        for group_name, features in self.feature_groups_.items():
            if features and group_name in self.scalers_:
                scaler = self.scalers_[group_name]
                X_scaled[features] = scaler.transform(X[features])
        
        return X_scaled
    
    def _categorize_features(self, X):
        """
        Categorize features based on aerodynamic parameter types.
        """
        self.feature_groups_ = {
            'ratios': [],
            'angles': [],
            'areas': [],
            'clearance': [],
            'interactions': [],
            'other': []
        }
        
        for col in X.columns:
            col_lower = col.lower()
            
            if 'ratio' in col_lower:
                self.feature_groups_['ratios'].append(col)
            elif any(angle_term in col_lower for angle_term in ['angle', 'taper', 'sin', 'cos', 'tan']):
                self.feature_groups_['angles'].append(col)
            elif 'area' in col_lower:
                self.feature_groups_['areas'].append(col)
            elif 'clearance' in col_lower:
                self.feature_groups_['clearance'].append(col)
            elif any(interaction_term in col_lower for interaction_term in ['interaction', 'effect', 'transition', 'complexity']):
                self.feature_groups_['interactions'].append(col)
            else:
                self.feature_groups_['other'].append(col)
    
    def _get_scaler_for_group(self, group_name):
        """
        Get appropriate scaler for feature group.
        """
        if self.scaling_strategy == 'mixed':
            scaler_map = {
                'ratios': StandardScaler(),
                'angles': RobustScaler(),
                'areas': PowerTransformer(method='yeo-johnson'),
                'clearance': Pipeline([('log', FunctionTransformer(np.log1p)), 
                                     ('scale', StandardScaler())]),
                'interactions': RobustScaler(),
                'other': StandardScaler()
            }
            return scaler_map.get(group_name, StandardScaler())
        else:
            scaler_map = {
                'standard': StandardScaler(),
                'robust': RobustScaler(),
                'minmax': MinMaxScaler(),
                'power': PowerTransformer(method='yeo-johnson')
            }
            return scaler_map.get(self.scaling_strategy, StandardScaler())


class OutlierDetector(BaseEstimator, TransformerMixin):
    """
    Aerodynamic outlier detection and handling.
    
    Uses domain knowledge to identify and handle outliers in aerodynamic data.
    """
    
    def __init__(self, 
                 method: str = 'iqr',
                 threshold: float = 3.0,
                 handle_method: str = 'clip'):
        """
        Initialize outlier detector.
        
        Args:
            method: 'zscore', 'iqr', 'isolation_forest'
            threshold: Threshold for outlier detection
            handle_method: 'clip', 'remove', 'flag'
        """
        self.method = method
        self.threshold = threshold
        self.handle_method = handle_method
        self.outlier_bounds_ = {}
        self.is_fitted_ = False
    
    def fit(self, X, y=None):
        """
        Fit outlier detection parameters.
        """
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)
        
        for col in X.columns:
            if self.method == 'zscore':
                mean = X[col].mean()
                std = X[col].std()
                self.outlier_bounds_[col] = {
                    'lower': mean - self.threshold * std,
                    'upper': mean + self.threshold * std
                }
            elif self.method == 'iqr':
                Q1 = X[col].quantile(0.25)
                Q3 = X[col].quantile(0.75)
                IQR = Q3 - Q1
                self.outlier_bounds_[col] = {
                    'lower': Q1 - self.threshold * IQR,
                    'upper': Q3 + self.threshold * IQR
                }
        
        self.is_fitted_ = True
        return self
    
    def transform(self, X):
        """
        Handle outliers based on fitted parameters.
        """
        check_is_fitted(self, 'is_fitted_')
        
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)
        
        X_clean = X.copy()
        
        if self.handle_method == 'clip':
            for col in X.columns:
                if col in self.outlier_bounds_:
                    bounds = self.outlier_bounds_[col]
                    X_clean[col] = X_clean[col].clip(bounds['lower'], bounds['upper'])
        elif self.handle_method == 'flag':
            # Add outlier flags
            for col in X.columns:
                if col in self.outlier_bounds_:
                    bounds = self.outlier_bounds_[col]
                    outliers = (X[col] < bounds['lower']) | (X[col] > bounds['upper'])
                    X_clean[f'{col}_outlier_flag'] = outliers.astype(int)
        
        return X_clean


class DataValidator:
    """
    Comprehensive data validation for aerodynamic datasets.
    
    Validates data quality, physical constraints, and aerodynamic reasonableness.
    """
    
    def __init__(self):
        self.validation_results = {}
        self.physical_bounds = {
            'cd': (0.1, 1.0),  # Reasonable drag coefficient range
            'cl': (-2.0, 2.0),  # Reasonable lift coefficient range
            'cs': (-0.5, 0.5),  # Side force should be small for symmetric bodies
            'cmy': (-0.5, 0.5),  # Pitching moment reasonable range
            'frontal_area': (0.05, 0.2),  # Reasonable frontal area range
            'clearance': (10, 300),  # Clearance in mm
        }
    
    def validate_dataset(self, X, y=None, feature_names=None):
        """
        Perform comprehensive dataset validation.
        """
        if isinstance(X, np.ndarray):
            if feature_names is None:
                feature_names = [f'feature_{i}' for i in range(X.shape[1])]
            X = pd.DataFrame(X, columns=feature_names)
        
        results = {
            'data_quality': self._check_data_quality(X),
            'physical_constraints': self._check_physical_constraints(X),
            'statistical_properties': self._check_statistical_properties(X),
            'aerodynamic_reasonableness': self._check_aerodynamic_reasonableness(X, y)
        }
        
        self.validation_results = results
        return results
    
    def _check_data_quality(self, X):
        """
        Check basic data quality metrics.
        """
        return {
            'missing_values': X.isnull().sum().to_dict(),
            'duplicate_rows': X.duplicated().sum(),
            'data_types': X.dtypes.to_dict(),
            'infinite_values': np.isinf(X.select_dtypes(include=[np.number])).sum().to_dict()
        }
    
    def _check_physical_constraints(self, X):
        """
        Check physical constraint violations.
        """
        violations = {}
        
        for col in X.columns:
            col_lower = col.lower()
            
            # Check against known physical bounds
            for param, (min_val, max_val) in self.physical_bounds.items():
                if param in col_lower:
                    violations[col] = {
                        'below_min': (X[col] < min_val).sum(),
                        'above_max': (X[col] > max_val).sum(),
                        'valid_range': f'[{min_val}, {max_val}]'
                    }
        
        return violations
    
    def _check_statistical_properties(self, X):
        """
        Check statistical properties of the data.
        """
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        
        return {
            'basic_stats': X[numeric_cols].describe().to_dict(),
            'skewness': X[numeric_cols].skew().to_dict(),
            'kurtosis': X[numeric_cols].kurtosis().to_dict(),
            'correlation_matrix': X[numeric_cols].corr().to_dict()
        }
    
    def _check_aerodynamic_reasonableness(self, X, y):
        """
        Check aerodynamic reasonableness of relationships.
        """
        if y is None:
            return {'message': 'No target data provided for aerodynamic validation'}
        
        if isinstance(y, np.ndarray):
            y = pd.DataFrame(y, columns=['cd', 'cl'] if y.shape[1] == 2 else [f'target_{i}' for i in range(y.shape[1])])
        
        reasonableness = {}
        
        # Check expected correlations
        expected_correlations = {
            ('frontal_area', 'cd'): 'positive',  # Larger area should increase drag
            ('clearance', 'cl'): 'negative',     # Lower clearance should increase downforce
        }
        
        for (feature, target), expected_sign in expected_correlations.items():
            if feature in X.columns and target in y.columns:
                corr = X[feature].corr(y[target])
                expected_positive = expected_sign == 'positive'
                is_reasonable = (corr > 0) == expected_positive
                
                reasonableness[f'{feature}_vs_{target}'] = {
                    'correlation': corr,
                    'expected_sign': expected_sign,
                    'is_reasonable': is_reasonable
                }
        
        return reasonableness
    
    def generate_validation_report(self):
        """
        Generate a comprehensive validation report.
        """
        if not self.validation_results:
            return "No validation results available. Run validate_dataset() first."
        
        report = []
        report.append("=" * 60)
        report.append("AERODYNAMIC DATA VALIDATION REPORT")
        report.append("=" * 60)
        
        # Data quality summary
        dq = self.validation_results['data_quality']
        report.append("\n📊 DATA QUALITY:")
        report.append(f"Missing values: {sum(dq['missing_values'].values())}")
        report.append(f"Duplicate rows: {dq['duplicate_rows']}")
        report.append(f"Infinite values: {sum(dq['infinite_values'].values())}")
        
        # Physical constraints
        pc = self.validation_results['physical_constraints']
        report.append("\n🔬 PHYSICAL CONSTRAINTS:")
        for param, violations in pc.items():
            total_violations = violations['below_min'] + violations['above_max']
            if total_violations > 0:
                report.append(f"⚠️  {param}: {total_violations} violations {violations['valid_range']}")
            else:
                report.append(f"✅ {param}: No violations")
        
        # Aerodynamic reasonableness
        ar = self.validation_results['aerodynamic_reasonableness']
        if 'message' not in ar:
            report.append("\n🌪️  AERODYNAMIC REASONABLENESS:")
            for relationship, check in ar.items():
                status = "✅" if check['is_reasonable'] else "⚠️"
                report.append(f"{status} {relationship}: r={check['correlation']:.3f} (expected {check['expected_sign']})")
        
        report.append("\n" + "=" * 60)
        
        return "\n".join(report)


class AerodynamicFeatureSelector(BaseEstimator, TransformerMixin):
    """
    Aerodynamic-aware feature selection.
    
    Combines statistical feature selection with domain knowledge
    to select the most relevant features for aerodynamic prediction.
    """
    
    def __init__(self, 
                 method: str = 'combined',
                 k_features: int = 10,
                 correlation_threshold: float = 0.95):
        """
        Initialize feature selector.
        
        Args:
            method: 'correlation', 'mutual_info', 'rfe', 'lasso', 'combined'
            k_features: Number of features to select
            correlation_threshold: Threshold for removing highly correlated features
        """
        self.method = method
        self.k_features = k_features
        self.correlation_threshold = correlation_threshold
        self.selected_features_ = None
        self.selector_ = None
        self.is_fitted_ = False
    
    def fit(self, X, y):
        """
        Fit feature selector.
        """
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)
        if isinstance(y, np.ndarray):
            y = pd.DataFrame(y) if y.ndim > 1 else pd.Series(y)
        
        # Remove highly correlated features first
        X_uncorr = self._remove_correlated_features(X)
        
        # Apply feature selection method
        if self.method == 'correlation':
            self.selected_features_ = self._correlation_based_selection(X_uncorr, y)
        elif self.method == 'mutual_info':
            self.selected_features_ = self._mutual_info_selection(X_uncorr, y)
        elif self.method == 'rfe':
            self.selected_features_ = self._rfe_selection(X_uncorr, y)
        elif self.method == 'lasso':
            self.selected_features_ = self._lasso_selection(X_uncorr, y)
        elif self.method == 'combined':
            self.selected_features_ = self._combined_selection(X_uncorr, y)
        
        self.is_fitted_ = True
        return self
    
    def transform(self, X):
        """
        Transform by selecting features.
        """
        check_is_fitted(self, 'is_fitted_')
        
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)
        
        return X[self.selected_features_]
    
    def _remove_correlated_features(self, X):
        """
        Remove highly correlated features.
        """
        corr_matrix = X.corr().abs()
        upper_tri = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        
        to_drop = [column for column in upper_tri.columns 
                  if any(upper_tri[column] > self.correlation_threshold)]
        
        logger.info(f"Removing {len(to_drop)} highly correlated features")
        return X.drop(columns=to_drop)
    
    def _correlation_based_selection(self, X, y):
        """
        Select features based on correlation with target.
        """
        if y.ndim > 1:
            # For multiple targets, use average correlation
            correlations = []
            for col in y.columns:
                corr = X.corrwith(y[col]).abs()
                correlations.append(corr)
            avg_corr = pd.concat(correlations, axis=1).mean(axis=1)
        else:
            avg_corr = X.corrwith(y).abs()
        
        return avg_corr.nlargest(self.k_features).index.tolist()
    
    def _mutual_info_selection(self, X, y):
        """
        Select features based on mutual information.
        """
        if y.ndim > 1:
            # Use first target for selection
            y_select = y.iloc[:, 0] if hasattr(y, 'iloc') else y[:, 0]
        else:
            y_select = y
        
        selector = SelectKBest(score_func=mutual_info_regression, k=self.k_features)
        selector.fit(X, y_select)
        return X.columns[selector.get_support()].tolist()
    
    def _rfe_selection(self, X, y):
        """
        Select features using Recursive Feature Elimination.
        """
        if y.ndim > 1:
            y_select = y.iloc[:, 0] if hasattr(y, 'iloc') else y[:, 0]
        else:
            y_select = y
        
        estimator = RandomForestRegressor(n_estimators=50, random_state=42)
        selector = RFE(estimator, n_features_to_select=self.k_features)
        selector.fit(X, y_select)
        return X.columns[selector.get_support()].tolist()
    
    def _lasso_selection(self, X, y):
        """
        Select features using Lasso regularization.
        """
        if y.ndim > 1:
            y_select = y.iloc[:, 0] if hasattr(y, 'iloc') else y[:, 0]
        else:
            y_select = y
        
        lasso = LassoCV(cv=5, random_state=42)
        lasso.fit(X, y_select)
        
        # Select features with non-zero coefficients
        selected_mask = lasso.coef_ != 0
        selected_features = X.columns[selected_mask].tolist()
        
        # If too many selected, take top k by coefficient magnitude
        if len(selected_features) > self.k_features:
            coef_abs = np.abs(lasso.coef_[selected_mask])
            top_indices = np.argsort(coef_abs)[-self.k_features:]
            selected_features = [selected_features[i] for i in top_indices]
        
        return selected_features
    
    def _combined_selection(self, X, y):
        """
        Combine multiple selection methods.
        """
        methods = ['correlation', 'mutual_info', 'rfe', 'lasso']
        all_selected = []
        
        for method in methods:
            try:
                if method == 'correlation':
                    features = self._correlation_based_selection(X, y)
                elif method == 'mutual_info':
                    features = self._mutual_info_selection(X, y)
                elif method == 'rfe':
                    features = self._rfe_selection(X, y)
                elif method == 'lasso':
                    features = self._lasso_selection(X, y)
                
                all_selected.extend(features)
            except Exception as e:
                logger.warning(f"Method {method} failed: {e}")
        
        # Count frequency of selection
        feature_counts = pd.Series(all_selected).value_counts()
        
        # Select features that appear in multiple methods
        return feature_counts.head(self.k_features).index.tolist()


class AerodynamicPreprocessor:
    """
    Complete preprocessing pipeline for aerodynamic data.
    
    Combines all preprocessing steps into a unified pipeline:
    1. Data validation
    2. Feature engineering
    3. Outlier handling
    4. Scaling
    5. Feature selection
    """
    
    def __init__(self, 
                 target_type: str = 'drag',
                 feature_engineering: bool = True,
                 outlier_handling: bool = True,
                 scaling_strategy: str = 'mixed',
                 feature_selection: bool = True,
                 feature_selection_method: str = 'combined',
                 n_features: int = 15,
                 test_size: float = 0.2,
                 random_state: int = 42,
                 stratify_by_performance: bool = True):
        """
        Initialize the complete preprocessing pipeline.
        
        Args:
            target_type: 'drag', 'lift', 'both' - determines target-specific preprocessing
            feature_engineering: Whether to apply feature engineering
            outlier_handling: Whether to handle outliers
            scaling_strategy: Scaling strategy for AerodynamicScaler
            feature_selection: Whether to apply feature selection
            feature_selection_method: Method for feature selection
            n_features: Number of features to select
            test_size: Test set size for train-test split
            random_state: Random state for reproducibility
            stratify_by_performance: Whether to stratify splits by performance quartiles
        """
        self.target_type = target_type
        self.feature_engineering = feature_engineering
        self.outlier_handling = outlier_handling
        self.scaling_strategy = scaling_strategy
        self.feature_selection = feature_selection
        self.feature_selection_method = feature_selection_method
        self.n_features = n_features
        self.test_size = test_size
        self.random_state = random_state
        self.stratify_by_performance = stratify_by_performance
        
        # Initialize components
        self.validator = DataValidator()
        self.feature_engineer = None
        self.outlier_detector = None
        self.scaler = None
        self.feature_selector = None
        
        # Store preprocessing objects for serialization
        self.preprocessing_pipeline = None
        self.is_fitted_ = False
        
        logger.info(f"Initialized AerodynamicPreprocessor for {target_type} prediction")
    
    def fit_transform(self, X, y=None):
        """
        Fit the complete preprocessing pipeline and transform data.
        
        Args:
            X: Feature data
            y: Target data
            
        Returns:
            Tuple of (X_processed, y_processed) or just X_processed if y is None
        """
        logger.info("Starting complete preprocessing pipeline...")
        
        # Step 1: Data validation
        logger.info("Step 1: Data validation")
        validation_results = self.validator.validate_dataset(X, y)
        
        # Step 2: Feature engineering
        if self.feature_engineering:
            logger.info("Step 2: Feature engineering")
            self.feature_engineer = AerodynamicFeatureEngineer(
                create_ratios=True,
                create_interactions=True,
                create_polynomial=True,
                polynomial_degree=2
            )
            X = self.feature_engineer.fit_transform(X)
        
        # Step 3: Outlier handling
        if self.outlier_handling:
            logger.info("Step 3: Outlier detection and handling")
            self.outlier_detector = OutlierDetector(
                method='iqr',
                threshold=2.5,
                handle_method='clip'
            )
            X = self.outlier_detector.fit_transform(X)
        
        # Step 4: Scaling
        logger.info("Step 4: Feature scaling")
        self.scaler = AerodynamicScaler(scaling_strategy=self.scaling_strategy)
        X = self.scaler.fit_transform(X)
        
        # Step 5: Feature selection
        if self.feature_selection and y is not None:
            logger.info("Step 5: Feature selection")
            self.feature_selector = AerodynamicFeatureSelector(
                method=self.feature_selection_method,
                k_features=self.n_features
            )
            X = self.feature_selector.fit_transform(X, y)
        
        self.is_fitted_ = True
        logger.info(f"Preprocessing complete: Final feature shape {X.shape}")
        
        if y is not None:
            return X, y
        return X
    
    def transform(self, X):
        """
        Transform new data using fitted preprocessing pipeline.
        """
        if not self.is_fitted_:
            raise ValueError("Preprocessor must be fitted before transforming data")
        
        X_transformed = X.copy()
        
        # Apply transformations in order
        if self.feature_engineer is not None:
            X_transformed = self.feature_engineer.transform(X_transformed)
        
        if self.outlier_detector is not None:
            X_transformed = self.outlier_detector.transform(X_transformed)
        
        if self.scaler is not None:
            X_transformed = self.scaler.transform(X_transformed)
        
        if self.feature_selector is not None:
            X_transformed = self.feature_selector.transform(X_transformed)
        
        return X_transformed
    
    def train_test_split(self, X, y, stratify=None):
        """
        Create train-test split with optional stratification by performance.
        
        Args:
            X: Features
            y: Targets
            stratify: Custom stratification variable
            
        Returns:
            X_train, X_test, y_train, y_test
        """
        if self.stratify_by_performance and stratify is None:
            # Create performance-based stratification
            if isinstance(y, pd.DataFrame):
                # Use primary target for stratification
                target_col = 'cd' if 'cd' in y.columns else y.columns[0]
                stratify_var = pd.qcut(y[target_col], q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
            else:
                stratify_var = pd.qcut(y if y.ndim == 1 else y[:, 0], 
                                     q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
        else:
            stratify_var = stratify
        
        return train_test_split(
            X, y, 
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=stratify_var
        )
    
    def create_cv_splits(self, X, y, n_splits=5):
        """
        Create cross-validation splits with stratification.
        
        Args:
            X: Features
            y: Targets
            n_splits: Number of CV splits
            
        Returns:
            StratifiedShuffleSplit object
        """
        if isinstance(y, pd.DataFrame):
            target_col = 'cd' if 'cd' in y.columns else y.columns[0]
            stratify_var = pd.qcut(y[target_col], q=4, labels=False)
        else:
            stratify_var = pd.qcut(y if y.ndim == 1 else y[:, 0], q=4, labels=False)
        
        return StratifiedShuffleSplit(
            n_splits=n_splits,
            test_size=self.test_size,
            random_state=self.random_state
        )
    
    def save_preprocessor(self, filepath):
        """
        Save the fitted preprocessing pipeline.
        
        Args:
            filepath: Path to save the preprocessor
        """
        if not self.is_fitted_:
            raise ValueError("Preprocessor must be fitted before saving")
        
        preprocessor_data = {
            'feature_engineer': self.feature_engineer,
            'outlier_detector': self.outlier_detector,
            'scaler': self.scaler,
            'feature_selector': self.feature_selector,
            'config': {
                'target_type': self.target_type,
                'feature_engineering': self.feature_engineering,
                'outlier_handling': self.outlier_handling,
                'scaling_strategy': self.scaling_strategy,
                'feature_selection': self.feature_selection,
                'feature_selection_method': self.feature_selection_method,
                'n_features': self.n_features
            }
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(preprocessor_data, f)
        
        logger.info(f"Preprocessor saved to {filepath}")
    
    @classmethod
    def load_preprocessor(cls, filepath):
        """
        Load a fitted preprocessing pipeline.
        
        Args:
            filepath: Path to the saved preprocessor
            
        Returns:
            Loaded AerodynamicPreprocessor instance
        """
        with open(filepath, 'rb') as f:
            preprocessor_data = pickle.load(f)
        
        # Create new instance with saved config
        config = preprocessor_data['config']
        instance = cls(**config)
        
        # Restore fitted components
        instance.feature_engineer = preprocessor_data['feature_engineer']
        instance.outlier_detector = preprocessor_data['outlier_detector']
        instance.scaler = preprocessor_data['scaler']
        instance.feature_selector = preprocessor_data['feature_selector']
        instance.is_fitted_ = True
        
        logger.info(f"Preprocessor loaded from {filepath}")
        return instance
    
    def get_feature_importance_summary(self):
        """
        Get summary of feature importance from various selection methods.
        
        Returns:
            Dictionary with feature importance information
        """
        if not self.is_fitted_:
            raise ValueError("Preprocessor must be fitted first")
        
        summary = {
            'original_features': getattr(self.feature_engineer, 'original_features_', []),
            'engineered_features': getattr(self.feature_engineer, 'feature_names_', []),
            'selected_features': getattr(self.feature_selector, 'selected_features_', []),
            'feature_engineering_enabled': self.feature_engineering,
            'feature_selection_enabled': self.feature_selection,
            'scaling_strategy': self.scaling_strategy
        }
        
        return summary
    
    def generate_preprocessing_report(self):
        """
        Generate comprehensive preprocessing report.
        
        Returns:
            Formatted report string
        """
        if not self.is_fitted_:
            return "Preprocessor has not been fitted yet."
        
        report = []
        report.append("=" * 60)
        report.append("AERODYNAMIC PREPROCESSING PIPELINE REPORT")
        report.append("=" * 60)
        
        # Configuration
        report.append(f"\n🎯 TARGET TYPE: {self.target_type.upper()}")
        report.append(f"📊 FEATURE ENGINEERING: {'✅ Enabled' if self.feature_engineering else '❌ Disabled'}")
        report.append(f"🔍 OUTLIER HANDLING: {'✅ Enabled' if self.outlier_handling else '❌ Disabled'}")
        report.append(f"⚖️ SCALING STRATEGY: {self.scaling_strategy}")
        report.append(f"🎛️ FEATURE SELECTION: {'✅ Enabled' if self.feature_selection else '❌ Disabled'}")
        
        # Feature engineering summary
        if self.feature_engineering and self.feature_engineer:
            original_count = len(getattr(self.feature_engineer, 'original_features_', []))
            engineered_count = len(getattr(self.feature_engineer, 'feature_names_', []))
            report.append(f"\n🛠️ FEATURE ENGINEERING:")
            report.append(f"   Original features: {original_count}")
            report.append(f"   After engineering: {engineered_count}")
            report.append(f"   Features added: {engineered_count - original_count}")
        
        # Feature selection summary
        if self.feature_selection and self.feature_selector:
            selected_count = len(getattr(self.feature_selector, 'selected_features_', []))
            report.append(f"\n🎯 FEATURE SELECTION:")
            report.append(f"   Method: {self.feature_selection_method}")
            report.append(f"   Features selected: {selected_count}")
            report.append(f"   Selected features: {getattr(self.feature_selector, 'selected_features_', [])}")
        
        # Validation summary
        if hasattr(self.validator, 'validation_results') and self.validator.validation_results:
            report.append("\n📋 DATA VALIDATION:")
            dq = self.validator.validation_results.get('data_quality', {})
            missing = sum(dq.get('missing_values', {}).values())
            duplicates = dq.get('duplicate_rows', 0)
            report.append(f"   Missing values: {missing}")
            report.append(f"   Duplicate rows: {duplicates}")
            
            # Physical constraints
            pc = self.validator.validation_results.get('physical_constraints', {})
            if pc:
                violations = sum(
                    v.get('below_min', 0) + v.get('above_max', 0) 
                    for v in pc.values() if isinstance(v, dict)
                )
                report.append(f"   Physical constraint violations: {violations}")
        
        report.append("\n" + "=" * 60)
        
        return "\n".join(report)




# Convenience functions for quick preprocessing
def create_drag_preprocessor(**kwargs):
    """
    Create preprocessor optimized for drag coefficient prediction.
    """
    default_config = {
        'target_type': 'drag',
        'feature_engineering': True,
        'scaling_strategy': 'mixed',
        'feature_selection_method': 'combined',
        'n_features': 12
    }
    default_config.update(kwargs)
    return AerodynamicPreprocessor(**default_config)


def create_lift_preprocessor(**kwargs):
    """
    Create preprocessor optimized for lift coefficient prediction.
    """
    default_config = {
        'target_type': 'lift',
        'feature_engineering': True,
        'scaling_strategy': 'mixed',
        'feature_selection_method': 'combined',
        'n_features': 10
    }
    default_config.update(kwargs)
    return AerodynamicPreprocessor(**default_config)


def create_multi_target_preprocessor(**kwargs):
    """
    Create preprocessor for multi-target (drag and lift) prediction.
    """
    default_config = {
        'target_type': 'both',
        'feature_engineering': True,
        'scaling_strategy': 'mixed',
        'feature_selection_method': 'combined',
        'n_features': 15
    }
    default_config.update(kwargs)
    return AerodynamicPreprocessor(**default_config)


def quick_preprocess_windsor_data(data_dir=None, target_type='both', **kwargs):
    """
    Quick preprocessing of Windsor data with sensible defaults.
    
    Args:
        data_dir: Path to data directory
        target_type: 'drag', 'lift', or 'both'
        **kwargs: Additional arguments for preprocessor
        
    Returns:
        Tuple of (X_train, X_test, y_train, y_test, preprocessor)
    """
    # Load data
    loader = WindsorDataLoader(data_dir)
    features, targets = loader.get_feature_target_split()
    
    # Select targets based on type
    if target_type == 'drag':
        y = targets[['cd']] if 'cd' in targets.columns else targets.iloc[:, [0]]
    elif target_type == 'lift':
        y = targets[['cl']] if 'cl' in targets.columns else targets.iloc[:, [1]]
    else:  # both
        y = targets
    
    # Create and fit preprocessor
    if target_type == 'drag':
        preprocessor = create_drag_preprocessor(**kwargs)
    elif target_type == 'lift':
        preprocessor = create_lift_preprocessor(**kwargs)
    else:
        preprocessor = create_multi_target_preprocessor(**kwargs)
    
    # Fit and transform
    X_processed, y_processed = preprocessor.fit_transform(features, y)
    
    # Create train-test split
    X_train, X_test, y_train, y_test = preprocessor.train_test_split(X_processed, y_processed)
    
    logger.info(f"Quick preprocessing complete:")
    logger.info(f"  Training set: {X_train.shape}")
    logger.info(f"  Test set: {X_test.shape}")
    logger.info(f"  Target type: {target_type}")
    
    return X_train, X_test, y_train, y_test, preprocessor


def validate_dataset(geo_data: pd.DataFrame, force_data: pd.DataFrame) -> Dict[str, any]:
    """
    Perform comprehensive validation of the dataset.
    
    Args:
        geo_data: Geometric parameters DataFrame
        force_data: Force coefficients DataFrame
        
    Returns:
        Dictionary containing validation results
    """
    validation_results = {}
    
    # Check data shapes
    validation_results['geo_shape'] = geo_data.shape
    validation_results['force_shape'] = force_data.shape
    
    # Check for missing values
    validation_results['geo_missing'] = geo_data.isnull().sum().to_dict()
    validation_results['force_missing'] = force_data.isnull().sum().to_dict()
    
    # Check data types
    validation_results['geo_dtypes'] = geo_data.dtypes.to_dict()
    validation_results['force_dtypes'] = force_data.dtypes.to_dict()
    
    # Check for duplicates
    validation_results['geo_duplicates'] = geo_data.duplicated().sum()
    validation_results['force_duplicates'] = force_data.duplicated().sum()
    
    # Check run number consistency
    geo_runs = set(geo_data['run'].values)
    force_runs = set(force_data['run'].values)
    validation_results['run_consistency'] = geo_runs == force_runs
    validation_results['missing_geo_runs'] = force_runs - geo_runs
    validation_results['missing_force_runs'] = geo_runs - force_runs
    
    # Basic statistics for numeric columns
    validation_results['geo_stats'] = geo_data.describe().to_dict()
    validation_results['force_stats'] = force_data.describe().to_dict()
    
    return validation_results


def generate_data_summary_report(validation_results: Dict[str, any]) -> str:
    """
    Generate a comprehensive data summary report.
    
    Args:
        validation_results: Results from validate_dataset function
        
    Returns:
        Formatted string report
    """
    report = []
    report.append("=" * 60)
    report.append("WINDSOR BODY CFD DATASET SUMMARY REPORT")
    report.append("=" * 60)
    
    # Dataset dimensions
    report.append("\n📊 DATASET DIMENSIONS:")
    report.append(f"Geometric Parameters: {validation_results['geo_shape'][0]} rows × {validation_results['geo_shape'][1]} columns")
    report.append(f"Force Coefficients: {validation_results['force_shape'][0]} rows × {validation_results['force_shape'][1]} columns")
    
    # Data quality checks
    report.append("\n🔍 DATA QUALITY:")
    geo_missing_total = sum(validation_results['geo_missing'].values())
    force_missing_total = sum(validation_results['force_missing'].values())
    report.append(f"Missing values in geometric data: {geo_missing_total}")
    report.append(f"Missing values in force data: {force_missing_total}")
    report.append(f"Duplicate rows in geometric data: {validation_results['geo_duplicates']}")
    report.append(f"Duplicate rows in force data: {validation_results['force_duplicates']}")
    report.append(f"Run number consistency: {'✅ PASS' if validation_results['run_consistency'] else '❌ FAIL'}")
    
    # Column information
    report.append("\n📋 GEOMETRIC PARAMETERS COLUMNS:")
    for col, dtype in validation_results['geo_dtypes'].items():
        report.append(f"  - {col}: {dtype}")
    
    report.append("\n📋 FORCE COEFFICIENTS COLUMNS:")
    for col, dtype in validation_results['force_dtypes'].items():
        report.append(f"  - {col}: {dtype}")
    
    # Summary statistics for key columns
    report.append("\n📈 KEY STATISTICS:")
    
    # Geometric parameters
    if 'ratio_length_back_fast' in validation_results['geo_stats']:
        stats = validation_results['geo_stats']['ratio_length_back_fast']
        report.append(f"Ratio Length Back Fast: min={stats['min']:.3f}, max={stats['max']:.3f}, mean={stats['mean']:.3f}")
    
    if 'frontal_area' in validation_results['geo_stats']:
        stats = validation_results['geo_stats']['frontal_area']
        report.append(f"Frontal Area: min={stats['min']:.6f}, max={stats['max']:.6f}, mean={stats['mean']:.6f}")
    
    # Force coefficients
    if 'cd' in validation_results['force_stats']:
        stats = validation_results['force_stats']['cd']
        report.append(f"Drag Coefficient (Cd): min={stats['min']:.3f}, max={stats['max']:.3f}, mean={stats['mean']:.3f}")
    
    if 'cl' in validation_results['force_stats']:
        stats = validation_results['force_stats']['cl']
        report.append(f"Lift Coefficient (Cl): min={stats['min']:.3f}, max={stats['max']:.3f}, mean={stats['mean']:.3f}")
    
    report.append("\n" + "=" * 60)
    
    return "\n".join(report)


# Convenience function for quick dataset loading
def load_windsor_dataset(data_dir: str = None, 
                        include_varref: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Convenience function to quickly load the Windsor dataset.
    
    Args:
        data_dir: Path to data directory
        include_varref: Whether to include variable reference area coefficients
        
    Returns:
        Tuple of (features, targets) ready for ML
    """
    loader = WindsorDataLoader(data_dir)
    return loader.get_feature_target_split(include_varref=include_varref)


if __name__ == "__main__":
    # Example usage and validation of the enhanced preprocessing pipeline
    try:
        print("=" * 70)
        print("ENHANCED AERODYNAMIC PREPROCESSING PIPELINE DEMO")
        print("=" * 70)
        
        # 1. Load and validate data
        print("\n1. Loading Windsor body dataset...")
        loader = WindsorDataLoader()
        geo_data, force_data = loader.load_dataset()
        
        # Original validation
        validation_results = validate_dataset(geo_data, force_data)
        report = generate_data_summary_report(validation_results)
        print(report)
        
        # 2. Test enhanced preprocessing pipeline
        print("\n2. Testing enhanced preprocessing pipeline...")
        
        # Quick preprocessing for drag prediction
        X_train, X_test, y_train, y_test, preprocessor = quick_preprocess_windsor_data(
            target_type='drag',
            n_features=12,
            scaling_strategy='mixed'
        )
        
        print(f"\nDrag prediction preprocessing results:")
        print(f"  Training features shape: {X_train.shape}")
        print(f"  Test features shape: {X_test.shape}")
        print(f"  Training targets shape: {y_train.shape}")
        print(f"  Test targets shape: {y_test.shape}")
        
        # 3. Show preprocessing report
        print("\n3. Preprocessing Pipeline Report:")
        print(preprocessor.generate_preprocessing_report())
        
        # 4. Feature importance summary
        print("\n4. Feature Engineering Summary:")
        feature_summary = preprocessor.get_feature_importance_summary()
        print(f"  Original features: {len(feature_summary['original_features'])}")
        print(f"  Engineered features: {len(feature_summary['engineered_features'])}")
        print(f"  Selected features: {len(feature_summary['selected_features'])}")
        print(f"  Selected feature names: {feature_summary['selected_features'][:5]}..." 
              if len(feature_summary['selected_features']) > 5 
              else f"  Selected feature names: {feature_summary['selected_features']}")
        
        # 5. Test multi-target preprocessing
        print("\n5. Testing multi-target preprocessing...")
        X_train_multi, X_test_multi, y_train_multi, y_test_multi, preprocessor_multi = quick_preprocess_windsor_data(
            target_type='both',
            n_features=15,
            feature_selection_method='combined'
        )
        
        print(f"  Multi-target training shape: {X_train_multi.shape}")
        print(f"  Multi-target targets shape: {y_train_multi.shape}")
        
        # 6. Demonstrate serialization
        print("\n6. Testing preprocessor serialization...")
        import tempfile
        import os
        
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as tmp_file:
            temp_path = tmp_file.name
        
        try:
            # Save preprocessor
            preprocessor.save_preprocessor(temp_path)
            
            # Load preprocessor
            loaded_preprocessor = AerodynamicPreprocessor.load_preprocessor(temp_path)
            
            # Test loaded preprocessor
            X_test_transform = loaded_preprocessor.transform(X_test)
            print(f"  Serialization test passed: {X_test_transform.shape}")
            
        finally:
            # Clean up temporary file
            if os.path.exists(temp_path):
                os.unlink(temp_path)
        
        print("\n7. Component Testing:")
        
        # Test individual components
        features, targets = loader.get_feature_target_split()
        
        # Feature engineering
        feature_engineer = AerodynamicFeatureEngineer()
        X_engineered = feature_engineer.fit_transform(features)
        print(f"  Feature engineering: {features.shape} → {X_engineered.shape}")
        
        # Data validation
        validator = DataValidator()
        validation_results = validator.validate_dataset(features, targets)
        validation_report = validator.generate_validation_report()
        print("  Data validation completed")
        
        # Domain-aware scaling
        scaler = AerodynamicScaler(scaling_strategy='mixed')
        X_scaled = scaler.fit_transform(X_engineered)
        print(f"  Aerodynamic scaling completed: {X_scaled.shape}")
        
        # Feature selection
        selector = AerodynamicFeatureSelector(method='combined', k_features=10)
        X_selected = selector.fit_transform(X_scaled, targets)
        print(f"  Feature selection: {X_scaled.shape} → {X_selected.shape}")
        print(f"  Selected features: {selector.selected_features_}")
        
        print("\n" + "=" * 70)
        print("✅ ENHANCED PREPROCESSING PIPELINE DEMO COMPLETED SUCCESSFULLY")
        print("=" * 70)
        
        print(f"\nPipeline Features:")
        print(f"  ✅ Domain-aware feature engineering")
        print(f"  ✅ Aerodynamic parameter scaling")
        print(f"  ✅ Multi-method feature selection")
        print(f"  ✅ Physical constraint validation")
        print(f"  ✅ Outlier detection and handling")
        print(f"  ✅ Performance-based stratification")
        print(f"  ✅ Pipeline serialization")
        print(f"  ✅ Comprehensive reporting")
        
    except Exception as e:
        logger.error(f"Error during enhanced preprocessing demo: {e}")
        import traceback
        traceback.print_exc()
        raise