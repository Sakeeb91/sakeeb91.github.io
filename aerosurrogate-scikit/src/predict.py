"""
Prediction script for aerodynamic surrogate models.

This script provides a comprehensive interface for making predictions with trained
aerodynamic surrogate models. It includes model loading, input validation,
batch prediction capabilities, and uncertainty quantification.

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
import argparse
from datetime import datetime

# Scientific computing
from scipy import stats

# Project imports
from data_processing import AerodynamicPreprocessor, WindsorDataLoader
from model_evaluation import AerodynamicModelEvaluator
from config import MODELS_DIR, PROJECT_ROOT

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)


class AerodynamicPredictor:
    """
    Comprehensive prediction interface for aerodynamic surrogate models.
    
    Features:
    - Model loading and management
    - Input validation and preprocessing
    - Single and batch predictions
    - Uncertainty quantification
    - Physics-informed validation
    - Multiple output formats
    - Error handling and logging
    """
    
    def __init__(self, models_dir: str = MODELS_DIR):
        """
        Initialize the aerodynamic predictor.
        
        Args:
            models_dir: Directory containing trained models
        """
        self.models_dir = Path(models_dir)
        self.loaded_models = {}
        self.loaded_preprocessors = {}
        self.model_metadata = {}
        
        # Define parameter bounds for validation
        self.parameter_bounds = {
            'ratio_length_back_fast': (0.0, 1.0),
            'ratio_height_nose_windshield': (0.0, 1.0),
            'ratio_height_fast_back': (0.0, 1.0),
            'side_taper': (0.0, 45.0),
            'clearance': (10.0, 300.0),
            'bottom_taper_angle': (0.0, 30.0),
            'frontal_area': (0.05, 0.2)
        }
        
        # Required parameters
        self.required_parameters = [
            'ratio_length_back_fast',
            'ratio_height_nose_windshield', 
            'ratio_height_fast_back',
            'side_taper',
            'clearance',
            'bottom_taper_angle',
            'frontal_area'
        ]
        
        logger.info(f"Initialized AerodynamicPredictor with models directory: {self.models_dir}")
    
    def load_model(self, 
                   model_path: str, 
                   preprocessor_path: str,
                   model_name: Optional[str] = None) -> str:
        """
        Load a trained model and its preprocessor.
        
        Args:
            model_path: Path to the trained model file
            preprocessor_path: Path to the preprocessor file
            model_name: Name to assign to the model (optional)
            
        Returns:
            Model identifier for later use
        """
        if model_name is None:
            model_name = Path(model_path).stem
        
        try:
            # Load model
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            
            # Load preprocessor
            with open(preprocessor_path, 'rb') as f:
                preprocessor = pickle.load(f)
            
            # Store loaded components
            self.loaded_models[model_name] = model
            self.loaded_preprocessors[model_name] = preprocessor
            
            # Store metadata
            self.model_metadata[model_name] = {
                'model_path': str(model_path),
                'preprocessor_path': str(preprocessor_path),
                'loaded_at': datetime.now().isoformat(),
                'model_type': type(model).__name__,
                'preprocessor_type': type(preprocessor).__name__
            }
            
            logger.info(f"Successfully loaded model '{model_name}'")
            logger.info(f"  Model type: {type(model).__name__}")
            logger.info(f"  Preprocessor type: {type(preprocessor).__name__}")
            
            return model_name
            
        except Exception as e:
            logger.error(f"Failed to load model from {model_path}: {str(e)}")
            raise
    
    def load_latest_models(self, pattern: str = "*model*.pkl") -> List[str]:
        """
        Load the most recent models matching a pattern.
        
        Args:
            pattern: File pattern to match model files
            
        Returns:
            List of loaded model names
        """
        model_files = list(self.models_dir.glob(pattern))
        
        if not model_files:
            logger.warning(f"No model files found matching pattern: {pattern}")
            return []
        
        # Sort by modification time (most recent first)
        model_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        
        loaded_models = []
        
        for model_file in model_files:
            try:
                # Try to find corresponding preprocessor
                model_stem = model_file.stem
                
                # Look for preprocessor with similar name
                possible_preprocessor_patterns = [
                    model_stem.replace('model', 'preprocessor') + '.pkl',
                    model_stem.replace('model', 'preprocessor') + '.pickle',
                    f"*preprocessor*{model_stem.split('_')[-1]}.pkl" if '_' in model_stem else "preprocessor*.pkl"
                ]
                
                preprocessor_file = None
                for pattern in possible_preprocessor_patterns:
                    matches = list(self.models_dir.glob(pattern))
                    if matches:
                        preprocessor_file = matches[0]
                        break
                
                if preprocessor_file is None:
                    logger.warning(f"No preprocessor found for model {model_file}")
                    continue
                
                model_name = self.load_model(model_file, preprocessor_file)
                loaded_models.append(model_name)
                
            except Exception as e:
                logger.error(f"Failed to load model {model_file}: {str(e)}")
                continue
        
        logger.info(f"Loaded {len(loaded_models)} models: {loaded_models}")
        return loaded_models
    
    def validate_input_parameters(self, parameters: Union[Dict, pd.DataFrame]) -> Tuple[bool, str]:
        """
        Validate input parameters for aerodynamic prediction.
        
        Args:
            parameters: Dictionary or DataFrame with geometric parameters
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if isinstance(parameters, dict):
            param_dict = parameters
        elif isinstance(parameters, pd.DataFrame):
            if len(parameters) != 1:
                return False, "DataFrame must contain exactly one row for single prediction"
            param_dict = parameters.iloc[0].to_dict()
        else:
            return False, "Parameters must be a dictionary or single-row DataFrame"
        
        # Check required parameters
        missing_params = set(self.required_parameters) - set(param_dict.keys())
        if missing_params:
            return False, f"Missing required parameters: {missing_params}"
        
        # Check parameter bounds
        for param, value in param_dict.items():
            if param in self.parameter_bounds:
                bounds = self.parameter_bounds[param]
                if not (bounds[0] <= value <= bounds[1]):
                    return False, f"Parameter '{param}' = {value} is outside valid range {bounds}"
        
        # Physics-based validation
        try:
            # Check for reasonable geometric relationships
            if param_dict.get('ratio_height_fast_back', 0) > param_dict.get('ratio_height_nose_windshield', 1):
                logger.warning("Fastback height ratio is greater than windshield height ratio - unusual configuration")
            
            if param_dict.get('clearance', 100) < 20:
                logger.warning("Very low clearance detected - may not be physically realistic")
            
            if param_dict.get('frontal_area', 0.1) > 0.15:
                logger.warning("Large frontal area detected - may indicate unusually large vehicle")
                
        except Exception as e:
            logger.warning(f"Physics validation warning: {str(e)}")
        
        return True, "Parameters are valid"
    
    def predict_single(self, 
                      parameters: Union[Dict, pd.DataFrame],
                      model_name: Optional[str] = None,
                      include_uncertainty: bool = False,
                      validate_physics: bool = True) -> Dict[str, Any]:
        """
        Make a single prediction with a specified model.
        
        Args:
            parameters: Geometric parameters for prediction
            model_name: Name of the model to use (uses first available if None)
            include_uncertainty: Whether to estimate prediction uncertainty
            validate_physics: Whether to validate physics consistency
            
        Returns:
            Dictionary with prediction results
        """
        # Validate inputs
        is_valid, error_msg = self.validate_input_parameters(parameters)
        if not is_valid:
            raise ValueError(f"Invalid input parameters: {error_msg}")
        
        # Select model
        if model_name is None:
            if not self.loaded_models:
                raise ValueError("No models loaded. Call load_model() first.")
            model_name = list(self.loaded_models.keys())[0]
        elif model_name not in self.loaded_models:
            raise ValueError(f"Model '{model_name}' not found. Available models: {list(self.loaded_models.keys())}")
        
        model = self.loaded_models[model_name]
        preprocessor = self.loaded_preprocessors[model_name]
        
        try:
            # Convert to DataFrame if needed
            if isinstance(parameters, dict):
                input_df = pd.DataFrame([parameters])
            else:
                input_df = parameters.copy()
            
            # Preprocess
            X_processed = preprocessor.transform(input_df)
            
            # Make prediction
            prediction = model.predict(X_processed)
            
            # Format results
            if prediction.ndim > 1 and prediction.shape[1] > 1:
                # Multi-target prediction
                if prediction.shape[1] == 2:
                    result = {
                        'cd': float(prediction[0, 0]),
                        'cl': float(prediction[0, 1])
                    }
                else:
                    result = {f'target_{i}': float(prediction[0, i]) for i in range(prediction.shape[1])}
            else:
                # Single target prediction
                result = {'prediction': float(prediction[0])}
            
            # Add metadata
            result.update({
                'model_used': model_name,
                'prediction_timestamp': datetime.now().isoformat(),
                'input_parameters': parameters if isinstance(parameters, dict) else parameters.iloc[0].to_dict()
            })
            
            # Physics validation
            if validate_physics:
                physics_check = self._validate_prediction_physics(result, parameters)
                result['physics_validation'] = physics_check
            
            # Uncertainty estimation (simplified)
            if include_uncertainty:
                uncertainty = self._estimate_prediction_uncertainty(X_processed, model, preprocessor)
                result['uncertainty'] = uncertainty
            
            logger.info(f"Prediction completed with model '{model_name}'")
            return result
            
        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            raise
    
    def predict_batch(self, 
                     parameters_df: pd.DataFrame,
                     model_name: Optional[str] = None,
                     include_uncertainty: bool = False,
                     validate_physics: bool = True) -> pd.DataFrame:
        """
        Make batch predictions for multiple parameter sets.
        
        Args:
            parameters_df: DataFrame with geometric parameters
            model_name: Name of the model to use
            include_uncertainty: Whether to estimate prediction uncertainty
            validate_physics: Whether to validate physics consistency
            
        Returns:
            DataFrame with prediction results
        """
        logger.info(f"Starting batch prediction for {len(parameters_df)} samples...")
        
        # Select model
        if model_name is None:
            if not self.loaded_models:
                raise ValueError("No models loaded. Call load_model() first.")
            model_name = list(self.loaded_models.keys())[0]
        elif model_name not in self.loaded_models:
            raise ValueError(f"Model '{model_name}' not found. Available models: {list(self.loaded_models.keys())}")
        
        model = self.loaded_models[model_name]
        preprocessor = self.loaded_preprocessors[model_name]
        
        # Validate all inputs
        invalid_rows = []
        for idx, row in parameters_df.iterrows():
            is_valid, error_msg = self.validate_input_parameters(row.to_dict())
            if not is_valid:
                invalid_rows.append((idx, error_msg))
        
        if invalid_rows:
            logger.warning(f"Found {len(invalid_rows)} invalid rows:")
            for idx, msg in invalid_rows[:5]:  # Show first 5
                logger.warning(f"  Row {idx}: {msg}")
            
            if len(invalid_rows) == len(parameters_df):
                raise ValueError("All input rows are invalid")
        
        try:
            # Preprocess
            X_processed = preprocessor.transform(parameters_df)
            
            # Make predictions
            predictions = model.predict(X_processed)
            
            # Create results DataFrame
            results_df = parameters_df.copy()
            
            # Add predictions
            if predictions.ndim > 1 and predictions.shape[1] > 1:
                # Multi-target predictions
                if predictions.shape[1] == 2:
                    results_df['cd_predicted'] = predictions[:, 0]
                    results_df['cl_predicted'] = predictions[:, 1]
                else:
                    for i in range(predictions.shape[1]):
                        results_df[f'target_{i}_predicted'] = predictions[:, i]
            else:
                # Single target predictions
                results_df['prediction'] = predictions
            
            # Add metadata
            results_df['model_used'] = model_name
            results_df['prediction_timestamp'] = datetime.now().isoformat()
            
            # Physics validation
            if validate_physics:
                physics_scores = []
                for idx, row in parameters_df.iterrows():
                    if predictions.ndim > 1 and predictions.shape[1] > 1:
                        pred_dict = {'cd': predictions[idx, 0], 'cl': predictions[idx, 1]}
                    else:
                        pred_dict = {'prediction': predictions[idx]}
                    
                    physics_check = self._validate_prediction_physics(pred_dict, row.to_dict())
                    physics_scores.append(physics_check.get('overall_score', 1.0))
                
                results_df['physics_validation_score'] = physics_scores
            
            # Uncertainty estimation
            if include_uncertainty:
                uncertainty_estimates = self._estimate_batch_uncertainty(X_processed, model, preprocessor)
                
                if predictions.ndim > 1 and predictions.shape[1] > 1:
                    results_df['cd_uncertainty'] = uncertainty_estimates[:, 0]
                    results_df['cl_uncertainty'] = uncertainty_estimates[:, 1]
                else:
                    results_df['prediction_uncertainty'] = uncertainty_estimates
            
            logger.info(f"Batch prediction completed for {len(results_df)} samples")
            return results_df
            
        except Exception as e:
            logger.error(f"Batch prediction failed: {str(e)}")
            raise
    
    def _validate_prediction_physics(self, prediction_dict: Dict, parameters: Dict) -> Dict[str, Any]:
        """
        Validate physics consistency of predictions.
        """
        validation = {'tests': {}, 'warnings': [], 'overall_score': 1.0}
        
        try:
            # Test 1: Drag coefficient range
            if 'cd' in prediction_dict:
                cd = prediction_dict['cd']
                if cd < 0.1 or cd > 1.0:
                    validation['warnings'].append(f"Drag coefficient {cd:.3f} outside typical automotive range [0.1, 1.0]")
                    validation['overall_score'] *= 0.8
                validation['tests']['cd_range'] = 0.1 <= cd <= 1.0
            
            # Test 2: Lift coefficient range
            if 'cl' in prediction_dict:
                cl = prediction_dict['cl']
                if cl < -2.0 or cl > 2.0:
                    validation['warnings'].append(f"Lift coefficient {cl:.3f} outside reasonable range [-2.0, 2.0]")
                    validation['overall_score'] *= 0.8
                validation['tests']['cl_range'] = -2.0 <= cl <= 2.0
            
            # Test 3: Clearance effect on lift
            if 'cl' in prediction_dict and 'clearance' in parameters:
                clearance = parameters['clearance']
                cl = prediction_dict['cl']
                
                # Very low clearance should typically produce more downforce (negative lift)
                if clearance < 50 and cl > 0.5:
                    validation['warnings'].append("Low clearance with positive lift is unusual - may indicate ground effect anomaly")
                    validation['overall_score'] *= 0.9
            
            # Test 4: Frontal area effect on drag
            if 'cd' in prediction_dict and 'frontal_area' in parameters:
                frontal_area = parameters['frontal_area']
                cd = prediction_dict['cd']
                
                # Large frontal area with very low drag is suspicious
                if frontal_area > 0.12 and cd < 0.2:
                    validation['warnings'].append("Large frontal area with very low drag is unusual")
                    validation['overall_score'] *= 0.9
            
            # Test 5: Geometric consistency
            if all(param in parameters for param in ['ratio_height_fast_back', 'ratio_height_nose_windshield']):
                if parameters['ratio_height_fast_back'] > parameters['ratio_height_nose_windshield'] * 1.5:
                    validation['warnings'].append("Unusual fastback geometry detected")
                    validation['overall_score'] *= 0.95
            
        except Exception as e:
            validation['warnings'].append(f"Physics validation error: {str(e)}")
            validation['overall_score'] *= 0.7
        
        return validation
    
    def _estimate_prediction_uncertainty(self, X_processed: np.ndarray, model, preprocessor) -> Dict[str, float]:
        """
        Estimate prediction uncertainty (simplified approach).
        """
        try:
            # For ensemble models, use prediction variance
            if hasattr(model, 'estimators_'):
                # Tree-based ensemble
                individual_predictions = []
                for estimator in model.estimators_:
                    pred = estimator.predict(X_processed)
                    individual_predictions.append(pred)
                
                individual_predictions = np.array(individual_predictions)
                
                if individual_predictions.ndim == 3:
                    # Multi-target
                    uncertainty = {
                        'cd_std': float(np.std(individual_predictions[:, 0, 0])),
                        'cl_std': float(np.std(individual_predictions[:, 0, 1])) if individual_predictions.shape[2] > 1 else 0.0
                    }
                else:
                    # Single target
                    uncertainty = {'std': float(np.std(individual_predictions[:, 0]))}
                
            else:
                # For other models, use a simple heuristic based on training performance
                # This is a placeholder - in practice, you'd use bootstrap or other methods
                uncertainty = {'estimated_std': 0.05}  # 5% uncertainty estimate
            
            return uncertainty
            
        except Exception as e:
            logger.warning(f"Uncertainty estimation failed: {str(e)}")
            return {'error': str(e)}
    
    def _estimate_batch_uncertainty(self, X_processed: np.ndarray, model, preprocessor) -> np.ndarray:
        """
        Estimate uncertainty for batch predictions.
        """
        try:
            if hasattr(model, 'estimators_'):
                # Tree-based ensemble
                all_predictions = []
                for estimator in model.estimators_:
                    pred = estimator.predict(X_processed)
                    all_predictions.append(pred)
                
                all_predictions = np.array(all_predictions)
                
                # Calculate standard deviation across estimators
                uncertainty = np.std(all_predictions, axis=0)
                return uncertainty
            else:
                # Placeholder uncertainty for non-ensemble models
                predictions = model.predict(X_processed)
                if predictions.ndim > 1:
                    uncertainty = np.full_like(predictions, 0.05)
                else:
                    uncertainty = np.full_like(predictions, 0.05)
                return uncertainty
                
        except Exception as e:
            logger.warning(f"Batch uncertainty estimation failed: {str(e)}")
            # Return zero uncertainty as fallback
            predictions = model.predict(X_processed)
            return np.zeros_like(predictions)
    
    def export_predictions(self, 
                          results: Union[Dict, pd.DataFrame],
                          output_path: str,
                          format: str = 'csv') -> None:
        """
        Export predictions to file.
        
        Args:
            results: Prediction results (single dict or DataFrame)
            output_path: Output file path
            format: Output format ('csv', 'json', 'excel')
        """
        output_path = Path(output_path)
        
        try:
            if isinstance(results, dict):
                # Single prediction
                if format.lower() == 'json':
                    with open(output_path, 'w') as f:
                        json.dump(results, f, indent=2, default=str)
                elif format.lower() == 'csv':
                    df = pd.DataFrame([results])
                    df.to_csv(output_path, index=False)
                else:
                    raise ValueError(f"Unsupported format for single prediction: {format}")
                    
            elif isinstance(results, pd.DataFrame):
                # Batch predictions
                if format.lower() == 'csv':
                    results.to_csv(output_path, index=False)
                elif format.lower() == 'json':
                    results.to_json(output_path, orient='records', indent=2)
                elif format.lower() in ['excel', 'xlsx']:
                    results.to_excel(output_path, index=False)
                else:
                    raise ValueError(f"Unsupported format: {format}")
            
            logger.info(f"Predictions exported to {output_path}")
            
        except Exception as e:
            logger.error(f"Failed to export predictions: {str(e)}")
            raise
    
    def get_model_info(self, model_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Get information about loaded models.
        
        Args:
            model_name: Specific model name (returns info for all if None)
            
        Returns:
            Dictionary with model information
        """
        if model_name is None:
            return {
                'available_models': list(self.loaded_models.keys()),
                'model_metadata': self.model_metadata
            }
        elif model_name in self.loaded_models:
            model = self.loaded_models[model_name]
            preprocessor = self.loaded_preprocessors[model_name]
            
            info = {
                'model_name': model_name,
                'model_type': type(model).__name__,
                'preprocessor_type': type(preprocessor).__name__,
                'metadata': self.model_metadata.get(model_name, {}),
                'capabilities': {
                    'single_prediction': True,
                    'batch_prediction': True,
                    'uncertainty_estimation': hasattr(model, 'estimators_'),
                    'multi_target': True  # Assume multi-target capability
                }
            }
            
            # Add model-specific information
            if hasattr(model, 'feature_importances_'):
                info['has_feature_importance'] = True
            
            if hasattr(model, 'n_estimators'):
                info['n_estimators'] = model.n_estimators
            
            return info
        else:
            raise ValueError(f"Model '{model_name}' not found")


def create_example_input() -> Dict[str, float]:
    """
    Create an example input for testing predictions.
    
    Returns:
        Dictionary with example geometric parameters
    """
    return {
        'ratio_length_back_fast': 0.5,
        'ratio_height_nose_windshield': 0.3,
        'ratio_height_fast_back': 0.2,
        'side_taper': 15.0,
        'clearance': 100.0,
        'bottom_taper_angle': 10.0,
        'frontal_area': 0.08
    }


def main():
    """
    Command-line interface for aerodynamic predictions.
    """
    parser = argparse.ArgumentParser(description='Make predictions with aerodynamic surrogate models')
    
    # Model selection
    parser.add_argument('--model-path', type=str, required=True,
                       help='Path to the trained model file')
    parser.add_argument('--preprocessor-path', type=str, required=True,
                       help='Path to the preprocessor file')
    parser.add_argument('--model-name', type=str, default=None,
                       help='Name for the loaded model')
    
    # Input options
    parser.add_argument('--input-file', type=str, default=None,
                       help='CSV file with input parameters for batch prediction')
    parser.add_argument('--single-prediction', action='store_true',
                       help='Make a single prediction with example parameters')
    
    # Output options
    parser.add_argument('--output-file', type=str, default=None,
                       help='Output file for predictions')
    parser.add_argument('--output-format', choices=['csv', 'json', 'excel'], default='csv',
                       help='Output file format')
    
    # Prediction options
    parser.add_argument('--include-uncertainty', action='store_true',
                       help='Include uncertainty estimation')
    parser.add_argument('--validate-physics', action='store_true', default=True,
                       help='Validate physics consistency')
    
    # Individual parameter inputs
    parser.add_argument('--ratio-length-back-fast', type=float, default=0.5)
    parser.add_argument('--ratio-height-nose-windshield', type=float, default=0.3)
    parser.add_argument('--ratio-height-fast-back', type=float, default=0.2)
    parser.add_argument('--side-taper', type=float, default=15.0)
    parser.add_argument('--clearance', type=float, default=100.0)
    parser.add_argument('--bottom-taper-angle', type=float, default=10.0)
    parser.add_argument('--frontal-area', type=float, default=0.08)
    
    args = parser.parse_args()
    
    try:
        # Initialize predictor
        predictor = AerodynamicPredictor()
        
        # Load model
        model_name = predictor.load_model(
            args.model_path, 
            args.preprocessor_path, 
            args.model_name
        )
        
        # Get model info
        model_info = predictor.get_model_info(model_name)
        print("\\nModel Information:")
        print(f"  Model Type: {model_info['model_type']}")
        print(f"  Preprocessor Type: {model_info['preprocessor_type']}")
        print(f"  Uncertainty Estimation: {model_info['capabilities']['uncertainty_estimation']}")
        
        if args.input_file:
            # Batch prediction
            print(f"\\nLoading input data from {args.input_file}...")
            input_df = pd.read_csv(args.input_file)
            
            print(f"Making batch predictions for {len(input_df)} samples...")
            results = predictor.predict_batch(
                input_df,
                model_name=model_name,
                include_uncertainty=args.include_uncertainty,
                validate_physics=args.validate_physics
            )
            
            print("\\nBatch prediction completed!")
            print(f"Results shape: {results.shape}")
            
            if args.output_file:
                predictor.export_predictions(results, args.output_file, args.output_format)
            else:
                print(results.head())
        
        elif args.single_prediction:
            # Single prediction with example parameters
            example_params = create_example_input()
            
            print("\\nMaking single prediction with example parameters:")
            for param, value in example_params.items():
                print(f"  {param}: {value}")
            
            result = predictor.predict_single(
                example_params,
                model_name=model_name,
                include_uncertainty=args.include_uncertainty,
                validate_physics=args.validate_physics
            )
            
            print("\\nPrediction Results:")
            for key, value in result.items():
                if key not in ['input_parameters', 'physics_validation', 'uncertainty']:
                    print(f"  {key}: {value}")
            
            if 'physics_validation' in result:
                physics = result['physics_validation']
                print(f"\\nPhysics Validation Score: {physics['overall_score']:.3f}")
                if physics['warnings']:
                    print("Physics Warnings:")
                    for warning in physics['warnings']:
                        print(f"  - {warning}")
            
            if args.output_file:
                predictor.export_predictions(result, args.output_file, args.output_format)
        
        else:
            # Single prediction with command-line parameters
            params = {
                'ratio_length_back_fast': args.ratio_length_back_fast,
                'ratio_height_nose_windshield': args.ratio_height_nose_windshield,
                'ratio_height_fast_back': args.ratio_height_fast_back,
                'side_taper': args.side_taper,
                'clearance': args.clearance,
                'bottom_taper_angle': args.bottom_taper_angle,
                'frontal_area': args.frontal_area
            }
            
            print("\\nMaking prediction with provided parameters:")
            for param, value in params.items():
                print(f"  {param}: {value}")
            
            result = predictor.predict_single(
                params,
                model_name=model_name,
                include_uncertainty=args.include_uncertainty,
                validate_physics=args.validate_physics
            )
            
            print("\\nPrediction Results:")
            if 'cd' in result and 'cl' in result:
                print(f"  Drag Coefficient (Cd): {result['cd']:.4f}")
                print(f"  Lift Coefficient (Cl): {result['cl']:.4f}")
            elif 'prediction' in result:
                print(f"  Prediction: {result['prediction']:.4f}")
            
            # Physical interpretation
            if 'cd' in result and 'cl' in result:
                print("\\nPhysical Interpretation:")
                if result['cd'] < 0.3:
                    print("  🚗 Low drag configuration - excellent for fuel efficiency")
                elif result['cd'] < 0.4:
                    print("  🚙 Moderate drag - typical automotive range")
                else:
                    print("  🚛 High drag - may need aerodynamic optimization")
                
                if result['cl'] < -0.1:
                    print("  ⬇️ Generates downforce - improves high-speed stability")
                elif result['cl'] > 0.1:
                    print("  ⬆️ Generates lift - may reduce traction at high speeds")
                else:
                    print("  ➡️ Neutral lift - balanced aerodynamic behavior")
            
            if args.output_file:
                predictor.export_predictions(result, args.output_file, args.output_format)
        
        print("\\nPrediction completed successfully!")
        
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()