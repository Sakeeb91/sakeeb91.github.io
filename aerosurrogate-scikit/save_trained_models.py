#!/usr/bin/env python3
"""
Save Trained Models for AeroSurrogate-Scikit
===========================================

This script trains the best-performing models and saves them to the models/ directory
for production use and deployment.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import numpy as np
import pandas as pd
import joblib
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# Import our modules
from data_processing import quick_preprocess_windsor_data

def train_and_save_models():
    """Train best models and save them with metadata."""
    print("🤖 Training and Saving Best-Performing Models...")
    print("=" * 50)
    
    # Create models directory if it doesn't exist
    models_dir = "models"
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
    
    # Preprocess data for both targets
    print("\n📊 Loading and preprocessing data...")
    X_train_drag, X_test_drag, y_train_drag, y_test_drag, preprocessor_drag = quick_preprocess_windsor_data(
        target_type='drag', test_size=0.2, random_state=42
    )
    X_train_lift, X_test_lift, y_train_lift, y_test_lift, preprocessor_lift = quick_preprocess_windsor_data(
        target_type='lift', test_size=0.2, random_state=42
    )
    
    # Convert to arrays
    y_train_drag = np.array(y_train_drag).ravel()
    y_test_drag = np.array(y_test_drag).ravel()
    y_train_lift = np.array(y_train_lift).ravel()
    y_test_lift = np.array(y_test_lift).ravel()
    
    print(f"✅ Data loaded: {X_train_drag.shape[0]} training samples")
    print(f"✅ Drag features: {X_train_drag.shape[1]}, Lift features: {X_train_lift.shape[1]}")
    
    # Define best models based on our analysis
    models_config = {
        'drag': {
            'gradient_boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'random_forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'ridge': Ridge(alpha=1.0)
        },
        'lift': {
            'gradient_boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'random_forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'linear': LinearRegression()
        }
    }
    
    # Train and save models
    model_metadata = {}
    
    # Train drag models
    print("\n🎯 Training Drag Coefficient Models...")
    for model_name, model in models_config['drag'].items():
        print(f"  Training {model_name}...")
        
        # Train model
        model.fit(X_train_drag, y_train_drag)
        
        # Evaluate
        y_pred = model.predict(X_test_drag)
        r2 = r2_score(y_test_drag, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test_drag, y_pred))
        mae = mean_absolute_error(y_test_drag, y_pred)
        
        print(f"    R² = {r2:.3f}, RMSE = {rmse:.4f}, MAE = {mae:.4f}")
        
        # Save model
        model_filename = f"drag_{model_name}_model.pkl"
        model_path = os.path.join(models_dir, model_filename)
        joblib.dump(model, model_path)
        
        # Store metadata
        model_metadata[f"drag_{model_name}"] = {
            'filename': model_filename,
            'target': 'drag_coefficient_cd',
            'r2_score': r2,
            'rmse': rmse,
            'mae': mae,
            'n_features': X_train_drag.shape[1],
            'training_samples': X_train_drag.shape[0],
            'test_samples': X_test_drag.shape[0],
            'trained_date': datetime.now().isoformat()
        }
        
        print(f"    ✅ Saved: {model_path}")
    
    # Train lift models
    print("\n🎯 Training Lift Coefficient Models...")
    for model_name, model in models_config['lift'].items():
        print(f"  Training {model_name}...")
        
        # Train model
        model.fit(X_train_lift, y_train_lift)
        
        # Evaluate
        y_pred = model.predict(X_test_lift)
        r2 = r2_score(y_test_lift, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test_lift, y_pred))
        mae = mean_absolute_error(y_test_lift, y_pred)
        
        print(f"    R² = {r2:.3f}, RMSE = {rmse:.4f}, MAE = {mae:.4f}")
        
        # Save model
        model_filename = f"lift_{model_name}_model.pkl"
        model_path = os.path.join(models_dir, model_filename)
        joblib.dump(model, model_path)
        
        # Store metadata
        model_metadata[f"lift_{model_name}"] = {
            'filename': model_filename,
            'target': 'lift_coefficient_cl',
            'r2_score': r2,
            'rmse': rmse,
            'mae': mae,
            'n_features': X_train_lift.shape[1],
            'training_samples': X_train_lift.shape[0],
            'test_samples': X_test_lift.shape[0],
            'trained_date': datetime.now().isoformat()
        }
        
        print(f"    ✅ Saved: {model_path}")
    
    # Save preprocessors
    print("\n⚙️ Saving Preprocessors...")
    joblib.dump(preprocessor_drag, os.path.join(models_dir, "drag_preprocessor.pkl"))
    joblib.dump(preprocessor_lift, os.path.join(models_dir, "lift_preprocessor.pkl"))
    print("    ✅ Saved: drag_preprocessor.pkl")
    print("    ✅ Saved: lift_preprocessor.pkl")
    
    # Save metadata
    metadata_df = pd.DataFrame.from_dict(model_metadata, orient='index')
    metadata_df.to_csv(os.path.join(models_dir, "model_metadata.csv"))
    
    # Create model registry
    model_registry = {
        'production_models': {
            'drag_prediction': {
                'best_model': 'drag_gradient_boosting_model.pkl',
                'preprocessor': 'drag_preprocessor.pkl',
                'performance': {
                    'r2': model_metadata['drag_gradient_boosting']['r2_score'],
                    'rmse': model_metadata['drag_gradient_boosting']['rmse']
                }
            },
            'lift_prediction': {
                'best_model': 'lift_gradient_boosting_model.pkl',
                'preprocessor': 'lift_preprocessor.pkl',
                'performance': {
                    'r2': model_metadata['lift_gradient_boosting']['r2_score'],
                    'rmse': model_metadata['lift_gradient_boosting']['rmse']
                }
            }
        },
        'alternative_models': {
            'drag': ['drag_random_forest_model.pkl', 'drag_ridge_model.pkl'],
            'lift': ['lift_random_forest_model.pkl', 'lift_linear_model.pkl']
        },
        'last_updated': datetime.now().isoformat(),
        'dataset_info': {
            'total_samples': 355,
            'training_samples': X_train_drag.shape[0],
            'test_samples': X_test_drag.shape[0],
            'source': 'Windsor body CFD simulations'
        }
    }
    
    # Save registry as JSON
    import json
    with open(os.path.join(models_dir, "model_registry.json"), 'w') as f:
        json.dump(model_registry, f, indent=2)
    
    print("\n📋 Model Registry Created...")
    print("    ✅ Saved: model_metadata.csv")
    print("    ✅ Saved: model_registry.json")
    
    # Create usage example
    usage_example = f'''
# AeroSurrogate-Scikit Model Usage Example

import joblib
import numpy as np

# Load best models and preprocessors
drag_model = joblib.load('models/drag_gradient_boosting_model.pkl')
lift_model = joblib.load('models/lift_gradient_boosting_model.pkl')
drag_preprocessor = joblib.load('models/drag_preprocessor.pkl')
lift_preprocessor = joblib.load('models/lift_preprocessor.pkl')

# Example geometric parameters (Windsor body)
geometry = np.array([[
    0.3,    # ratio_length_back_fast
    0.5,    # ratio_height_nose_windshield  
    0.4,    # ratio_height_fast_back
    75.0,   # side_taper (degrees)
    100.0,  # clearance (mm)
    25.0,   # bottom_taper_angle (degrees)
    0.116   # frontal_area (m²)
]])

# Make predictions
drag_prediction = drag_model.predict(geometry)
lift_prediction = lift_model.predict(geometry)

print(f"Predicted Drag Coefficient (Cd): {{drag_prediction[0]:.3f}}")
print(f"Predicted Lift Coefficient (Cl): {{lift_prediction[0]:.3f}}")

# Model performance
print(f"\\nDrag Model Performance: R² = {model_metadata['drag_gradient_boosting']['r2_score']:.3f}")
print(f"Lift Model Performance: R² = {model_metadata['lift_gradient_boosting']['r2_score']:.3f}")
'''
    
    with open(os.path.join(models_dir, "usage_example.py"), 'w') as f:
        f.write(usage_example)
    
    print("    ✅ Saved: usage_example.py")
    
    # Summary
    print(f"\n🎉 Model Training Complete!")
    print(f"📁 Models saved in '{models_dir}/' directory:")
    print(f"   • 6 trained models (3 drag + 3 lift)")
    print(f"   • 2 preprocessors")
    print(f"   • Model metadata and registry")
    print(f"   • Usage example")
    
    print(f"\n📈 Best Model Performance:")
    print(f"   • Drag (Gradient Boosting): R² = {model_metadata['drag_gradient_boosting']['r2_score']:.3f}")
    print(f"   • Lift (Gradient Boosting): R² = {model_metadata['lift_gradient_boosting']['r2_score']:.3f}")
    
    return model_metadata

if __name__ == "__main__":
    metadata = train_and_save_models()