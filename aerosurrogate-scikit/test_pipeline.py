#!/usr/bin/env python3
"""
Simple test script to validate our AeroSurrogate-Scikit pipeline works end-to-end.
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

from data_processing import WindsorDataLoader, AerodynamicPreprocessor

def test_pipeline():
    """Test the complete pipeline from data loading to model training."""
    
    print("🧪 Testing AeroSurrogate-Scikit Pipeline...")
    
    # 1. Load data
    print("\n1️⃣ Loading Windsor body dataset...")
    loader = WindsorDataLoader()
    features, targets = loader.get_feature_target_split()
    print(f"   ✅ Loaded {features.shape[0]} samples with {features.shape[1]} features")
    print(f"   ✅ Targets: Cd (drag) and Cl (lift)")
    
    # 2. Test preprocessing
    print("\n2️⃣ Testing preprocessing pipeline...")
    from data_processing import quick_preprocess_windsor_data
    X_processed, X_test, y_processed, y_test, preprocessor = quick_preprocess_windsor_data(target_type='drag', test_size=0.2, random_state=42)
    print(f"   ✅ Training set: {X_processed.shape}")
    print(f"   ✅ Test set: {X_test.shape}")
    print(f"   ✅ Feature engineering: {features.shape[1]} → {X_processed.shape[1]} features")
    
    # 3. Train simple models
    print("\n3️⃣ Training models...")
    
    # Convert to arrays if needed
    if hasattr(y_processed, 'values'):
        y_processed_array = y_processed.values.ravel()
        y_test_array = y_test.values.ravel()
    else:
        y_processed_array = np.array(y_processed).ravel()
        y_test_array = np.array(y_test).ravel()
    
    # Linear model
    lr_model = LinearRegression()
    lr_model.fit(X_processed, y_processed_array)
    lr_pred = lr_model.predict(X_test)
    lr_r2 = r2_score(y_test_array, lr_pred)
    lr_rmse = np.sqrt(mean_squared_error(y_test_array, lr_pred))
    print(f"   📊 Linear Regression - R²: {lr_r2:.3f}, RMSE: {lr_rmse:.4f}")
    
    # Random Forest
    rf_model = RandomForestRegressor(n_estimators=50, random_state=42)
    rf_model.fit(X_processed, y_processed_array)
    rf_pred = rf_model.predict(X_test)
    rf_r2 = r2_score(y_test_array, rf_pred)
    rf_rmse = np.sqrt(mean_squared_error(y_test_array, rf_pred))
    print(f"   📊 Random Forest - R²: {rf_r2:.3f}, RMSE: {rf_rmse:.4f}")
    
    # 4. Test with lift coefficient
    print("\n4️⃣ Testing lift coefficient prediction...")
    X_processed_lift, X_test_lift, y_processed_lift, y_test_lift, preprocessor_lift = quick_preprocess_windsor_data(target_type='lift', test_size=0.2, random_state=42)
    
    # Convert lift data to arrays
    if hasattr(y_processed_lift, 'values'):
        y_processed_lift_array = y_processed_lift.values.ravel()
        y_test_lift_array = y_test_lift.values.ravel()
    else:
        y_processed_lift_array = np.array(y_processed_lift).ravel()
        y_test_lift_array = np.array(y_test_lift).ravel()
    
    rf_lift = RandomForestRegressor(n_estimators=50, random_state=42)
    rf_lift.fit(X_processed_lift, y_processed_lift_array)
    rf_lift_pred = rf_lift.predict(X_test_lift)
    rf_lift_r2 = r2_score(y_test_lift_array, rf_lift_pred)
    rf_lift_rmse = np.sqrt(mean_squared_error(y_test_lift_array, rf_lift_pred))
    print(f"   📊 Lift Random Forest - R²: {rf_lift_r2:.3f}, RMSE: {rf_lift_rmse:.4f}")
    
    print("\n🎉 Pipeline test completed successfully!")
    print(f"\n📈 Performance Summary:")
    print(f"   Drag Coefficient (Cd) Prediction:")
    print(f"     • Linear Regression: R² = {lr_r2:.3f}")
    print(f"     • Random Forest: R² = {rf_r2:.3f}")
    print(f"   Lift Coefficient (Cl) Prediction:")
    print(f"     • Random Forest: R² = {rf_lift_r2:.3f}")
    
    if rf_r2 > 0.8 and rf_lift_r2 > 0.7:
        print("\n✅ PIPELINE VALIDATED: Ready for production use!")
        return True
    else:
        print("\n⚠️  Performance below expectations - needs optimization")
        return False

if __name__ == "__main__":
    success = test_pipeline()
    sys.exit(0 if success else 1)