
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

print(f"Predicted Drag Coefficient (Cd): {drag_prediction[0]:.3f}")
print(f"Predicted Lift Coefficient (Cl): {lift_prediction[0]:.3f}")

# Model performance
print(f"\nDrag Model Performance: R² = 0.201")
print(f"Lift Model Performance: R² = 0.503")
