#!/usr/bin/env python3
"""
Simple test script to verify the FitzHugh-Nagumo Lift & Learn setup.
"""

import sys
import os

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_imports():
    """Test that all required modules can be imported."""
    print("Testing imports...")
    
    try:
        import numpy as np
        print(f"✓ NumPy {np.__version__}")
    except ImportError as e:
        print(f"✗ NumPy: {e}")
        return False
    
    try:
        import scipy
        print(f"✓ SciPy {scipy.__version__}")
    except ImportError as e:
        print(f"✗ SciPy: {e}")
        return False
    
    try:
        import matplotlib
        print(f"✓ Matplotlib {matplotlib.__version__}")
    except ImportError as e:
        print(f"✗ Matplotlib: {e}")
        return False
        
    return True

def test_solver():
    """Test the FitzHugh-Nagumo solver."""
    print("\nTesting FitzHugh-Nagumo solver...")
    
    try:
        from src.fitzhugh_nagumo_solver import FitzHughNagumoSolver
        import numpy as np
        
        # Create small test case
        solver = FitzHughNagumoSolver(nx=10)
        t_eval = np.linspace(0, 0.1, 6)  # Very short simulation
        
        solution = solver.solve(
            t_span=(0, 0.1),
            t_eval=t_eval,
            alpha=1.0,
            beta=1.0,
            g_func=lambda t: 0.1 * np.sin(2*np.pi*t)
        )
        
        if solution['success']:
            print(f"✓ Solver working: s1 shape {solution['s1'].shape}, s2 shape {solution['s2'].shape}")
            return True
        else:
            print(f"✗ Solver failed: {solution['message']}")
            return False
            
    except Exception as e:
        print(f"✗ Solver test failed: {e}")
        return False

def test_lifting():
    """Test the lifting transformation."""
    print("\nTesting lifting transformation...")
    
    try:
        from src.lifting_transformation import LiftingTransformation
        import numpy as np
        
        nx = 10
        lifting = LiftingTransformation(nx)
        
        # Create test state
        s1 = np.random.randn(nx) * 0.1
        s2 = np.random.randn(nx) * 0.1
        original_state = np.concatenate([s1, s2])
        
        # Test lift and unlift
        lifted_state = lifting.lift(original_state)
        recovered_state = lifting.unlift(lifted_state)
        
        error = np.linalg.norm(original_state - recovered_state)
        
        if error < 1e-10:
            print(f"✓ Lifting transformation working: error = {error:.2e}")
            return True
        else:
            print(f"✗ Lifting transformation failed: error = {error:.2e}")
            return False
            
    except Exception as e:
        print(f"✗ Lifting test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("FitzHugh-Nagumo Lift & Learn Setup Test")
    print("=" * 50)
    
    success = True
    
    # Test imports
    success &= test_imports()
    
    # Test core functionality
    success &= test_solver()
    success &= test_lifting()
    
    print("\n" + "=" * 50)
    if success:
        print("✓ All tests passed! Setup is working correctly.")
        print("\nYou can now run:")
        print("  python main.py")
    else:
        print("✗ Some tests failed. Check the error messages above.")
        return 1
        
    return 0

if __name__ == "__main__":
    sys.exit(main())