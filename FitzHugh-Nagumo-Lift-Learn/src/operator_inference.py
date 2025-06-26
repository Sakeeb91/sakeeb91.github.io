"""
Phase 4: Operator Inference (The "Learn" Step)

This module implements the non-intrusive operator inference method to learn
the reduced-order model operators from data. The key idea is to learn the
operators Â, Ĥ, B̂ such that:

dŵ/dt = Âŵ + Ĥ(ŵ⊗ŵ) + B̂u

where ŵ is the reduced state, u is the input, and ⊗ is the Kronecker product.
"""

import numpy as np
from typing import Dict, Tuple, Optional, Callable
from scipy.optimize import nnls
from sklearn.linear_model import Ridge, Lasso
import matplotlib.pyplot as plt
from .fitzhugh_nagumo_solver import FitzHughNagumoSolver
from .lifting_transformation import LiftingTransformation
from .pod_reduction import PODReducer


class OperatorInference:
    """Implements operator inference for learning reduced-order models."""
    
    def __init__(self, solver: FitzHughNagumoSolver, 
                 lifting: LiftingTransformation,
                 pod: PODReducer):
        """
        Initialize operator inference.
        
        Args:
            solver: High-fidelity solver (for derivative computation)
            lifting: Lifting transformation
            pod: POD reducer
        """
        self.solver = solver
        self.lifting = lifting
        self.pod = pod
        self.operators = {}
        self.regularization = None
    
    def _compute_derivatives_nonintrusive(self, 
                                        reduced_states: np.ndarray,
                                        parameters: np.ndarray,
                                        g_func: Callable[[float], float]) -> np.ndarray:
        """
        Compute time derivatives non-intrusively using the original solver.
        
        This is the key step in operator inference: we reconstruct the full state,
        evaluate the original system's RHS, then project back to reduced coordinates.
        
        Args:
            reduced_states: Reduced states ŵ (r, n_snapshots)
            parameters: Parameter matrix [alpha, beta, t] (3, n_snapshots)
            g_func: Input function g(t)
            
        Returns:
            Reduced derivatives dŵ/dt (r, n_snapshots)
        """
        r, n_snapshots = reduced_states.shape
        reduced_derivatives = np.zeros((r, n_snapshots))
        
        print(f"Computing derivatives for {n_snapshots} snapshots...")
        
        for i in range(n_snapshots):
            # Extract parameters for this snapshot
            alpha = parameters[0, i]
            beta = parameters[1, i]
            t = parameters[2, i]
            
            # Step 1: Reconstruct full lifted state
            w_full = self.pod.reconstruct(reduced_states[:, i])
            
            # Step 2: Convert to original coordinates
            s_full = self.lifting.unlift(w_full)
            
            # Step 3: Evaluate original system RHS
            ds_dt = self.solver.system_rhs(t, s_full, alpha, beta, g_func)
            
            # Step 4: Convert derivative to lifted coordinates using chain rule
            J = self.lifting.jacobian_lift(s_full)  # dw/ds
            dw_dt = J @ ds_dt
            
            # Step 5: Project to reduced coordinates
            # Center the derivative (subtract mean derivative if computed)
            dw_dt_centered = dw_dt - self.pod.mean_state.flatten() * 0  # No time-derivative of mean
            reduced_derivatives[:, i] = self.pod.basis.T @ dw_dt_centered
            
            if (i + 1) % 100 == 0:
                print(f"  Processed {i+1}/{n_snapshots} snapshots")
        
        return reduced_derivatives
    
    def _build_data_matrix(self, reduced_states: np.ndarray,
                          parameters: np.ndarray) -> np.ndarray:
        """
        Build the data matrix D for operator inference.
        
        For quadratic system: dŵ/dt = Âŵ + Ĥ(ŵ⊗ŵ) + B̂u
        
        D = [ŵ₁ ŵ₁⊗ŵ₁ u₁]
            [ŵ₂ ŵ₂⊗ŵ₂ u₂]
            [⋮   ⋮    ⋮ ]
            
        Args:
            reduced_states: Reduced states ŵ (r, n_snapshots)
            parameters: Parameter matrix [alpha, beta, t] (3, n_snapshots)
            
        Returns:
            Data matrix D (n_snapshots, n_terms)
        """
        r, n_snapshots = reduced_states.shape
        
        # Linear terms: ŵ
        linear_terms = reduced_states.T  # (n_snapshots, r)
        
        # Quadratic terms: ŵ⊗ŵ (unique elements only)
        # For symmetric Kronecker product, we only need upper triangular elements
        quad_indices = []
        for i in range(r):
            for j in range(i, r):
                quad_indices.append((i, j))
        
        n_quad = len(quad_indices)
        quadratic_terms = np.zeros((n_snapshots, n_quad))
        
        for k, (i, j) in enumerate(quad_indices):
            if i == j:
                quadratic_terms[:, k] = reduced_states[i, :] * reduced_states[j, :]
            else:
                # Off-diagonal terms: w_i * w_j + w_j * w_i = 2 * w_i * w_j
                quadratic_terms[:, k] = 2 * reduced_states[i, :] * reduced_states[j, :]
        
        # Input terms: u = [α*g(t), β] (could be extended)
        alpha_vals = parameters[0, :]
        beta_vals = parameters[1, :]
        t_vals = parameters[2, :]
        
        # For simplicity, let's use α as the input parameter
        # In practice, you might want to include g(t) directly
        input_terms = alpha_vals.reshape(-1, 1)  # (n_snapshots, 1)
        
        # Combine all terms
        D = np.hstack([linear_terms, quadratic_terms, input_terms])
        
        return D, quad_indices
    
    def learn_operators(self, reduced_states: np.ndarray,
                       parameters: np.ndarray,
                       g_func: Callable[[float], float],
                       regularization: Optional[Dict] = None) -> Dict[str, np.ndarray]:
        """
        Learn the reduced operators using operator inference.
        
        Args:
            reduced_states: Reduced states ŵ (r, n_snapshots)
            parameters: Parameter matrix [alpha, beta, t] (3, n_snapshots)
            g_func: Input function g(t)
            regularization: Regularization parameters
            
        Returns:
            Dictionary of learned operators
        """
        r, n_snapshots = reduced_states.shape
        
        print(f"Learning operators for {r}-dimensional system...")
        
        # Step 1: Compute derivatives non-intrusively
        reduced_derivatives = self._compute_derivatives_nonintrusive(
            reduced_states, parameters, g_func
        )
        
        # Step 2: Build data matrix
        D, quad_indices = self._build_data_matrix(reduced_states, parameters)
        n_terms = D.shape[1]
        
        print(f"Data matrix shape: {D.shape}")
        print(f"Number of terms: {n_terms} (linear: {r}, quadratic: {len(quad_indices)}, input: 1)")
        
        # Step 3: Solve for operators using least squares
        # Each row of the system: dŵᵢ/dt = Dᵢ @ [Âᵢ, Ĥᵢ, B̂ᵢ]ᵀ
        
        operators = {}
        
        # Store indices for reconstruction
        operators['linear_size'] = r
        operators['quadratic_size'] = len(quad_indices)
        operators['input_size'] = 1
        operators['quad_indices'] = quad_indices
        
        # Learn operators for each reduced state component
        A_hat = np.zeros((r, r))
        H_hat = np.zeros((r, len(quad_indices)))
        B_hat = np.zeros((r, 1))
        
        for i in range(r):
            target = reduced_derivatives[i, :]  # dŵᵢ/dt
            
            if regularization is None:
                # Standard least squares
                coeffs, residual, rank, s = np.linalg.lstsq(D, target, rcond=None)
            else:
                # Regularized least squares
                reg_type = regularization.get('type', 'ridge')
                reg_param = regularization.get('lambda', 1e-3)
                
                if reg_type == 'ridge':
                    reg_model = Ridge(alpha=reg_param, fit_intercept=False)
                elif reg_type == 'lasso':
                    reg_model = Lasso(alpha=reg_param, fit_intercept=False, max_iter=2000)
                else:
                    raise ValueError(f"Unknown regularization type: {reg_type}")
                
                reg_model.fit(D, target)
                coeffs = reg_model.coef_
                residual = [np.linalg.norm(D @ coeffs - target)**2]
            
            # Extract operator components
            A_hat[i, :] = coeffs[:r]
            H_hat[i, :] = coeffs[r:r+len(quad_indices)]
            B_hat[i, :] = coeffs[r+len(quad_indices):]
            
            # Print residual for debugging
            if i == 0:
                print(f"  Residual for component {i}: {residual[0]:.2e}")
        
        operators['A'] = A_hat
        operators['H'] = H_hat
        operators['B'] = B_hat
        
        self.operators = operators
        self.regularization = regularization
        
        return operators
    
    def predict(self, t_span: Tuple[float, float], t_eval: np.ndarray,
                initial_condition: np.ndarray, alpha: float, beta: float,
                g_func: Callable[[float], float]) -> Dict[str, np.ndarray]:
        """
        Predict using the learned ROM.
        
        Args:
            t_span: Time span for prediction
            t_eval: Time evaluation points
            initial_condition: Initial reduced state ŵ₀
            alpha: Parameter α
            beta: Parameter β
            g_func: Input function g(t)
            
        Returns:
            Dictionary with prediction results
        """
        if not self.operators:
            raise ValueError("Operators not learned. Call learn_operators first.")
        
        from scipy.integrate import solve_ivp
        
        A = self.operators['A']
        H = self.operators['H']
        B = self.operators['B']
        quad_indices = self.operators['quad_indices']
        
        def rom_rhs(t, w_hat):
            """ROM right-hand side function."""
            # Linear term
            linear = A @ w_hat
            
            # Quadratic term
            quadratic = np.zeros_like(w_hat)
            for i, (j, k) in enumerate(quad_indices):
                coeff = w_hat[j] * w_hat[k]
                quadratic += H[:, i] * coeff
            
            # Input term
            u = alpha  # Simplified input
            input_term = B[:, 0] * u
            
            return linear + quadratic + input_term
        
        # Solve ROM
        sol = solve_ivp(
            rom_rhs, t_span, initial_condition, t_eval=t_eval,
            method='RK45', rtol=1e-8, atol=1e-10
        )
        
        return {
            't': sol.t,
            'w_hat': sol.y,
            'success': sol.success,
            'message': sol.message
        }
    
    def compute_prediction_error(self, prediction: Dict[str, np.ndarray],
                                truth: Dict[str, np.ndarray]) -> float:
        """
        Compute relative prediction error.
        
        Args:
            prediction: ROM prediction
            truth: High-fidelity truth
            
        Returns:
            Relative error
        """
        # Reconstruct full states
        w_pred_full = self.pod.reconstruct(prediction['w_hat'])
        
        # Convert truth to lifted coordinates
        nt = truth['s1'].shape[0]
        w_truth_full = np.zeros((self.lifting.n_lifted, nt))
        for i in range(nt):
            s_truth = np.concatenate([truth['s1'][i, :], truth['s2'][i, :]])
            w_truth_full[:, i] = self.lifting.lift(s_truth)
        
        # Compute error
        error_norm = np.linalg.norm(w_pred_full - w_truth_full, 'fro')
        truth_norm = np.linalg.norm(w_truth_full, 'fro')
        
        return error_norm / truth_norm


if __name__ == "__main__":
    # Test operator inference with synthetic data
    print("Testing operator inference...")
    
    # This would normally use real data from the lifting transformation
    # For now, we'll create a simple test case
    
    # Create synthetic reduced states
    np.random.seed(42)
    r = 5
    n_snapshots = 100
    reduced_states = np.random.randn(r, n_snapshots)
    
    # Create synthetic parameters
    alpha_vals = np.random.uniform(0.8, 1.2, n_snapshots)
    beta_vals = np.random.uniform(0.8, 1.2, n_snapshots)
    t_vals = np.linspace(0, 4, n_snapshots)
    parameters = np.vstack([alpha_vals, beta_vals, t_vals])
    
    print(f"Test data: {r}-dimensional system, {n_snapshots} snapshots")
    print(f"Parameter ranges: α ∈ [{alpha_vals.min():.2f}, {alpha_vals.max():.2f}]")
    print(f"                  β ∈ [{beta_vals.min():.2f}, {beta_vals.max():.2f}]")
    
    # Note: Full test would require integrated solver, lifting, and POD
    print("Note: Full operator inference test requires integrated pipeline")
    print("This will be tested in the main execution script.")