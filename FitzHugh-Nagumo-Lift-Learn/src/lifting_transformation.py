"""
Phase 2: Data Generation and Lifting Transformation

This module implements the lifting transformation T that converts the 
nonlinear FitzHugh-Nagumo system into a quadratic form:

Original: [s1, s2]
Lifted: [w1, w2, w3] = [s1, s2, s1²]

The lifted system becomes quadratic in the w coordinates.
"""

import numpy as np
from typing import Tuple, List, Dict, Callable
from .fitzhugh_nagumo_solver import FitzHughNagumoSolver, default_input_function


class LiftingTransformation:
    """Implements the lifting transformation for FitzHugh-Nagumo system."""
    
    def __init__(self, nx: int):
        """
        Initialize lifting transformation.
        
        Args:
            nx: Number of spatial grid points
        """
        self.nx = nx
        self.n_original = 2 * nx  # [s1, s2]
        self.n_lifted = 3 * nx    # [w1, w2, w3] = [s1, s2, s1²]
    
    def lift(self, state: np.ndarray) -> np.ndarray:
        """
        Apply lifting transformation T: [s1, s2] → [w1, w2, w3].
        
        Args:
            state: Original state [s1, s2] (shape: 2*nx,)
            
        Returns:
            Lifted state [w1, w2, w3] (shape: 3*nx,)
        """
        s1 = state[:self.nx]
        s2 = state[self.nx:2*self.nx]
        
        w1 = s1          # w1 = s1
        w2 = s2          # w2 = s2  
        w3 = s1**2       # w3 = s1²
        
        return np.concatenate([w1, w2, w3])
    
    def unlift(self, lifted_state: np.ndarray) -> np.ndarray:
        """
        Apply inverse lifting transformation T⁻¹: [w1, w2, w3] → [s1, s2].
        
        Args:
            lifted_state: Lifted state [w1, w2, w3] (shape: 3*nx,)
            
        Returns:
            Original state [s1, s2] (shape: 2*nx,)
        """
        w1 = lifted_state[:self.nx]
        w2 = lifted_state[self.nx:2*self.nx]
        # w3 is discarded since s1 = w1
        
        s1 = w1
        s2 = w2
        
        return np.concatenate([s1, s2])
    
    def jacobian_lift(self, state: np.ndarray) -> np.ndarray:
        """
        Compute Jacobian of lifting transformation dw/ds.
        
        Args:
            state: Original state [s1, s2] (shape: 2*nx,)
            
        Returns:
            Jacobian matrix (shape: 3*nx, 2*nx)
        """
        s1 = state[:self.nx]
        
        # Initialize Jacobian
        J = np.zeros((self.n_lifted, self.n_original))
        
        # dw1/ds1 = 1, dw1/ds2 = 0
        J[:self.nx, :self.nx] = np.eye(self.nx)
        
        # dw2/ds1 = 0, dw2/ds2 = 1  
        J[self.nx:2*self.nx, self.nx:2*self.nx] = np.eye(self.nx)
        
        # dw3/ds1 = 2*s1, dw3/ds2 = 0
        J[2*self.nx:3*self.nx, :self.nx] = 2 * np.diag(s1)
        
        return J
    
    def lifted_system_rhs(self, t: float, w: np.ndarray, 
                         alpha: float, beta: float, gamma: float,
                         g_func: Callable[[float], float],
                         D2: np.ndarray) -> np.ndarray:
        """
        Right-hand side of the lifted quadratic system.
        
        Lifted system (from paper Eq. 60):
        ∂w₁/∂t = γ(∂²w₁/∂x²) - w₁w₃ + 1.1w₃ - 0.1w₁ + w₂ + 0.05
        ∂w₂/∂t = 0.5w₁ - 2βw₂ + 0.05  
        ∂w₃/∂t = 2(γ(∂²w₁/∂x²)w₁ - w₁²w₃ + 1.1w₁w₃ - 0.1w₁² + w₁w₂ + 0.05w₁)
        
        Args:
            t: Time
            w: Lifted state [w1, w2, w3]
            alpha: Parameter α
            beta: Parameter β
            gamma: Diffusion coefficient γ
            g_func: Boundary input function
            D2: Second derivative matrix
            
        Returns:
            Time derivative dw/dt
        """
        w1 = w[:self.nx]
        w2 = w[self.nx:2*self.nx]
        w3 = w[2*self.nx:3*self.nx]
        
        # Apply boundary condition
        w1_bc = w1.copy()
        w1_bc[0] = alpha * g_func(t)
        
        # Compute spatial derivatives
        d2w1_dx2 = D2 @ w1_bc
        
        # Lifted system equations
        dw1_dt = gamma * d2w1_dx2 - w1 * w3 + 1.1 * w3 - 0.1 * w1 + w2 + 0.05
        dw2_dt = 0.5 * w1 - 2 * beta * w2 + 0.05
        dw3_dt = 2 * (gamma * d2w1_dx2 * w1 - w1**2 * w3 + 1.1 * w1 * w3 - 
                      0.1 * w1**2 + w1 * w2 + 0.05 * w1)
        
        return np.concatenate([dw1_dt, dw2_dt, dw3_dt])


class DataGenerator:
    """Generates training data for the Lift & Learn method."""
    
    def __init__(self, solver: FitzHughNagumoSolver, 
                 lifting: LiftingTransformation):
        """
        Initialize data generator.
        
        Args:
            solver: High-fidelity FitzHugh-Nagumo solver
            lifting: Lifting transformation
        """
        self.solver = solver
        self.lifting = lifting
    
    def generate_training_data(self, 
                             parameter_sets: List[Tuple[float, float]],
                             t_span: Tuple[float, float],
                             t_eval: np.ndarray,
                             g_func: Callable[[float], float] = default_input_function
                             ) -> Dict[str, np.ndarray]:
        """
        Generate training snapshots for multiple parameter sets.
        
        Args:
            parameter_sets: List of (alpha, beta) parameter pairs
            t_span: Time span for simulation
            t_eval: Time evaluation points
            g_func: Input function g(t)
            
        Returns:
            Dictionary with snapshot matrices
        """
        all_snapshots_original = []
        all_snapshots_lifted = []
        all_parameters = []
        
        for alpha, beta in parameter_sets:
            print(f"Generating data for α={alpha:.2f}, β={beta:.2f}")
            
            # Solve high-fidelity system
            solution = self.solver.solve(
                t_span=t_span,
                t_eval=t_eval,
                alpha=alpha,
                beta=beta,
                g_func=g_func
            )
            
            if not solution['success']:
                raise RuntimeError(f"Solver failed: {solution['message']}")
            
            # Extract snapshots
            nt = len(t_eval)
            for i in range(nt):
                # Original state snapshot
                s1_snap = solution['s1'][i, :]
                s2_snap = solution['s2'][i, :]
                original_snapshot = np.concatenate([s1_snap, s2_snap])
                
                # Lifted state snapshot
                lifted_snapshot = self.lifting.lift(original_snapshot)
                
                all_snapshots_original.append(original_snapshot)
                all_snapshots_lifted.append(lifted_snapshot)
                all_parameters.append([alpha, beta, t_eval[i]])
        
        # Convert to matrices (snapshots as columns)
        S = np.array(all_snapshots_original).T  # Shape: (2*nx, n_snapshots)
        W = np.array(all_snapshots_lifted).T    # Shape: (3*nx, n_snapshots)
        P = np.array(all_parameters).T          # Shape: (3, n_snapshots)
        
        return {
            'S': S,  # Original snapshots
            'W': W,  # Lifted snapshots  
            'P': P,  # Parameters [alpha, beta, t]
            't_eval': t_eval,
            'parameter_sets': parameter_sets
        }


if __name__ == "__main__":
    # Test lifting transformation
    nx = 50
    lifting = LiftingTransformation(nx)
    
    # Test state
    np.random.seed(42)
    s1 = np.random.randn(nx)
    s2 = np.random.randn(nx)
    original_state = np.concatenate([s1, s2])
    
    # Test lifting and unlifting
    lifted_state = lifting.lift(original_state)
    recovered_state = lifting.unlift(lifted_state)
    
    print(f"Original state shape: {original_state.shape}")
    print(f"Lifted state shape: {lifted_state.shape}")
    print(f"Recovery error: {np.linalg.norm(original_state - recovered_state):.2e}")
    
    # Test Jacobian
    J = lifting.jacobian_lift(original_state)
    print(f"Jacobian shape: {J.shape}")
    
    # Test data generation
    solver = FitzHughNagumoSolver(nx=nx)
    data_gen = DataGenerator(solver, lifting)
    
    # Small test with 2 parameter sets
    param_sets = [(1.0, 1.0), (1.2, 0.8)]
    t_eval = np.linspace(0, 1, 11)  # Short test
    
    data = data_gen.generate_training_data(
        parameter_sets=param_sets,
        t_span=(0, 1),
        t_eval=t_eval
    )
    
    print(f"Generated data shapes:")
    print(f"  S (original): {data['S'].shape}")
    print(f"  W (lifted): {data['W'].shape}")
    print(f"  P (parameters): {data['P'].shape}")