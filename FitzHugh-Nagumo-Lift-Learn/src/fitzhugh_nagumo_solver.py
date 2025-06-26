"""
Phase 1: High-Fidelity FitzHugh-Nagumo Solver

This module implements the full-order solver for the FitzHugh-Nagumo system:
- ∂s₁/∂t = γ(∂²s₁/∂x²) - s₁³ + 1.1s₁² - 0.1s₁ + s₂ + 0.05
- ∂s₂/∂t = 0.5s₁ - 2s₂ + 0.05

Using method of lines: spatial discretization + ODE solver
"""

import numpy as np
from scipy.integrate import solve_ivp
from typing import Tuple, Callable, Optional
import matplotlib.pyplot as plt


class FitzHughNagumoSolver:
    """High-fidelity solver for the FitzHugh-Nagumo system."""
    
    def __init__(self, nx: int = 100, L: float = 1.0, gamma: float = 0.01):
        """
        Initialize the FitzHugh-Nagumo solver.
        
        Args:
            nx: Number of spatial grid points
            L: Domain length [0, L]
            gamma: Diffusion coefficient
        """
        self.nx = nx
        self.L = L
        self.gamma = gamma
        
        # Spatial discretization
        self.x = np.linspace(0, L, nx)
        self.dx = L / (nx - 1)
        
        # Second derivative matrix (central differences with boundary conditions)
        self.D2 = self._build_second_derivative_matrix()
    
    def _build_second_derivative_matrix(self) -> np.ndarray:
        """Build second derivative matrix using central differences."""
        D2 = np.zeros((self.nx, self.nx))
        
        # Interior points: central differences
        for i in range(1, self.nx - 1):
            D2[i, i-1] = 1.0
            D2[i, i] = -2.0
            D2[i, i+1] = 1.0
        
        # Boundary conditions: Neumann (zero flux)
        # Left boundary: ∂s₁/∂x = 0
        D2[0, 0] = -2.0
        D2[0, 1] = 2.0
        
        # Right boundary: ∂s₁/∂x = 0  
        D2[-1, -2] = 2.0
        D2[-1, -1] = -2.0
        
        return D2 / (self.dx**2)
    
    def system_rhs(self, t: float, y: np.ndarray, alpha: float, beta: float, 
                   g_func: Callable[[float], float]) -> np.ndarray:
        """
        Right-hand side of the FitzHugh-Nagumo system.
        
        Args:
            t: Time
            y: State vector [s1, s2] flattened
            alpha: Parameter α  
            beta: Parameter β
            g_func: Boundary input function g(t)
            
        Returns:
            Time derivative dy/dt
        """
        # Reshape state vector
        s1 = y[:self.nx]
        s2 = y[self.nx:]
        
        # Apply boundary condition g(t) at x=0
        s1_bc = s1.copy()
        s1_bc[0] = alpha * g_func(t)
        
        # Compute derivatives
        ds1_dt = self.gamma * self.D2 @ s1_bc - s1**3 + 1.1*s1**2 - 0.1*s1 + s2 + 0.05
        ds2_dt = 0.5*s1 - 2*beta*s2 + 0.05
        
        return np.concatenate([ds1_dt, ds2_dt])
    
    def solve(self, t_span: Tuple[float, float], t_eval: np.ndarray,
              alpha: float, beta: float, g_func: Callable[[float], float],
              initial_condition: Optional[np.ndarray] = None) -> dict:
        """
        Solve the FitzHugh-Nagumo system.
        
        Args:
            t_span: Time span (t_start, t_end)
            t_eval: Time points for evaluation
            alpha: Parameter α
            beta: Parameter β  
            g_func: Boundary input function g(t)
            initial_condition: Initial state [s1_0, s2_0]
            
        Returns:
            Dictionary with solution data
        """
        # Default initial condition: small random perturbation
        if initial_condition is None:
            np.random.seed(42)  # For reproducibility
            s1_0 = 0.1 * np.random.randn(self.nx)
            s2_0 = 0.1 * np.random.randn(self.nx)
            y0 = np.concatenate([s1_0, s2_0])
        else:
            y0 = initial_condition.flatten()
        
        # Solve ODE system
        sol = solve_ivp(
            fun=lambda t, y: self.system_rhs(t, y, alpha, beta, g_func),
            t_span=t_span,
            y0=y0,
            t_eval=t_eval,
            method='RK45',
            rtol=1e-8,
            atol=1e-10
        )
        
        # Reshape solution
        nt = len(t_eval)
        s1_sol = sol.y[:self.nx, :].T  # Shape: (nt, nx)
        s2_sol = sol.y[self.nx:, :].T  # Shape: (nt, nx)
        
        return {
            't': sol.t,
            's1': s1_sol,
            's2': s2_sol,
            'x': self.x,
            'success': sol.success,
            'message': sol.message
        }


def default_input_function(t: float) -> float:
    """
    Default boundary input function g(t) from the paper.
    
    Args:
        t: Time
        
    Returns:
        Input value g(t)
    """
    return np.sin(2 * np.pi * t) + 0.5 * np.cos(4 * np.pi * t)


if __name__ == "__main__":
    # Test the solver
    solver = FitzHughNagumoSolver(nx=50)
    
    # Time discretization
    t_end = 4.0
    dt = 0.01
    t_eval = np.arange(0, t_end + dt, dt)
    
    # Parameters (example from paper)
    alpha = 1.0
    beta = 1.0
    
    # Solve
    solution = solver.solve(
        t_span=(0, t_end),
        t_eval=t_eval,
        alpha=alpha,
        beta=beta,
        g_func=default_input_function
    )
    
    print(f"Solution computed successfully: {solution['success']}")
    print(f"Final time: {solution['t'][-1]:.2f}")
    print(f"State shapes: s1={solution['s1'].shape}, s2={solution['s2'].shape}")