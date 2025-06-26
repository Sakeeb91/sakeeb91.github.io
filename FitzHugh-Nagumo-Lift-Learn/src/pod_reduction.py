"""
Phase 3: Proper Orthogonal Decomposition (POD) for Dimensionality Reduction

This module implements POD to find a low-dimensional basis for the lifted data.
The POD basis is computed via SVD of the lifted snapshot matrix.
"""

import numpy as np
from typing import Tuple, Dict, Optional
import matplotlib.pyplot as plt


class PODReducer:
    """Implements Proper Orthogonal Decomposition for dimensionality reduction."""
    
    def __init__(self):
        """Initialize POD reducer."""
        self.basis = None
        self.singular_values = None
        self.mean_state = None
        self.n_full = None
        self.r = None
    
    def compute_pod_basis(self, snapshots: np.ndarray, 
                         r: Optional[int] = None,
                         energy_threshold: float = 0.99) -> Dict[str, np.ndarray]:
        """
        Compute POD basis from snapshot matrix using SVD.
        
        Args:
            snapshots: Snapshot matrix (n_states, n_snapshots)
            r: Number of POD modes to retain (if None, use energy threshold)
            energy_threshold: Energy threshold for automatic mode selection
            
        Returns:
            Dictionary with POD data
        """
        self.n_full = snapshots.shape[0]
        n_snapshots = snapshots.shape[1]
        
        # Compute mean and center snapshots
        self.mean_state = np.mean(snapshots, axis=1, keepdims=True)
        centered_snapshots = snapshots - self.mean_state
        
        print(f"Computing POD for {self.n_full} states, {n_snapshots} snapshots")
        
        # Compute SVD
        if n_snapshots < self.n_full:
            # Economy SVD: compute V first, then U
            _, S, Vt = np.linalg.svd(centered_snapshots.T, full_matrices=False)
        else:
            # Standard SVD
            U, S, Vt = np.linalg.svd(centered_snapshots, full_matrices=False)
            
        # Singular values and cumulative energy
        self.singular_values = S
        energy = S**2
        cumulative_energy = np.cumsum(energy) / np.sum(energy)
        
        # Determine number of modes
        if r is None:
            r = np.argmax(cumulative_energy >= energy_threshold) + 1
            print(f"Selected {r} modes for {energy_threshold*100:.1f}% energy")
        else:
            print(f"Using {r} modes ({cumulative_energy[r-1]*100:.1f}% energy)")
        
        self.r = r
        
        # POD basis (first r left singular vectors)
        if n_snapshots < self.n_full:
            # Reconstruct U from economy SVD
            self.basis = centered_snapshots @ Vt[:r, :].T / S[:r]
        else:
            self.basis = U[:, :r]
        
        return {
            'basis': self.basis,
            'singular_values': self.singular_values,
            'cumulative_energy': cumulative_energy,
            'mean_state': self.mean_state,
            'r': r
        }
    
    def project(self, state: np.ndarray) -> np.ndarray:
        """
        Project full state onto POD subspace.
        
        Args:
            state: Full state vector (n_full,) or (n_full, n_times)
            
        Returns:
            Reduced state vector (r,) or (r, n_times)
        """
        if self.basis is None:
            raise ValueError("POD basis not computed. Call compute_pod_basis first.")
        
        if state.ndim == 1:
            # Single state vector
            centered_state = state - self.mean_state.flatten()
            return self.basis.T @ centered_state
        else:
            # Multiple state vectors
            centered_states = state - self.mean_state
            return self.basis.T @ centered_states
    
    def reconstruct(self, reduced_state: np.ndarray) -> np.ndarray:
        """
        Reconstruct full state from reduced coordinates.
        
        Args:
            reduced_state: Reduced state vector (r,) or (r, n_times)
            
        Returns:
            Reconstructed full state (n_full,) or (n_full, n_times)
        """
        if self.basis is None:
            raise ValueError("POD basis not computed. Call compute_pod_basis first.")
        
        if reduced_state.ndim == 1:
            # Single state vector
            return self.basis @ reduced_state + self.mean_state.flatten()
        else:
            # Multiple state vectors
            return self.basis @ reduced_state + self.mean_state
    
    def compute_projection_error(self, test_snapshots: np.ndarray) -> float:
        """
        Compute projection error for test snapshots.
        
        Args:
            test_snapshots: Test snapshot matrix (n_full, n_test)
            
        Returns:
            Relative projection error
        """
        # Project and reconstruct
        reduced_states = self.project(test_snapshots)
        reconstructed_states = self.reconstruct(reduced_states)
        
        # Compute relative error
        error_norm = np.linalg.norm(test_snapshots - reconstructed_states, 'fro')
        data_norm = np.linalg.norm(test_snapshots, 'fro')
        
        return error_norm / data_norm
    
    def plot_energy_spectrum(self, max_modes: int = 50) -> None:
        """
        Plot singular values and cumulative energy.
        
        Args:
            max_modes: Maximum number of modes to plot
        """
        if self.singular_values is None:
            raise ValueError("POD not computed. Call compute_pod_basis first.")
        
        n_plot = min(max_modes, len(self.singular_values))
        modes = np.arange(1, n_plot + 1)
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        # Singular values
        ax1.semilogy(modes, self.singular_values[:n_plot], 'bo-')
        ax1.set_xlabel('Mode Number')
        ax1.set_ylabel('Singular Value')
        ax1.set_title('POD Singular Values')
        ax1.grid(True)
        
        # Cumulative energy
        energy = self.singular_values**2
        cumulative_energy = np.cumsum(energy) / np.sum(energy)
        ax2.plot(modes, cumulative_energy[:n_plot] * 100, 'ro-')
        ax2.axhline(y=99, color='k', linestyle='--', alpha=0.7, label='99% threshold')
        ax2.axhline(y=99.9, color='k', linestyle=':', alpha=0.7, label='99.9% threshold')
        ax2.set_xlabel('Mode Number')
        ax2.set_ylabel('Cumulative Energy (%)')
        ax2.set_title('POD Energy Content')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        
        # Auto-save for GitHub README
        auto_save_path = "images/plots/pod_energy_spectrum.png"
        plt.savefig(auto_save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {auto_save_path}")
        
        plt.show()
    
    def plot_pod_modes(self, x: np.ndarray, n_modes: int = 6) -> None:
        """
        Plot first few POD modes.
        
        Args:
            x: Spatial coordinates
            n_modes: Number of modes to plot
        """
        if self.basis is None:
            raise ValueError("POD basis not computed. Call compute_pod_basis first.")
        
        nx = len(x)
        n_vars = self.n_full // nx  # Number of variables (should be 3 for lifted system)
        n_plot = min(n_modes, self.r)
        
        fig, axes = plt.subplots(n_vars, n_plot, figsize=(3*n_plot, 2*n_vars))
        if n_vars == 1:
            axes = axes.reshape(1, -1)
        if n_plot == 1:
            axes = axes.reshape(-1, 1)
        
        var_names = ['w₁', 'w₂', 'w₃'] if n_vars == 3 else [f'var_{i+1}' for i in range(n_vars)]
        
        for i in range(n_plot):
            mode = self.basis[:, i]
            for j in range(n_vars):
                var_mode = mode[j*nx:(j+1)*nx]
                axes[j, i].plot(x, var_mode)
                axes[j, i].set_title(f'Mode {i+1}, {var_names[j]}')
                axes[j, i].grid(True)
                if i == 0:
                    axes[j, i].set_ylabel(var_names[j])
                if j == n_vars - 1:
                    axes[j, i].set_xlabel('x')
        
        plt.tight_layout()
        
        # Auto-save for GitHub README
        auto_save_path = "images/plots/pod_modes.png"
        plt.savefig(auto_save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {auto_save_path}")
        
        plt.show()


if __name__ == "__main__":
    # Test POD reducer with synthetic data
    np.random.seed(42)
    
    # Create synthetic snapshot matrix
    nx = 50
    n_vars = 3  # Lifted system has 3 variables
    n_full = n_vars * nx
    n_snapshots = 200
    
    # Generate synthetic snapshots with known low-rank structure
    r_true = 5  # True reduced dimension
    U_true = np.random.randn(n_full, r_true)
    U_true, _ = np.linalg.qr(U_true)  # Orthogonalize
    
    # Random coefficients
    coeffs = np.random.randn(r_true, n_snapshots)
    snapshots = U_true @ coeffs + 0.1 * np.random.randn(n_full, n_snapshots)
    
    print(f"Test data: {n_full} states, {n_snapshots} snapshots")
    print(f"True rank: {r_true}")
    
    # Test POD
    pod = PODReducer()
    pod_data = pod.compute_pod_basis(snapshots, r=10)
    
    print(f"POD computed with {pod_data['r']} modes")
    print(f"Energy content: {pod_data['cumulative_energy'][pod_data['r']-1]*100:.2f}%")
    
    # Test projection and reconstruction
    test_state = snapshots[:, 0]
    reduced_state = pod.project(test_state)
    reconstructed_state = pod.reconstruct(reduced_state)
    
    reconstruction_error = np.linalg.norm(test_state - reconstructed_state) / np.linalg.norm(test_state)
    print(f"Single state reconstruction error: {reconstruction_error:.2e}")
    
    # Test batch projection
    test_snapshots = snapshots[:, :10]
    projection_error = pod.compute_projection_error(test_snapshots)
    print(f"Batch projection error: {projection_error:.2e}")
    
    # Plot results
    x = np.linspace(0, 1, nx)
    pod.plot_energy_spectrum(max_modes=20)
    pod.plot_pod_modes(x, n_modes=4)