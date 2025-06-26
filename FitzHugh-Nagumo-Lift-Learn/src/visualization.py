"""
Visualization utilities for the FitzHugh-Nagumo Lift & Learn project.

This module provides comprehensive plotting and visualization functions
for analyzing and presenting results from the Lift & Learn methodology.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
from typing import Dict, List, Tuple, Optional, Callable
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


class FitzHughNagumoVisualizer:
    """Comprehensive visualization suite for FitzHugh-Nagumo results."""
    
    def __init__(self, x: np.ndarray, figsize: Tuple[int, int] = (12, 8)):
        """
        Initialize visualizer.
        
        Args:
            x: Spatial coordinates
            figsize: Default figure size
        """
        self.x = x
        self.figsize = figsize
        self.nx = len(x)
        
        # Set up plotting style
        plt.style.use('default')
        plt.rcParams['figure.figsize'] = figsize
        plt.rcParams['font.size'] = 12
        plt.rcParams['axes.grid'] = True
        plt.rcParams['grid.alpha'] = 0.3
    
    def plot_solution_evolution(self, solution: Dict[str, np.ndarray],
                               title: str = "FitzHugh-Nagumo Solution Evolution",
                               save_path: Optional[str] = None):
        """
        Plot the evolution of s1 and s2 over time and space.
        
        This creates a comprehensive 2x2 subplot layout showing:
        
        TOP ROW - Space-Time Contour Plots:
        - Left: s₁(x,t) evolution showing activator variable dynamics
          * X-axis: Time progression
          * Y-axis: Spatial domain [0,1]  
          * Colors: s₁ magnitude (blue=negative, red=positive)
          * Reveals: Wave propagation, pattern formation, excitation fronts
          
        - Right: s₂(x,t) evolution showing inhibitor variable dynamics
          * Same axes as s₁ plot for direct comparison
          * Shows slower inhibitor response to activator excitation
          * Reveals: Recovery dynamics, spatial coupling effects
        
        BOTTOM ROW - Detailed Analysis:
        - Left: Final spatial profiles at t=T
          * Shows steady-state or final transient patterns
          * Blue line: s₁ final profile (activator)
          * Red line: s₂ final profile (inhibitor)
          * Reveals: Pattern wavelength, amplitude, spatial structure
          
        - Right: Temporal evolution at domain center (x=0.5)
          * Shows typical oscillatory dynamics at one point
          * Blue line: s₁(0.5,t) - fast activator oscillations
          * Red line: s₂(0.5,t) - slower inhibitor oscillations  
          * Reveals: Period, amplitude, phase relationships
        
        INTERPRETATION GUIDE:
        - Traveling waves appear as diagonal bands in space-time plots
        - Spiral patterns show as curved contours
        - Oscillations appear as horizontal stripes
        - Excitation events show as red regions followed by blue recovery
        
        Args:
            solution: Solution dictionary with 't', 's1', 's2', 'x'
            title: Plot title
            save_path: Path to save figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        t = solution['t']
        s1 = solution['s1']
        s2 = solution['s2']
        
        # Space-time plots
        T, X = np.meshgrid(t, self.x)
        
        # s1 space-time
        im1 = axes[0, 0].contourf(T, X, s1.T, levels=20, cmap='RdBu_r')
        axes[0, 0].set_xlabel('Time')
        axes[0, 0].set_ylabel('Space (x)')
        axes[0, 0].set_title('s₁(x,t)')
        plt.colorbar(im1, ax=axes[0, 0])
        
        # s2 space-time
        im2 = axes[0, 1].contourf(T, X, s2.T, levels=20, cmap='RdBu_r')
        axes[0, 1].set_xlabel('Time')
        axes[0, 1].set_ylabel('Space (x)')
        axes[0, 1].set_title('s₂(x,t)')
        plt.colorbar(im2, ax=axes[0, 1])
        
        # Final profiles
        axes[1, 0].plot(self.x, s1[-1, :], 'b-', linewidth=2, label='s₁(x,T)')
        axes[1, 0].plot(self.x, s2[-1, :], 'r-', linewidth=2, label='s₂(x,T)')
        axes[1, 0].set_xlabel('x')
        axes[1, 0].set_ylabel('State')
        axes[1, 0].set_title(f'Final Profiles at t = {t[-1]:.2f}')
        axes[1, 0].legend()
        
        # Time evolution at center
        mid_idx = self.nx // 2
        axes[1, 1].plot(t, s1[:, mid_idx], 'b-', linewidth=2, label='s₁')
        axes[1, 1].plot(t, s2[:, mid_idx], 'r-', linewidth=2, label='s₂')
        axes[1, 1].set_xlabel('Time')
        axes[1, 1].set_ylabel('State')
        axes[1, 1].set_title(f'Evolution at x = {self.x[mid_idx]:.2f}')
        axes[1, 1].legend()
        
        plt.suptitle(title)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        # Auto-save for GitHub README
        auto_save_path = "images/plots/solution_evolution.png"
        plt.savefig(auto_save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {auto_save_path}")
        
        plt.show()
    
    def plot_phase_space(self, solution: Dict[str, np.ndarray],
                        spatial_indices: Optional[List[int]] = None,
                        title: str = "Phase Space Evolution"):
        """
        Plot phase space trajectories (s1 vs s2) at different spatial locations.
        
        This creates a phase portrait showing the relationship between activator (s₁) 
        and inhibitor (s₂) variables at multiple spatial locations:
        
        PLOT ELEMENTS:
        - X-axis: s₁ (activator variable)
        - Y-axis: s₂ (inhibitor variable)  
        - Colored curves: Trajectories for different spatial positions
        - Circles (○): Starting points of trajectories
        - Squares (□): Ending points of trajectories
        - Color gradient: Represents spatial position (blue=left, yellow=right)
        
        INTERPRETATION GUIDE:
        
        Trajectory Shapes:
        - Limit cycles: Closed loops indicate sustained oscillations
        - Spirals: Convergence to or divergence from equilibrium
        - Figure-8: Complex multi-modal oscillations
        - Open curves: Transient behavior or edge effects
        
        Spatial Differences:
        - Similar trajectories: Spatially synchronized dynamics
        - Different trajectories: Spatial heterogeneity or traveling waves
        - Phase shifts: Delayed oscillations across space
        
        FitzHugh-Nagumo Dynamics:
        - Excitable regime: Small perturbations decay, large ones trigger spikes
        - Oscillatory regime: Self-sustained limit cycle oscillations
        - Bistable regime: Two stable states with switching dynamics
        
        Typical Features:
        - Fast s₁ dynamics (horizontal movements)
        - Slow s₂ recovery (vertical movements)
        - Excitation threshold visible as trajectory kinks
        - Recovery phase shows as curved return path
        
        Args:
            solution: Solution dictionary
            spatial_indices: Spatial indices to plot (if None, use subset)
            title: Plot title
        """
        if spatial_indices is None:
            # Select a few representative spatial points
            spatial_indices = [0, self.nx//4, self.nx//2, 3*self.nx//4, self.nx-1]
        
        fig, ax = plt.subplots(figsize=self.figsize)
        
        s1 = solution['s1']
        s2 = solution['s2']
        
        colors = plt.cm.viridis(np.linspace(0, 1, len(spatial_indices)))
        
        for i, idx in enumerate(spatial_indices):
            ax.plot(s1[:, idx], s2[:, idx], color=colors[i], linewidth=2,
                   label=f'x = {self.x[idx]:.2f}')
            # Mark starting point
            ax.plot(s1[0, idx], s2[0, idx], 'o', color=colors[i], markersize=8)
            # Mark ending point
            ax.plot(s1[-1, idx], s2[-1, idx], 's', color=colors[i], markersize=8)
        
        ax.set_xlabel('s₁')
        ax.set_ylabel('s₂')
        ax.set_title(title)
        ax.legend()
        
        # Auto-save for GitHub README
        auto_save_path = "images/plots/phase_space.png"
        plt.savefig(auto_save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {auto_save_path}")
        
        plt.show()
    
    def plot_pod_analysis(self, pod_data: Dict, max_modes: int = 20):
        """
        Comprehensive POD analysis plots showing reduced-order model quality.
        
        This creates a 2x3 subplot layout for thorough POD analysis:
        
        TOP ROW - Statistical Analysis:
        
        1. Singular Values (Top-Left):
        - Y-axis: Singular values (log scale)
        - X-axis: Mode number
        - Blue circles: Individual singular values
        - Interpretation: Shows energy content of each mode
          * Steep drop: Good low-dimensional approximation possible
          * Gradual decay: Many modes needed for accuracy
          * Plateaus: Noise floor or system complexity
        
        2. Cumulative Energy (Top-Center):
        - Y-axis: Cumulative energy percentage
        - X-axis: Mode number
        - Red circles: Energy captured by first N modes
        - Horizontal lines: 99% and 99.9% thresholds
        - Interpretation: Determines required modes for accuracy
          * Rapid rise: Efficient compression
          * 99% line crossing: Practical truncation point
        
        3. Decay Rate Analysis (Top-Right):
        - Y-axis: Logarithmic decay rate between consecutive modes
        - X-axis: Mode number
        - Green circles: Rate of singular value decrease
        - Interpretation: Identifies decay patterns
          * Constant slope: Exponential decay (ideal)
          * Oscillations: Complex system dynamics
          * Sharp changes: Regime transitions
        
        BOTTOM ROW - Spatial Mode Structures:
        
        For Lifted FitzHugh-Nagumo System (3 variables):
        
        4. w₁ Modes (Bottom-Left): s₁ activator variable modes
        - X-axis: Spatial coordinate x ∈ [0,1]
        - Y-axis: Mode amplitude
        - Multiple lines: First few POD modes for w₁
        - Interpretation: Spatial patterns in activator
          * Mode 1: Usually mean profile or dominant pattern
          * Higher modes: Increasingly complex spatial variations
          * Oscillatory patterns: Wave-like structures
        
        5. w₂ Modes (Bottom-Center): s₂ inhibitor variable modes  
        - Similar to w₁ but for inhibitor variable
        - Often shows different spatial structure than activator
        - Reveals coupling between activator-inhibitor dynamics
        
        6. w₃ Modes (Bottom-Right): s₁² quadratic variable modes
        - Shows spatial patterns in the quadratic lifting variable
        - Important for capturing cubic nonlinearity effects
        - Often smoother than linear modes due to squaring
        
        INTERPRETATION GUIDE:
        
        Good POD Quality Indicators:
        - Singular values drop by 3+ orders of magnitude
        - 99% energy captured in <20 modes
        - Smooth, interpretable spatial modes
        - Clear separation between signal and noise
        
        Poor POD Quality Indicators:
        - Slow singular value decay
        - Noisy or complex spatial modes
        - No clear truncation point
        - High-frequency oscillations in modes
        
        Physical Meaning:
        - First mode: Mean behavior or dominant pattern
        - Second mode: Primary oscillation or variation
        - Higher modes: Fine-scale structures, noise, boundaries
        
        Args:
            pod_data: POD analysis results
            max_modes: Maximum modes to display
        """
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        
        singular_values = pod_data['singular_values']
        cumulative_energy = pod_data['cumulative_energy']
        basis = pod_data['basis']
        
        n_plot = min(max_modes, len(singular_values))
        modes = np.arange(1, n_plot + 1)
        
        # Singular values
        axes[0, 0].semilogy(modes, singular_values[:n_plot], 'bo-', linewidth=2, markersize=6)
        axes[0, 0].set_xlabel('Mode Number')
        axes[0, 0].set_ylabel('Singular Value')
        axes[0, 0].set_title('POD Singular Values')
        
        # Cumulative energy
        axes[0, 1].plot(modes, cumulative_energy[:n_plot] * 100, 'ro-', linewidth=2, markersize=6)
        axes[0, 1].axhline(y=99, color='k', linestyle='--', alpha=0.7, label='99%')
        axes[0, 1].axhline(y=99.9, color='k', linestyle=':', alpha=0.7, label='99.9%')
        axes[0, 1].set_xlabel('Mode Number')
        axes[0, 1].set_ylabel('Cumulative Energy (%)')
        axes[0, 1].set_title('Energy Capture')
        axes[0, 1].legend()
        
        # Energy decay rate
        if len(singular_values) > 10:
            decay_rate = np.diff(np.log(singular_values[:20]))
            axes[0, 2].plot(decay_rate, 'go-', linewidth=2, markersize=6)
            axes[0, 2].set_xlabel('Mode Number')
            axes[0, 2].set_ylabel('Log Decay Rate')
            axes[0, 2].set_title('Singular Value Decay')
        
        # First few POD modes
        n_vars = basis.shape[0] // self.nx
        n_modes_plot = min(6, basis.shape[1])
        
        if n_vars == 3:  # Lifted system
            var_names = ['w₁ (s₁)', 'w₂ (s₂)', 'w₃ (s₁²)']
            for i, var_name in enumerate(var_names):
                ax = axes[1, i]
                for mode in range(min(3, n_modes_plot)):
                    mode_data = basis[i*self.nx:(i+1)*self.nx, mode]
                    ax.plot(self.x, mode_data, linewidth=2, label=f'Mode {mode+1}')
                ax.set_xlabel('x')
                ax.set_ylabel(f'{var_name} Mode')
                ax.set_title(f'POD Modes: {var_name}')
                ax.legend()
        
        plt.tight_layout()
        
        # Auto-save for GitHub README
        auto_save_path = "images/plots/pod_analysis.png"
        plt.savefig(auto_save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {auto_save_path}")
        
        plt.show()
    
    def plot_rom_comparison(self, hf_solution: Dict, rom_solution: Dict,
                           pod: 'PODReducer', lifting: 'LiftingTransformation',
                           alpha: float, beta: float, error: float):
        """
        Detailed comparison between high-fidelity and ROM solutions.
        
        This creates a comprehensive 3x3 subplot layout for ROM validation:
        
        TOP ROW - s₁ Variable Analysis:
        
        1. High-Fidelity s₁ (Top-Left):
        - Space-time contour plot of truth solution
        - X-axis: Time, Y-axis: Space
        - Colors: s₁ magnitude (blue=low, red=high)
        - Shows: Reference solution dynamics
        
        2. ROM s₁ (Top-Center):  
        - Space-time contour plot of ROM prediction
        - Same color scale as high-fidelity for comparison
        - Shows: ROM approximation quality
        
        3. s₁ Error (Top-Right):
        - Space-time plot of ROM - High-fidelity difference
        - Reveals: Where and when ROM fails
        - Blue regions: ROM underestimates
        - Red regions: ROM overestimates
        
        MIDDLE ROW - s₂ Variable Analysis:
        
        4-6. Similar layout for s₂ inhibitor variable
        - Same interpretation as s₁ row
        - Often shows different error patterns than s₁
        - Important for validating inhibitor dynamics
        
        BOTTOM ROW - Detailed Comparisons:
        
        7. Final Profile Comparison s₁ (Bottom-Left):
        - X-axis: Spatial coordinate x
        - Y-axis: s₁ value at final time
        - Thick blue line: High-fidelity solution
        - Dashed red line: ROM prediction
        - Shows: Steady-state accuracy
        
        8. Final Profile Comparison s₂ (Bottom-Center):
        - Similar to s₁ but for inhibitor variable
        - Reveals spatial pattern accuracy
        
        9. Error Evolution (Bottom-Right):
        - X-axis: Time
        - Y-axis: Relative error (log scale)
        - Green line: ||ROM(t) - Truth(t)|| / ||Truth(t)||
        - Shows: Temporal error accumulation
        
        INTERPRETATION GUIDE:
        
        Good ROM Quality Indicators:
        - Error plots show mostly blue/white (small errors)
        - Final profiles nearly overlap
        - Error evolution stays below 1e-2
        - Similar space-time patterns in ROM vs. truth
        
        Poor ROM Quality Indicators:
        - Large red/blue regions in error plots
        - Significant profile differences
        - Growing error over time
        - Missing or distorted patterns
        
        Common Error Sources:
        - Insufficient POD modes: Smooth, large-scale errors
        - Poor operator learning: Oscillatory or chaotic errors
        - Boundary effects: Errors concentrated at domain edges
        - Nonlinearity neglect: Amplitude or frequency mismatches
        
        Physical Validation:
        - Phase relationships preserved between s₁ and s₂
        - Wave speeds and patterns maintained
        - Oscillation frequencies accurate
        - Spatial correlations captured
        
        Args:
            hf_solution: High-fidelity solution
            rom_solution: ROM solution
            pod: POD reducer
            lifting: Lifting transformation
            alpha: Parameter α
            beta: Parameter β
            error: Overall error
        """
        # Reconstruct ROM solution
        w_hat = rom_solution['w_hat']
        w_full = pod.reconstruct(w_hat)
        
        nt = len(rom_solution['t'])
        s1_rom = np.zeros((nt, self.nx))
        s2_rom = np.zeros((nt, self.nx))
        
        for i in range(nt):
            s_rom = lifting.unlift(w_full[:, i])
            s1_rom[i, :] = s_rom[:self.nx]
            s2_rom[i, :] = s_rom[self.nx:]
        
        fig, axes = plt.subplots(3, 3, figsize=(18, 12))
        
        # Space-time comparisons
        T, X = np.meshgrid(hf_solution['t'], self.x)
        
        # High-fidelity s1
        im1 = axes[0, 0].contourf(T, X, hf_solution['s1'].T, levels=20, cmap='RdBu_r')
        axes[0, 0].set_title('High-Fidelity s₁')
        axes[0, 0].set_ylabel('x')
        plt.colorbar(im1, ax=axes[0, 0])
        
        # ROM s1
        T_rom, X_rom = np.meshgrid(rom_solution['t'], self.x)
        im2 = axes[0, 1].contourf(T_rom, X_rom, s1_rom.T, levels=20, cmap='RdBu_r')
        axes[0, 1].set_title('ROM s₁')
        plt.colorbar(im2, ax=axes[0, 1])
        
        # Error s1
        s1_interp = np.interp(rom_solution['t'], hf_solution['t'], hf_solution['s1'].T).T
        error_s1 = s1_rom - s1_interp
        im3 = axes[0, 2].contourf(T_rom, X_rom, error_s1.T, levels=20, cmap='RdBu_r')
        axes[0, 2].set_title('Error s₁')
        plt.colorbar(im3, ax=axes[0, 2])
        
        # Similar for s2
        im4 = axes[1, 0].contourf(T, X, hf_solution['s2'].T, levels=20, cmap='RdBu_r')
        axes[1, 0].set_title('High-Fidelity s₂')
        axes[1, 0].set_ylabel('x')
        plt.colorbar(im4, ax=axes[1, 0])
        
        im5 = axes[1, 1].contourf(T_rom, X_rom, s2_rom.T, levels=20, cmap='RdBu_r')
        axes[1, 1].set_title('ROM s₂')
        plt.colorbar(im5, ax=axes[1, 1])
        
        s2_interp = np.interp(rom_solution['t'], hf_solution['t'], hf_solution['s2'].T).T
        error_s2 = s2_rom - s2_interp
        im6 = axes[1, 2].contourf(T_rom, X_rom, error_s2.T, levels=20, cmap='RdBu_r')
        axes[1, 2].set_title('Error s₂')
        axes[1, 2].set_xlabel('Time')
        plt.colorbar(im6, ax=axes[1, 2])
        
        # Profile comparisons at final time
        axes[2, 0].plot(self.x, hf_solution['s1'][-1, :], 'b-', linewidth=3, label='High-Fidelity')
        axes[2, 0].plot(self.x, s1_rom[-1, :], 'r--', linewidth=2, label='ROM')
        axes[2, 0].set_xlabel('x')
        axes[2, 0].set_ylabel('s₁')
        axes[2, 0].set_title(f's₁ at Final Time')
        axes[2, 0].legend()
        
        axes[2, 1].plot(self.x, hf_solution['s2'][-1, :], 'b-', linewidth=3, label='High-Fidelity')
        axes[2, 1].plot(self.x, s2_rom[-1, :], 'r--', linewidth=2, label='ROM')
        axes[2, 1].set_xlabel('x')
        axes[2, 1].set_ylabel('s₂')
        axes[2, 1].set_title(f's₂ at Final Time')
        axes[2, 1].legend()
        
        # Error evolution
        error_evolution = np.zeros(len(rom_solution['t']))
        for i in range(len(rom_solution['t'])):
            hf_interp_s1 = np.interp(rom_solution['t'][i], hf_solution['t'], hf_solution['s1'].T).T
            hf_interp_s2 = np.interp(rom_solution['t'][i], hf_solution['t'], hf_solution['s2'].T).T
            
            rom_state = np.concatenate([s1_rom[i, :], s2_rom[i, :]])
            hf_state = np.concatenate([hf_interp_s1, hf_interp_s2])
            
            error_evolution[i] = np.linalg.norm(rom_state - hf_state) / np.linalg.norm(hf_state)
        
        axes[2, 2].semilogy(rom_solution['t'], error_evolution, 'g-', linewidth=2)
        axes[2, 2].set_xlabel('Time')
        axes[2, 2].set_ylabel('Relative Error')
        axes[2, 2].set_title('Error Evolution')
        
        plt.suptitle(f'ROM Validation: α={alpha:.2f}, β={beta:.2f}, Overall Error={error:.2e}')
        plt.tight_layout()
        
        # Auto-save for GitHub README
        auto_save_path = "images/plots/rom_validation.png"
        plt.savefig(auto_save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {auto_save_path}")
        
        plt.show()
    
    def plot_parameter_study_results(self, results: Dict):
        """
        Plot results from parameter sweep studies showing ROM performance across parameter space.
        
        This creates a 2x2 subplot layout for comprehensive parameter analysis:
        
        1. Error in Parameter Space (Top-Left):
        - X-axis: α parameter values
        - Y-axis: β parameter values  
        - Colors: ROM prediction error (blue=low, yellow=high)
        - Scatter points: Individual test cases
        - Interpretation: Reveals parameter regions where ROM works well/poorly
          * Blue regions: ROM generalizes well
          * Yellow/red regions: ROM struggles, may need retraining
          * Clusters: Similar parameter combinations
          * Gradients: Smooth parameter dependencies
        
        2. Error Distribution (Top-Right):
        - X-axis: Prediction error magnitude
        - Y-axis: Frequency (log scale)
        - Histogram bars: Distribution of errors across all tests
        - Interpretation: Statistical overview of ROM performance
          * Left-skewed: Most cases have low error (good)
          * Right-skewed: Some cases have very high error (concerning)
          * Bimodal: Two distinct performance regimes
          * Long tail: Outlier cases needing investigation
        
        3. Error vs α Parameter (Bottom-Left):
        - X-axis: α parameter values
        - Y-axis: Prediction error (log scale)
        - Scatter points: Individual test results
        - Interpretation: Sensitivity to α parameter
          * Flat distribution: ROM robust to α changes
          * Trends: Systematic α dependence
          * Clusters: Discrete parameter effects
          * Outliers: Problematic α values
        
        4. Error vs β Parameter (Bottom-Right):
        - X-axis: β parameter values
        - Y-axis: Prediction error (log scale)
        - Scatter points: Individual test results
        - Interpretation: Sensitivity to β parameter
          * Similar analysis as for α parameter
          * Compare patterns between α and β dependencies
        
        INTERPRETATION GUIDE:
        
        Good ROM Generalization:
        - Errors concentrated in blue regions (< 1e-2)
        - Uniform low errors across parameter space
        - No strong parameter dependencies
        - Few outliers or problematic regions
        
        Poor ROM Generalization:
        - Large yellow/red regions in parameter space
        - Strong parameter dependencies (trends in scatter plots)
        - Bimodal error distributions
        - Many outliers at parameter extremes
        
        Physical Insights:
        - Parameter boundaries may correspond to regime changes
        - High errors near bifurcation points
        - Training data coverage affects extrapolation
        - Nonlinear parameter effects visible as curves/clusters
        
        Remedial Actions:
        - High error regions: Add training data
        - Parameter trends: Include parameter-dependent operators
        - Outliers: Investigate individual failure cases
        - Bimodality: Consider regime-specific ROMs
        
        Args:
            results: Parameter study results with 'alpha_test', 'beta_test', 'errors'
        """
        alpha_test = results['alpha_test']
        beta_test = results['beta_test']
        errors = results['errors']
        
        # Remove infinite errors
        valid_indices = np.isfinite(errors)
        alpha_valid = alpha_test[valid_indices]
        beta_valid = beta_test[valid_indices]
        errors_valid = errors[valid_indices]
        
        if len(errors_valid) == 0:
            print("No valid results to plot")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Scatter plot in parameter space
        scatter = axes[0, 0].scatter(alpha_valid, beta_valid, c=errors_valid, 
                                   cmap='viridis', s=50, alpha=0.7)
        axes[0, 0].set_xlabel('α')
        axes[0, 0].set_ylabel('β')
        axes[0, 0].set_title('Error in Parameter Space')
        plt.colorbar(scatter, ax=axes[0, 0], label='Error')
        
        # Error histogram
        axes[0, 1].hist(errors_valid, bins=20, alpha=0.7, edgecolor='black')
        axes[0, 1].set_xlabel('Error')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Error Distribution')
        axes[0, 1].set_yscale('log')
        
        # Error vs alpha
        axes[1, 0].scatter(alpha_valid, errors_valid, alpha=0.7)
        axes[1, 0].set_xlabel('α')
        axes[1, 0].set_ylabel('Error')
        axes[1, 0].set_title('Error vs α')
        axes[1, 0].set_yscale('log')
        
        # Error vs beta
        axes[1, 1].scatter(beta_valid, errors_valid, alpha=0.7)
        axes[1, 1].set_xlabel('β')
        axes[1, 1].set_ylabel('Error')
        axes[1, 1].set_title('Error vs β')
        axes[1, 1].set_yscale('log')
        
        plt.tight_layout()
        
        # Auto-save for GitHub README
        auto_save_path = "images/plots/parameter_study.png"
        plt.savefig(auto_save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {auto_save_path}")
        
        plt.show()
    
    def create_interactive_solution_plot(self, solution: Dict[str, np.ndarray],
                                       title: str = "Interactive FitzHugh-Nagumo Solution"):
        """
        Create interactive Plotly visualization of the solution with hover capabilities.
        
        This creates a 2x2 interactive dashboard with the following features:
        
        1. s₁ Evolution Heatmap (Top-Left):
        - Interactive space-time heatmap
        - Hover: Shows exact values and coordinates
        - Zoom: Pan and zoom for detailed inspection
        - Colors: Dynamic range with colorbar
        - Features: Reveals wave propagation and excitation patterns
        
        2. s₂ Evolution Heatmap (Top-Right):
        - Similar to s₁ but for inhibitor variable
        - Coordinated interaction with s₁ plot
        - Shows inhibitor response to activator dynamics
        
        3. Phase Space Trajectory (Bottom-Left):
        - Interactive s₁ vs s₂ trajectory at domain center
        - Hover: Shows time progression along trajectory
        - Features: Limit cycles, excitation events, recovery
        - Colors: May encode time or other variables
        
        4. Time Series (Bottom-Right):
        - Interactive time evolution at domain center
        - Two lines: s₁ (blue) and s₂ (red)
        - Hover: Precise values and timing
        - Zoom: Examine oscillation details
        
        INTERACTIVE FEATURES:
        
        Zoom and Pan:
        - Box zoom: Select region to zoom into
        - Pan: Click and drag to move view
        - Auto-scale: Double-click to reset view
        - Linked axes: Optional coordinate across subplots
        
        Hover Information:
        - Exact numerical values
        - Coordinate positions (x, t)
        - Variable identification
        - Additional derived quantities
        
        Toggle Features:
        - Legend clicking: Show/hide data series
        - Trace selection: Highlight specific elements
        - Color scale adjustment: Modify contrast
        
        Export Options:
        - PNG download: High-resolution static images
        - HTML save: Preserve interactivity
        - Data export: Underlying numerical values
        
        INTERPRETATION BENEFITS:
        
        Detailed Inspection:
        - Zoom into specific events or regions
        - Measure precise values and timing
        - Compare patterns across space and time
        - Identify subtle features missed in static plots
        
        Pattern Recognition:
        - Dynamic exploration of wave patterns
        - Interactive phase space analysis
        - Correlation between space-time and phase plots
        - Real-time parameter sensitivity (if implemented)
        
        Presentation Quality:
        - Professional interactive figures
        - Embedded in web pages or notebooks
        - Shareable with full functionality
        - Publication-ready visualizations
        
        Args:
            solution: Solution dictionary with 't', 's1', 's2', 'x'
            title: Plot title for the dashboard
        """
        t = solution['t']
        s1 = solution['s1']
        s2 = solution['s2']
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('s₁ Evolution', 's₂ Evolution', 'Phase Space', 'Time Series'),
            specs=[[{"type": "heatmap"}, {"type": "heatmap"}],
                   [{"type": "scatter"}, {"type": "scatter"}]]
        )
        
        # s1 heatmap
        fig.add_trace(
            go.Heatmap(z=s1.T, x=t, y=self.x, colorscale='RdBu', name='s₁'),
            row=1, col=1
        )
        
        # s2 heatmap
        fig.add_trace(
            go.Heatmap(z=s2.T, x=t, y=self.x, colorscale='RdBu', name='s₂'),
            row=1, col=2
        )
        
        # Phase space at center
        mid_idx = self.nx // 2
        fig.add_trace(
            go.Scatter(x=s1[:, mid_idx], y=s2[:, mid_idx], mode='lines+markers',
                      name=f'Phase Space (x={self.x[mid_idx]:.2f})'),
            row=2, col=1
        )
        
        # Time series at center
        fig.add_trace(
            go.Scatter(x=t, y=s1[:, mid_idx], mode='lines', name='s₁', line=dict(color='blue')),
            row=2, col=2
        )
        fig.add_trace(
            go.Scatter(x=t, y=s2[:, mid_idx], mode='lines', name='s₂', line=dict(color='red')),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            title=title,
            height=800,
            showlegend=True
        )
        
        # Update axes labels
        fig.update_xaxes(title_text="Time", row=1, col=1)
        fig.update_xaxes(title_text="Time", row=1, col=2)
        fig.update_xaxes(title_text="s₁", row=2, col=1)
        fig.update_xaxes(title_text="Time", row=2, col=2)
        
        fig.update_yaxes(title_text="Space (x)", row=1, col=1)
        fig.update_yaxes(title_text="Space (x)", row=1, col=2)
        fig.update_yaxes(title_text="s₂", row=2, col=1)
        fig.update_yaxes(title_text="State", row=2, col=2)
        
        fig.show()
    
    def create_animation(self, solution: Dict[str, np.ndarray],
                        filename: str = "fitzhugh_nagumo_animation.gif",
                        fps: int = 10):
        """
        Create animated visualization of the FitzHugh-Nagumo solution evolution.
        
        This generates a side-by-side animated visualization with two synchronized panels:
        
        LEFT PANEL - Spatial Profiles Animation:
        - X-axis: Spatial coordinate x ∈ [0,1]
        - Y-axis: Variable magnitude
        - Blue line: s₁(x,t) activator variable (evolving)
        - Red line: s₂(x,t) inhibitor variable (evolving)
        - Title: Shows current simulation time
        - Features: Wave propagation, pattern formation, excitation fronts
        
        RIGHT PANEL - Phase Space Evolution:
        - X-axis: s₁ at domain center (x=0.5)
        - Y-axis: s₂ at domain center (x=0.5)
        - Green trail: Historical trajectory (fading)
        - Red dot: Current state position
        - Features: Limit cycle formation, excitation events, recovery phases
        
        ANIMATION ELEMENTS:
        
        Temporal Evolution:
        - Each frame: One time step of the simulation
        - Smooth transitions between frames
        - Consistent axis scaling throughout
        - Time progression clearly visible
        
        Visual Enhancements:
        - Trail effects in phase space show history
        - Color-coded variables for easy identification
        - Grid lines for quantitative reading
        - Professional labeling and titles
        
        INTERPRETATION GUIDE:
        
        Spatial Panel Patterns:
        - Traveling waves: Profiles that move left/right
        - Standing waves: Oscillating amplitudes at fixed positions
        - Pattern formation: Emergence of spatial structures
        - Boundary effects: Behavior near x=0 and x=1
        
        Phase Space Panel Dynamics:
        - Limit cycles: Closed trajectories indicating oscillations
        - Spirals: Approach to or departure from equilibrium
        - Complex orbits: Multi-modal or chaotic behavior
        - Excitation events: Large rapid excursions from rest
        
        Combined Analysis:
        - Correlation between spatial and temporal patterns
        - Wave propagation visible as spatial shifts
        - Oscillations appear as repeated phase cycles
        - Nonlinear effects visible as trajectory curvature
        
        TECHNICAL SPECIFICATIONS:
        
        File Format:
        - GIF: Widely compatible, looping animation
        - Pillow backend: High-quality rendering
        - Optimized file size for sharing
        
        Frame Rate:
        - Default 10 fps: Smooth but manageable file size
        - Adjustable for different purposes
        - Higher fps: Smoother but larger files
        - Lower fps: Smaller files, choppier motion
        
        Quality Features:
        - Consistent color mapping across frames
        - Anti-aliased lines for smooth appearance
        - Proper aspect ratios maintained
        - Legend and labels always visible
        
        USE CASES:
        
        Research Presentations:
        - Dynamic illustration of solution behavior
        - Clear visualization of wave phenomena
        - Effective communication of results
        
        Educational Materials:
        - Demonstrate PDE solution evolution
        - Show connection between equations and dynamics
        - Illustrate phase space concepts
        
        Publication Support:
        - Supplementary material for papers
        - High-quality figures for talks
        - Shareable documentation of results
        
        Args:
            solution: Solution dictionary with 't', 's1', 's2', 'x'
            filename: Output GIF filename
            fps: Animation frame rate (frames per second)
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        t = solution['t']
        s1 = solution['s1']
        s2 = solution['s2']
        
        # Initialize plots
        line1, = ax1.plot([], [], 'b-', linewidth=2, label='s₁')
        line2, = ax1.plot([], [], 'r-', linewidth=2, label='s₂')
        ax1.set_xlim(self.x[0], self.x[-1])
        ax1.set_ylim(min(s1.min(), s2.min()), max(s1.max(), s2.max()))
        ax1.set_xlabel('x')
        ax1.set_ylabel('State')
        ax1.legend()
        ax1.grid(True)
        
        # Phase space plot
        line3, = ax2.plot([], [], 'g-', linewidth=1, alpha=0.7)
        point, = ax2.plot([], [], 'ro', markersize=8)
        mid_idx = self.nx // 2
        ax2.set_xlim(s1[:, mid_idx].min(), s1[:, mid_idx].max())
        ax2.set_ylim(s2[:, mid_idx].min(), s2[:, mid_idx].max())
        ax2.set_xlabel('s₁')
        ax2.set_ylabel('s₂')
        ax2.set_title(f'Phase Space at x = {self.x[mid_idx]:.2f}')
        ax2.grid(True)
        
        def animate(frame):
            # Update spatial profiles
            line1.set_data(self.x, s1[frame, :])
            line2.set_data(self.x, s2[frame, :])
            ax1.set_title(f'Spatial Profiles at t = {t[frame]:.3f}')
            
            # Update phase space
            line3.set_data(s1[:frame+1, mid_idx], s2[:frame+1, mid_idx])
            point.set_data([s1[frame, mid_idx]], [s2[frame, mid_idx]])
            
            return line1, line2, line3, point
        
        # Create animation
        anim = animation.FuncAnimation(fig, animate, frames=len(t),
                                     interval=1000//fps, blit=True, repeat=True)
        
        # Save animation
        anim.save(filename, writer='pillow', fps=fps)
        plt.show()
        
        print(f"Animation saved as {filename}")


if __name__ == "__main__":
    # Test visualization with synthetic data
    nx = 50
    x = np.linspace(0, 1, nx)
    visualizer = FitzHughNagumoVisualizer(x)
    
    # Create synthetic solution data
    nt = 100
    t = np.linspace(0, 4, nt)
    s1 = np.random.randn(nt, nx)
    s2 = np.random.randn(nt, nx)
    
    solution = {'t': t, 's1': s1, 's2': s2, 'x': x}
    
    print("Visualization module loaded successfully.")
    print("Create synthetic data for testing...")
    
    # Test some plotting functions
    visualizer.plot_solution_evolution(solution, "Test Solution")
    visualizer.plot_phase_space(solution, title="Test Phase Space")