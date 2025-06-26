"""
Phase 5: Validation and Verification

This module implements validation of the learned ROM against high-fidelity
solutions and reproduces the results from the paper.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Callable, Optional
from .fitzhugh_nagumo_solver import FitzHughNagumoSolver, default_input_function
from .lifting_transformation import LiftingTransformation, DataGenerator
from .pod_reduction import PODReducer
from .operator_inference import OperatorInference


class ValidationSuite:
    """Comprehensive validation suite for the Lift & Learn method."""
    
    def __init__(self, solver: FitzHughNagumoSolver,
                 lifting: LiftingTransformation,
                 pod: PODReducer,
                 opi: OperatorInference):
        """
        Initialize validation suite.
        
        Args:
            solver: High-fidelity solver
            lifting: Lifting transformation
            pod: POD reducer
            opi: Operator inference
        """
        self.solver = solver
        self.lifting = lifting
        self.pod = pod
        self.opi = opi
        
    def validate_single_prediction(self, 
                                 alpha: float, 
                                 beta: float,
                                 t_span: Tuple[float, float],
                                 t_eval: np.ndarray,
                                 g_func: Callable[[float], float] = default_input_function,
                                 plot: bool = True) -> Dict[str, float]:
        """
        Validate ROM prediction against high-fidelity solution for single parameter set.
        
        Args:
            alpha: Parameter α
            beta: Parameter β
            t_span: Time span
            t_eval: Time evaluation points
            g_func: Input function
            plot: Whether to create plots
            
        Returns:
            Dictionary with error metrics
        """
        print(f"Validating prediction for α={alpha:.2f}, β={beta:.2f}")
        
        # Solve high-fidelity system
        hf_solution = self.solver.solve(t_span, t_eval, alpha, beta, g_func)
        
        if not hf_solution['success']:
            raise RuntimeError(f"High-fidelity solver failed: {hf_solution['message']}")
        
        # Prepare initial condition for ROM
        s1_0 = hf_solution['s1'][0, :]
        s2_0 = hf_solution['s2'][0, :]
        s_0 = np.concatenate([s1_0, s2_0])
        w_0 = self.lifting.lift(s_0)
        w_hat_0 = self.pod.project(w_0)
        
        # Solve ROM
        rom_solution = self.opi.predict(t_span, t_eval, w_hat_0, alpha, beta, g_func)
        
        if not rom_solution['success']:
            print(f"Warning: ROM solver failed: {rom_solution['message']}")
            return {'error': np.inf, 'success': False}
        
        # Compute error
        error = self.opi.compute_prediction_error(rom_solution, hf_solution)
        
        # Plotting
        if plot:
            self._plot_comparison(hf_solution, rom_solution, alpha, beta, error)
        
        return {
            'error': error,
            'success': True,
            'hf_solution': hf_solution,
            'rom_solution': rom_solution
        }
    
    def validate_parameter_sweep(self,
                               alpha_range: Tuple[float, float],
                               beta_range: Tuple[float, float],
                               n_test: int,
                               t_span: Tuple[float, float],
                               t_eval: np.ndarray,
                               g_func: Callable[[float], float] = default_input_function) -> Dict[str, np.ndarray]:
        """
        Validate ROM over a range of parameters.
        
        Args:
            alpha_range: Range of α values (min, max)
            beta_range: Range of β values (min, max)
            n_test: Number of test points
            t_span: Time span
            t_eval: Time evaluation points
            g_func: Input function
            
        Returns:
            Dictionary with validation results
        """
        print(f"Parameter sweep validation with {n_test} test points")
        
        # Generate test parameters
        np.random.seed(123)  # For reproducible test results
        alpha_test = np.random.uniform(alpha_range[0], alpha_range[1], n_test)
        beta_test = np.random.uniform(beta_range[0], beta_range[1], n_test)
        
        errors = []
        successful_tests = 0
        
        for i, (alpha, beta) in enumerate(zip(alpha_test, beta_test)):
            try:
                result = self.validate_single_prediction(
                    alpha, beta, t_span, t_eval, g_func, plot=False
                )
                errors.append(result['error'])
                if result['success']:
                    successful_tests += 1
                    
            except Exception as e:
                print(f"Test {i+1} failed: {e}")
                errors.append(np.inf)
        
        errors = np.array(errors)
        valid_errors = errors[np.isfinite(errors)]
        
        print(f"Successful tests: {successful_tests}/{n_test}")
        if len(valid_errors) > 0:
            print(f"Error statistics:")
            print(f"  Mean: {np.mean(valid_errors):.2e}")
            print(f"  Median: {np.median(valid_errors):.2e}")
            print(f"  Min: {np.min(valid_errors):.2e}")
            print(f"  Max: {np.max(valid_errors):.2e}")
        
        return {
            'alpha_test': alpha_test,
            'beta_test': beta_test,
            'errors': errors,
            'success_rate': successful_tests / n_test,
            'valid_errors': valid_errors
        }
    
    def validate_reduced_dimensions(self,
                                  r_values: List[int],
                                  training_params: List[Tuple[float, float]],
                                  test_params: List[Tuple[float, float]],
                                  t_span: Tuple[float, float],
                                  t_eval: np.ndarray,
                                  g_func: Callable[[float], float] = default_input_function) -> Dict[str, np.ndarray]:
        """
        Validate ROM accuracy vs reduced dimension (reproduce Figure 2 from paper).
        
        Args:
            r_values: List of reduced dimensions to test
            training_params: Training parameter sets
            test_params: Test parameter sets
            t_span: Time span
            t_eval: Time evaluation points
            g_func: Input function
            
        Returns:
            Dictionary with results for each dimension
        """
        print(f"Testing reduced dimensions: {r_values}")
        
        # Generate training data once
        data_gen = DataGenerator(self.solver, self.lifting)
        training_data = data_gen.generate_training_data(
            training_params, t_span, t_eval, g_func
        )
        
        results = {
            'r_values': r_values,
            'errors': [],
            'training_errors': [],
            'test_errors': []
        }
        
        for r in r_values:
            print(f"\nTesting r = {r}")
            
            # Compute POD basis for this dimension
            pod_r = PODReducer()
            pod_data = pod_r.compute_pod_basis(training_data['W'], r=r)
            
            # Project training data
            W_reduced = pod_r.project(training_data['W'])
            
            # Learn operators
            opi_r = OperatorInference(self.solver, self.lifting, pod_r)
            operators = opi_r.learn_operators(W_reduced, training_data['P'], g_func)
            
            # Test on training parameters (should be low error)
            train_errors = []
            for alpha, beta in training_params[:3]:  # Test subset for speed
                try:
                    result = self._test_single_parameter(
                        opi_r, alpha, beta, t_span, t_eval, g_func
                    )
                    train_errors.append(result['error'])
                except:
                    train_errors.append(np.inf)
            
            # Test on test parameters
            test_errors = []
            for alpha, beta in test_params:
                try:
                    result = self._test_single_parameter(
                        opi_r, alpha, beta, t_span, t_eval, g_func
                    )
                    test_errors.append(result['error'])
                except:
                    test_errors.append(np.inf)
            
            # Store results
            train_errors = np.array(train_errors)
            test_errors = np.array(test_errors)
            
            valid_train = train_errors[np.isfinite(train_errors)]
            valid_test = test_errors[np.isfinite(test_errors)]
            
            results['training_errors'].append(np.median(valid_train) if len(valid_train) > 0 else np.inf)
            results['test_errors'].append(np.median(valid_test) if len(valid_test) > 0 else np.inf)
            results['errors'].append(np.median(valid_test) if len(valid_test) > 0 else np.inf)
            
            print(f"  Training error (median): {results['training_errors'][-1]:.2e}")
            print(f"  Test error (median): {results['test_errors'][-1]:.2e}")
        
        return results
    
    def _test_single_parameter(self, opi: OperatorInference, 
                              alpha: float, beta: float,
                              t_span: Tuple[float, float],
                              t_eval: np.ndarray,
                              g_func: Callable[[float], float]) -> Dict[str, float]:
        """Helper function for testing single parameter set."""
        # High-fidelity solution
        hf_solution = self.solver.solve(t_span, t_eval, alpha, beta, g_func)
        
        # Initial condition for ROM
        s1_0 = hf_solution['s1'][0, :]
        s2_0 = hf_solution['s2'][0, :]
        s_0 = np.concatenate([s1_0, s2_0])
        w_0 = self.lifting.lift(s_0)
        w_hat_0 = opi.pod.project(w_0)
        
        # ROM solution
        rom_solution = opi.predict(t_span, t_eval, w_hat_0, alpha, beta, g_func)
        
        # Error
        error = opi.compute_prediction_error(rom_solution, hf_solution)
        
        return {'error': error}
    
    def _plot_comparison(self, hf_solution: Dict, rom_solution: Dict,
                        alpha: float, beta: float, error: float):
        """Plot comparison between high-fidelity and ROM solutions."""
        # Reconstruct ROM solution in original coordinates
        w_hat = rom_solution['w_hat']
        w_full = self.pod.reconstruct(w_hat)
        
        nt = len(rom_solution['t'])
        s1_rom = np.zeros((nt, self.solver.nx))
        s2_rom = np.zeros((nt, self.solver.nx))
        
        for i in range(nt):
            s_rom = self.lifting.unlift(w_full[:, i])
            s1_rom[i, :] = s_rom[:self.solver.nx]
            s2_rom[i, :] = s_rom[self.solver.nx:]
        
        # Create plots
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # s1 at final time
        axes[0, 0].plot(self.solver.x, hf_solution['s1'][-1, :], 'b-', label='High-fidelity', linewidth=2)
        axes[0, 0].plot(self.solver.x, s1_rom[-1, :], 'r--', label='ROM', linewidth=2)
        axes[0, 0].set_xlabel('x')
        axes[0, 0].set_ylabel('s₁')
        axes[0, 0].set_title(f's₁ at t = {hf_solution["t"][-1]:.2f}')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # s2 at final time
        axes[0, 1].plot(self.solver.x, hf_solution['s2'][-1, :], 'b-', label='High-fidelity', linewidth=2)
        axes[0, 1].plot(self.solver.x, s2_rom[-1, :], 'r--', label='ROM', linewidth=2)
        axes[0, 1].set_xlabel('x')
        axes[0, 1].set_ylabel('s₂')
        axes[0, 1].set_title(f's₂ at t = {hf_solution["t"][-1]:.2f}')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Time evolution at x = 0.5
        mid_idx = len(self.solver.x) // 2
        axes[1, 0].plot(hf_solution['t'], hf_solution['s1'][:, mid_idx], 'b-', label='High-fidelity', linewidth=2)
        axes[1, 0].plot(rom_solution['t'], s1_rom[:, mid_idx], 'r--', label='ROM', linewidth=2)
        axes[1, 0].set_xlabel('t')
        axes[1, 0].set_ylabel('s₁')
        axes[1, 0].set_title(f's₁ at x = {self.solver.x[mid_idx]:.2f}')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Error vs time
        error_t = np.zeros(len(rom_solution['t']))
        for i in range(len(rom_solution['t'])):
            hf_state = np.concatenate([hf_solution['s1'][i, :], hf_solution['s2'][i, :]])
            rom_state = np.concatenate([s1_rom[i, :], s2_rom[i, :]])
            error_t[i] = np.linalg.norm(hf_state - rom_state) / np.linalg.norm(hf_state)
        
        axes[1, 1].semilogy(rom_solution['t'], error_t, 'g-', linewidth=2)
        axes[1, 1].set_xlabel('t')
        axes[1, 1].set_ylabel('Relative Error')
        axes[1, 1].set_title('Error Evolution')
        axes[1, 1].grid(True)
        
        plt.suptitle(f'ROM Validation: α={alpha:.2f}, β={beta:.2f}, Error={error:.2e}')
        plt.tight_layout()
        
        # Auto-save for GitHub README  
        auto_save_path = "images/plots/rom_comparison.png"
        plt.savefig(auto_save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {auto_save_path}")
        
        plt.show()
    
    def plot_error_vs_dimension(self, validation_results: Dict):
        """
        Plot error vs reduced dimension - reproduces key result from the Lift & Learn paper.
        
        This creates the critical validation plot showing ROM accuracy as a function
        of the number of POD modes retained (reduced dimension r):
        
        PLOT ELEMENTS:
        - X-axis: Reduced dimension r (number of POD modes)
        - Y-axis: Relative prediction error (log scale)
        - Blue line with circles: Training error (ROM tested on training parameters)
        - Red line with squares: Test error (ROM tested on new parameters)
        - Markers: Individual dimension tests
        - Log scale: Reveals error decay over several orders of magnitude
        
        INTERPRETATION GUIDE:
        
        Expected Behavior (Good Implementation):
        - Training error: Monotonically decreasing with r
        - Test error: U-shaped or L-shaped curve
        - Both errors: Several orders of magnitude decrease
        - Convergence: Errors plateau at machine precision or model limitations
        
        Key Observations:
        
        1. Training vs Test Error Gap:
        - Small gap: Good generalization, robust ROM
        - Large gap: Overfitting to training data
        - Growing gap with r: Too many modes for available data
        
        2. Optimal Dimension Selection:
        - Test error minimum: Best balance of approximation vs overfitting
        - Training error never increases: More modes = better training fit
        - Practical choice: Where test error levels off (elbow point)
        
        3. Error Decay Rate:
        - Fast initial decay: Efficient POD compression
        - Slow decay: Complex dynamics requiring many modes
        - Plateaus: Hitting fundamental limitations
        
        PHYSICAL INTERPRETATION:
        
        Low r (r < 5):
        - High errors: Missing essential dynamics
        - Smooth approximations only
        - Major patterns captured but details lost
        
        Medium r (5 < r < 20):
        - Decreasing errors: Capturing more physics
        - Good trade-off region
        - Most practical applications optimal here
        
        High r (r > 20):
        - Marginal improvements: Diminishing returns
        - Risk of overfitting
        - Computational benefits reduced
        
        PAPER COMPARISON:
        
        Expected Results (from FitzHugh-Nagumo paper section):
        - Errors should drop from ~1e-1 to ~1e-3 or better
        - Optimal r typically around 10-15 for this system
        - Training error should be consistently lower than test error
        - Both should plateau rather than continuing to decrease
        
        TROUBLESHOOTING GUIDE:
        
        Poor Results (High Errors):
        - Check POD basis quality (singular value decay)
        - Verify operator inference convergence
        - Increase training data coverage
        - Check lifting transformation implementation
        
        Overfitting Signs:
        - Test error increases with r while training error decreases
        - Large gap between training and test errors
        - Oscillatory test error behavior
        
        Underfitting Signs:
        - Both errors high and not decreasing much
        - Insufficient POD modes or poor basis
        - May need different lifting transformation
        
        PRACTICAL RECOMMENDATIONS:
        
        Dimension Selection:
        - Choose r at test error minimum or elbow
        - Consider computational cost vs accuracy trade-off
        - Validate with additional test cases
        - Use cross-validation for robust selection
        
        Quality Indicators:
        - Test error < 1e-2: Excellent for most applications
        - Test error < 1e-1: Acceptable for qualitative studies
        - Test error > 1e-1: Needs improvement
        
        Args:
            validation_results: Dictionary containing:
                               - 'r_values': List of tested reduced dimensions
                               - 'training_errors': Training error for each r
                               - 'test_errors': Test error for each r
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        r_values = validation_results['r_values']
        train_errors = validation_results['training_errors']
        test_errors = validation_results['test_errors']
        
        # Remove infinite errors for plotting
        valid_indices = [i for i, e in enumerate(test_errors) if np.isfinite(e)]
        r_plot = [r_values[i] for i in valid_indices]
        train_plot = [train_errors[i] for i in valid_indices]
        test_plot = [test_errors[i] for i in valid_indices]
        
        ax.semilogy(r_plot, train_plot, 'bo-', label='Training Error', linewidth=2, markersize=8)
        ax.semilogy(r_plot, test_plot, 'rs-', label='Test Error', linewidth=2, markersize=8)
        
        ax.set_xlabel('Reduced Dimension r')
        ax.set_ylabel('Relative Error')
        ax.set_title('ROM Accuracy vs Reduced Dimension')
        ax.legend()
        ax.grid(True)
        
        plt.tight_layout()
        
        # Auto-save for GitHub README
        auto_save_path = "images/plots/error_vs_dimension.png"
        plt.savefig(auto_save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {auto_save_path}")
        
        plt.show()


if __name__ == "__main__":
    print("Validation module loaded successfully.")
    print("Run the main execution script to perform full validation.")