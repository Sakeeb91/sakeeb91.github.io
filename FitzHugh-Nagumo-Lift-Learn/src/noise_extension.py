"""
Extension: Robustness to Noise Analysis

This module extends the Lift & Learn method to investigate robustness
to noisy data and the effectiveness of regularization techniques.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Callable
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from .fitzhugh_nagumo_solver import FitzHughNagumoSolver, default_input_function
from .lifting_transformation import LiftingTransformation, DataGenerator
from .pod_reduction import PODReducer
from .operator_inference import OperatorInference
from .validation import ValidationSuite


class NoiseAnalysis:
    """Analysis of noise robustness for the Lift & Learn method."""
    
    def __init__(self, solver: FitzHughNagumoSolver,
                 lifting: LiftingTransformation):
        """
        Initialize noise analysis.
        
        Args:
            solver: High-fidelity solver
            lifting: Lifting transformation
        """
        self.solver = solver
        self.lifting = lifting
    
    def add_noise_to_snapshots(self, snapshots: np.ndarray,
                              noise_level: float,
                              noise_type: str = 'gaussian') -> np.ndarray:
        """
        Add noise to snapshot data.
        
        Args:
            snapshots: Clean snapshot matrix (n_states, n_snapshots)
            noise_level: Noise level (fraction of signal RMS)
            noise_type: Type of noise ('gaussian', 'uniform')
            
        Returns:
            Noisy snapshot matrix
        """
        signal_rms = np.sqrt(np.mean(snapshots**2))
        noise_std = noise_level * signal_rms
        
        if noise_type == 'gaussian':
            noise = np.random.normal(0, noise_std, snapshots.shape)
        elif noise_type == 'uniform':
            noise_range = noise_std * np.sqrt(3)  # For same RMS as Gaussian
            noise = np.random.uniform(-noise_range, noise_range, snapshots.shape)
        else:
            raise ValueError(f"Unknown noise type: {noise_type}")
        
        return snapshots + noise
    
    def regularized_operator_inference(self, reduced_states: np.ndarray,
                                     parameters: np.ndarray,
                                     g_func: Callable[[float], float],
                                     regularization_params: List[Dict],
                                     solver: FitzHughNagumoSolver,
                                     lifting: LiftingTransformation,
                                     pod: PODReducer) -> Dict[str, Dict]:
        """
        Learn operators with different regularization methods.
        
        Args:
            reduced_states: Reduced states
            parameters: Parameter matrix
            g_func: Input function
            regularization_params: List of regularization configurations
            solver: High-fidelity solver
            lifting: Lifting transformation
            pod: POD reducer
            
        Returns:
            Dictionary of results for each regularization method
        """
        results = {}
        
        for reg_config in regularization_params:
            reg_name = reg_config.get('name', 'unknown')
            print(f"Learning operators with {reg_name} regularization...")
            
            # Create operator inference with regularization
            opi = OperatorInference(solver, lifting, pod)
            
            try:
                operators = opi.learn_operators(
                    reduced_states, parameters, g_func,
                    regularization=reg_config
                )
                
                results[reg_name] = {
                    'operators': operators,
                    'opi': opi,
                    'success': True
                }
                
            except Exception as e:
                print(f"  Failed: {e}")
                results[reg_name] = {
                    'success': False,
                    'error': str(e)
                }
        
        return results
    
    def analyze_noise_robustness(self,
                                training_params: List[Tuple[float, float]],
                                test_params: List[Tuple[float, float]],
                                t_span: Tuple[float, float],
                                t_eval: np.ndarray,
                                r: int,
                                noise_levels: List[float],
                                regularization_methods: List[Dict],
                                g_func: Callable[[float], float] = default_input_function,
                                n_trials: int = 5) -> Dict:
        """
        Comprehensive noise robustness analysis.
        
        Args:
            training_params: Training parameter sets
            test_params: Test parameter sets
            t_span: Time span
            t_eval: Time evaluation points
            r: Reduced dimension
            noise_levels: List of noise levels to test
            regularization_methods: List of regularization configurations
            g_func: Input function
            n_trials: Number of trials per noise level
            
        Returns:
            Dictionary with comprehensive results
        """
        print(f"Noise robustness analysis:")
        print(f"  Noise levels: {noise_levels}")
        print(f"  Regularization methods: {[m['name'] for m in regularization_methods]}")
        print(f"  Trials per noise level: {n_trials}")
        
        # Generate clean training data once
        data_gen = DataGenerator(self.solver, self.lifting)
        clean_data = data_gen.generate_training_data(
            training_params, t_span, t_eval, g_func
        )
        
        results = {
            'noise_levels': noise_levels,
            'regularization_methods': [m['name'] for m in regularization_methods],
            'results': {}
        }
        
        for noise_level in noise_levels:
            print(f"\nTesting noise level: {noise_level}")
            noise_results = {
                'method_results': {},
                'summary': {}
            }
            
            for reg_config in regularization_methods:
                reg_name = reg_config['name']
                print(f"  Method: {reg_name}")
                
                trial_errors = []
                
                for trial in range(n_trials):
                    try:
                        # Add noise to training data
                        noisy_snapshots = self.add_noise_to_snapshots(
                            clean_data['W'], noise_level
                        )
                        
                        # Compute POD basis
                        pod = PODReducer()
                        pod.compute_pod_basis(noisy_snapshots, r=r)
                        
                        # Project data
                        reduced_states = pod.project(noisy_snapshots)
                        
                        # Learn operators with regularization
                        opi = OperatorInference(self.solver, self.lifting, pod)
                        operators = opi.learn_operators(
                            reduced_states, clean_data['P'], g_func,
                            regularization=reg_config
                        )
                        
                        # Test on clean test data
                        test_errors = []
                        for alpha, beta in test_params:
                            hf_solution = self.solver.solve(t_span, t_eval, alpha, beta, g_func)
                            
                            # Initial condition
                            s1_0 = hf_solution['s1'][0, :]
                            s2_0 = hf_solution['s2'][0, :]
                            s_0 = np.concatenate([s1_0, s2_0])
                            w_0 = self.lifting.lift(s_0)
                            w_hat_0 = pod.project(w_0)
                            
                            # ROM prediction
                            rom_solution = opi.predict(t_span, t_eval, w_hat_0, alpha, beta, g_func)
                            
                            if rom_solution['success']:
                                error = opi.compute_prediction_error(rom_solution, hf_solution)
                                test_errors.append(error)
                        
                        if test_errors:
                            trial_errors.append(np.median(test_errors))
                        else:
                            trial_errors.append(np.inf)
                            
                    except Exception as e:
                        print(f"    Trial {trial+1} failed: {e}")
                        trial_errors.append(np.inf)
                
                # Summarize trial results
                trial_errors = np.array(trial_errors)
                valid_errors = trial_errors[np.isfinite(trial_errors)]
                
                if len(valid_errors) > 0:
                    method_summary = {
                        'mean_error': np.mean(valid_errors),
                        'std_error': np.std(valid_errors),
                        'median_error': np.median(valid_errors),
                        'success_rate': len(valid_errors) / n_trials,
                        'all_errors': trial_errors
                    }
                else:
                    method_summary = {
                        'mean_error': np.inf,
                        'std_error': 0,
                        'median_error': np.inf,
                        'success_rate': 0,
                        'all_errors': trial_errors
                    }
                
                noise_results['method_results'][reg_name] = method_summary
                
                print(f"    Success rate: {method_summary['success_rate']:.1%}")
                if method_summary['success_rate'] > 0:
                    print(f"    Mean error: {method_summary['mean_error']:.2e}")
            
            results['results'][noise_level] = noise_results
        
        return results
    
    def plot_noise_analysis_results(self, results: Dict):
        """
        Plot comprehensive results of noise robustness analysis.
        
        This creates a 2x2 subplot layout analyzing ROM performance under noisy conditions:
        
        1. Mean Error vs Noise Level (Top-Left):
        - X-axis: Noise level (fraction of signal RMS)
        - Y-axis: Mean prediction error (log scale)
        - Multiple lines: Different regularization methods
        - Markers: Individual noise level tests
        - Interpretation: Shows degradation of ROM accuracy with noise
          * Flat lines: Robust methods that handle noise well
          * Steep slopes: Methods sensitive to noise
          * Crossing lines: Different methods optimal at different noise levels
          * No-regularization typically performs worst
        
        2. Success Rate vs Noise Level (Top-Right):
        - X-axis: Noise level (fraction of signal RMS)
        - Y-axis: Success rate (fraction of stable solutions)
        - Multiple lines: Different regularization methods
        - Interpretation: Shows robustness of operator learning
          * High success rate: Method produces stable ROMs
          * Declining success: Noise causes unstable operator learning
          * Ridge often most robust, LASSO can be unstable
          * Success rate < 50% indicates method breakdown
        
        3. Error Distribution at High Noise (Bottom-Left):
        - X-axis: Prediction error
        - Y-axis: Probability density (log scale)
        - Histogram bars: Error distribution for different methods
        - Shows: Statistical spread of errors at challenging noise level
        - Interpretation: Reveals failure modes and consistency
          * Narrow distributions: Consistent performance
          * Wide distributions: Unpredictable performance
          * Bimodal: Some cases work well, others fail completely
          * Long tails: Occasional catastrophic failures
        
        4. Error Variability vs Noise Level (Bottom-Right):
        - X-axis: Noise level
        - Y-axis: Error standard deviation (log scale)
        - Multiple lines: Different regularization methods
        - Interpretation: Shows prediction consistency
          * Low variability: Consistent performance across trials
          * High variability: Unpredictable behavior
          * Increasing trends: Noise makes performance less reliable
          * Plateaus: Method breakdown threshold
        
        INTERPRETATION GUIDE:
        
        Method Performance Ranking:
        - Ridge Regression (typically best): Smooth regularization, stable
        - Elastic Net: Balanced L1/L2, good compromise
        - LASSO: Sparse solutions, can be unstable with noise
        - No Regularization (typically worst): Overfits to noise
        
        Noise Level Effects:
        - Low noise (< 1%): All methods may work acceptably
        - Medium noise (1-5%): Regularization becomes important
        - High noise (> 5%): Only robust methods survive
        - Extreme noise (> 10%): Most methods fail
        
        Practical Recommendations:
        - Use Ridge regression for noisy data
        - Tune regularization parameter based on noise level
        - Monitor success rate, not just accuracy
        - Consider ensemble methods for extreme noise
        
        Warning Signs:
        - Success rate < 80%: Method unreliable
        - Error variance > 10x mean: Inconsistent performance
        - Bimodal error distributions: Some cases fail completely
        - Sudden performance drops: Method breakdown threshold
        
        Research Insights:
        - Regularization is essential for practical applications
        - There's a trade-off between accuracy and robustness
        - Noise affects operator learning more than POD
        - Some physics-based regularization may outperform standard methods
        
        Args:
            results: Dictionary with noise analysis results containing:
                    - 'noise_levels': List of tested noise levels
                    - 'regularization_methods': List of method names  
                    - 'results': Nested dict with detailed results per noise level
        """
        noise_levels = results['noise_levels']
        methods = results['regularization_methods']
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot 1: Mean error vs noise level
        ax1 = axes[0, 0]
        for method in methods:
            mean_errors = []
            for noise_level in noise_levels:
                method_result = results['results'][noise_level]['method_results'][method]
                mean_errors.append(method_result['mean_error'])
            
            valid_indices = [i for i, e in enumerate(mean_errors) if np.isfinite(e)]
            noise_plot = [noise_levels[i] for i in valid_indices]
            error_plot = [mean_errors[i] for i in valid_indices]
            
            if error_plot:
                ax1.semilogy(noise_plot, error_plot, 'o-', label=method, linewidth=2, markersize=6)
        
        ax1.set_xlabel('Noise Level')
        ax1.set_ylabel('Mean Prediction Error')
        ax1.set_title('Error vs Noise Level')
        ax1.legend()
        ax1.grid(True)
        
        # Plot 2: Success rate vs noise level
        ax2 = axes[0, 1]
        for method in methods:
            success_rates = []
            for noise_level in noise_levels:
                method_result = results['results'][noise_level]['method_results'][method]
                success_rates.append(method_result['success_rate'])
            
            ax2.plot(noise_levels, success_rates, 'o-', label=method, linewidth=2, markersize=6)
        
        ax2.set_xlabel('Noise Level')
        ax2.set_ylabel('Success Rate')
        ax2.set_title('Success Rate vs Noise Level')
        ax2.legend()
        ax2.grid(True)
        
        # Plot 3: Error distribution for highest noise level
        ax3 = axes[1, 0]
        highest_noise = noise_levels[-1]
        
        for i, method in enumerate(methods):
            method_result = results['results'][highest_noise]['method_results'][method]
            errors = method_result['all_errors']
            valid_errors = errors[np.isfinite(errors)]
            
            if len(valid_errors) > 0:
                ax3.hist(valid_errors, bins=10, alpha=0.7, label=method, 
                        density=True, histtype='step', linewidth=2)
        
        ax3.set_xlabel('Prediction Error')
        ax3.set_ylabel('Density')
        ax3.set_title(f'Error Distribution (Noise Level = {highest_noise})')
        ax3.legend()
        ax3.grid(True)
        ax3.set_yscale('log')
        
        # Plot 4: Error variance vs noise level
        ax4 = axes[1, 1]
        for method in methods:
            error_stds = []
            for noise_level in noise_levels:
                method_result = results['results'][noise_level]['method_results'][method]
                error_stds.append(method_result['std_error'])
            
            valid_indices = [i for i, e in enumerate(error_stds) if np.isfinite(e)]
            noise_plot = [noise_levels[i] for i in valid_indices]
            std_plot = [error_stds[i] for i in valid_indices]
            
            if std_plot:
                ax4.semilogy(noise_plot, std_plot, 'o-', label=method, linewidth=2, markersize=6)
        
        ax4.set_xlabel('Noise Level')
        ax4.set_ylabel('Error Standard Deviation')
        ax4.set_title('Error Variability vs Noise Level')
        ax4.legend()
        ax4.grid(True)
        
        plt.tight_layout()
        
        # Auto-save for GitHub README
        auto_save_path = "images/plots/noise_robustness.png"
        plt.savefig(auto_save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {auto_save_path}")
        
        plt.show()
        
    def compare_regularization_methods(self,
                                     training_params: List[Tuple[float, float]],
                                     test_params: List[Tuple[float, float]],
                                     t_span: Tuple[float, float],
                                     t_eval: np.ndarray,
                                     r: int,
                                     noise_level: float,
                                     lambda_values: List[float],
                                     g_func: Callable[[float], float] = default_input_function) -> Dict:
        """
        Compare different regularization methods and hyperparameters.
        
        Args:
            training_params: Training parameters
            test_params: Test parameters
            t_span: Time span
            t_eval: Time evaluation points
            r: Reduced dimension
            noise_level: Fixed noise level for comparison
            lambda_values: Regularization parameters to test
            g_func: Input function
            
        Returns:
            Comparison results
        """
        print(f"Comparing regularization methods at noise level {noise_level}")
        
        # Generate noisy training data
        data_gen = DataGenerator(self.solver, self.lifting)
        clean_data = data_gen.generate_training_data(
            training_params, t_span, t_eval, g_func
        )
        noisy_snapshots = self.add_noise_to_snapshots(clean_data['W'], noise_level)
        
        # Compute POD basis
        pod = PODReducer()
        pod.compute_pod_basis(noisy_snapshots, r=r)
        reduced_states = pod.project(noisy_snapshots)
        
        results = {
            'lambda_values': lambda_values,
            'methods': {}
        }
        
        reg_methods = ['ridge', 'lasso', 'elastic_net']
        
        for method in reg_methods:
            print(f"  Testing {method} regularization")
            method_results = {
                'lambda_values': lambda_values,
                'train_errors': [],
                'test_errors': [],
                'sparsity': []
            }
            
            for lambda_val in lambda_values:
                reg_config = {
                    'type': method,
                    'lambda': lambda_val,
                    'name': f'{method}_lambda_{lambda_val}'
                }
                
                if method == 'elastic_net':
                    reg_config['l1_ratio'] = 0.5  # Mix of L1 and L2
                
                try:
                    # Learn operators
                    opi = OperatorInference(self.solver, self.lifting, pod)
                    operators = opi.learn_operators(
                        reduced_states, clean_data['P'], g_func,
                        regularization=reg_config
                    )
                    
                    # Test errors
                    test_errors = []
                    for alpha, beta in test_params[:3]:  # Subset for speed
                        hf_solution = self.solver.solve(t_span, t_eval, alpha, beta, g_func)
                        
                        s1_0 = hf_solution['s1'][0, :]
                        s2_0 = hf_solution['s2'][0, :]
                        s_0 = np.concatenate([s1_0, s2_0])
                        w_0 = self.lifting.lift(s_0)
                        w_hat_0 = pod.project(w_0)
                        
                        rom_solution = opi.predict(t_span, t_eval, w_hat_0, alpha, beta, g_func)
                        
                        if rom_solution['success']:
                            error = opi.compute_prediction_error(rom_solution, hf_solution)
                            test_errors.append(error)
                    
                    # Compute sparsity (fraction of near-zero coefficients)
                    A = operators['A']
                    H = operators['H']
                    B = operators['B']
                    
                    all_coeffs = np.concatenate([A.flatten(), H.flatten(), B.flatten()])
                    sparsity = np.mean(np.abs(all_coeffs) < 1e-6)
                    
                    method_results['test_errors'].append(np.median(test_errors) if test_errors else np.inf)
                    method_results['sparsity'].append(sparsity)
                    
                except Exception as e:
                    print(f"    Lambda {lambda_val} failed: {e}")
                    method_results['test_errors'].append(np.inf)
                    method_results['sparsity'].append(0)
            
            results['methods'][method] = method_results
        
        return results
    
    def plot_regularization_comparison(self, results: Dict):
        """
        Plot detailed comparison of regularization methods and hyperparameters.
        
        This creates a 1x2 subplot layout for hyperparameter optimization analysis:
        
        1. Error vs Regularization Parameter (Left):
        - X-axis: Regularization parameter λ (log scale)
        - Y-axis: Test error (log scale)
        - Multiple lines: Different regularization methods (Ridge, LASSO, Elastic Net)
        - Markers: Individual λ values tested
        - Features: Shows trade-off between bias and variance
        
        INTERPRETATION OF ERROR CURVES:
        
        Typical U-Shape Pattern:
        - Left side (small λ): Underfitting, high variance, overfits to noise
        - Bottom (optimal λ): Best bias-variance trade-off
        - Right side (large λ): Overfitting, high bias, underfits signal
        - Minimum point: Optimal regularization strength
        
        Method Comparisons:
        - Ridge: Usually smooth, stable curves
        - LASSO: May show discontinuities due to sparsity
        - Elastic Net: Intermediate between Ridge and LASSO
        - Crossing points: Different methods optimal in different regimes
        
        2. Sparsity vs Regularization Parameter (Right):
        - X-axis: Regularization parameter λ (log scale)
        - Y-axis: Sparsity (fraction of near-zero coefficients)
        - Multiple lines: Different regularization methods
        - Shows: Model complexity vs regularization strength
        
        INTERPRETATION OF SPARSITY PATTERNS:
        
        Ridge Regression:
        - Gradual sparsity increase: Smooth coefficient shrinkage
        - Never truly sparse: All coefficients remain non-zero
        - Useful for: Feature shrinkage without selection
        
        LASSO Regression:
        - Sharp sparsity increases: Sudden coefficient elimination
        - Can achieve high sparsity: Automatic feature selection
        - Useful for: Identifying important physics terms
        
        Elastic Net:
        - Intermediate behavior: Balanced selection and shrinkage
        - Grouped selection: Correlated features selected together
        - Useful for: Balanced regularization approaches
        
        PRACTICAL INTERPRETATION GUIDE:
        
        Optimal λ Selection:
        - Choose λ at minimum of error curve
        - Consider error bars and stability
        - Validate on independent test data
        - Balance accuracy vs interpretability
        
        Method Selection Criteria:
        - Ridge: Want to keep all physics terms but shrink them
        - LASSO: Want to identify most important terms only
        - Elastic Net: Want balanced approach for correlated terms
        - Cross-validation: Use multiple methods and compare
        
        Warning Signs:
        - No clear minimum: May need more data or different approach
        - Very flat curves: Regularization not helping much
        - Oscillatory patterns: Unstable optimization
        - High sparsity with low error: May be overfitting to sparse solution
        
        Physics Insights:
        - Low sparsity: Complex dynamics require many terms
        - High sparsity: Simple underlying physics
        - Sparsity patterns: Reveal dominant physical mechanisms
        - λ magnitude: Indicates noise level in data
        
        Hyperparameter Recommendations:
        - Grid search: Test λ values spanning several orders of magnitude
        - Cross-validation: Use k-fold CV for robust λ selection
        - Warm starts: Use solution from previous λ as initialization
        - Stability: Choose λ where small changes don't affect results much
        
        Research Applications:
        - Model discovery: Use LASSO sparsity to identify key terms
        - Robustness analysis: Compare methods across noise levels
        - Physical insight: Interpret sparsity patterns in terms of physics
        - Method development: Benchmark new regularization approaches
        
        Args:
            results: Dictionary with regularization comparison results containing:
                    - 'lambda_values': List of tested regularization parameters
                    - 'methods': Dict with results for each method ('ridge', 'lasso', etc.)
                    - Each method contains 'test_errors' and 'sparsity' lists
        """
        lambda_values = results['lambda_values']
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot 1: Error vs regularization parameter
        for method, data in results['methods'].items():
            test_errors = data['test_errors']
            valid_indices = [i for i, e in enumerate(test_errors) if np.isfinite(e)]
            lambda_plot = [lambda_values[i] for i in valid_indices]
            error_plot = [test_errors[i] for i in valid_indices]
            
            if error_plot:
                ax1.loglog(lambda_plot, error_plot, 'o-', label=method.replace('_', ' ').title(), 
                          linewidth=2, markersize=6)
        
        ax1.set_xlabel('Regularization Parameter (λ)')
        ax1.set_ylabel('Test Error')
        ax1.set_title('Error vs Regularization Strength')
        ax1.legend()
        ax1.grid(True)
        
        # Plot 2: Sparsity vs regularization parameter
        for method, data in results['methods'].items():
            sparsity = data['sparsity']
            ax2.semilogx(lambda_values, sparsity, 'o-', label=method.replace('_', ' ').title(),
                        linewidth=2, markersize=6)
        
        ax2.set_xlabel('Regularization Parameter (λ)')
        ax2.set_ylabel('Sparsity (Fraction of Near-Zero Coefficients)')
        ax2.set_title('Model Sparsity vs Regularization Strength')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        
        # Auto-save for GitHub README
        auto_save_path = "images/plots/regularization_comparison.png"
        plt.savefig(auto_save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {auto_save_path}")
        
        plt.show()


if __name__ == "__main__":
    print("Noise analysis module loaded successfully.")
    print("Run the main execution script to perform noise robustness analysis.")