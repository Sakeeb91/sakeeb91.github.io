"""
Main Execution Script for FitzHugh-Nagumo Lift & Learn Project

This script orchestrates the complete Lift & Learn methodology:
1. High-fidelity data generation
2. Lifting transformation
3. POD dimensionality reduction
4. Operator inference
5. Validation and verification
6. Noise robustness analysis (extension)

Usage:
    python main.py [--phase PHASE] [--config CONFIG]
"""

import argparse
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
import time
import json

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.fitzhugh_nagumo_solver import FitzHughNagumoSolver, default_input_function
from src.lifting_transformation import LiftingTransformation, DataGenerator
from src.pod_reduction import PODReducer
from src.operator_inference import OperatorInference
from src.validation import ValidationSuite
from src.noise_extension import NoiseAnalysis
from src.visualization import FitzHughNagumoVisualizer


class LiftLearnPipeline:
    """Complete Lift & Learn pipeline for FitzHugh-Nagumo system."""
    
    def __init__(self, config: Dict):
        """
        Initialize pipeline with configuration.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        
        # Initialize components
        self.solver = FitzHughNagumoSolver(
            nx=config['spatial']['nx'],
            L=config['spatial']['L'],
            gamma=config['system']['gamma']
        )
        
        self.lifting = LiftingTransformation(config['spatial']['nx'])
        self.pod = PODReducer()
        self.opi = None
        self.visualizer = FitzHughNagumoVisualizer(self.solver.x)
        
        # Data storage
        self.training_data = None
        self.pod_data = None
        self.operators = None
        
        print(f"Pipeline initialized:")
        print(f"  Spatial points: {config['spatial']['nx']}")
        print(f"  Domain: [0, {config['spatial']['L']}]")
        print(f"  Diffusion coefficient: {config['system']['gamma']}")
    
    def run_phase_1(self) -> None:
        """Phase 1: Generate high-fidelity training data."""
        print("\n" + "="*60)
        print("PHASE 1: High-Fidelity Data Generation")
        print("="*60)
        
        # Parameters from the paper (Section 5.1.2)
        training_params = self.config['training']['parameter_sets']
        
        # Time discretization
        t_end = self.config['time']['t_end']
        dt = self.config['time']['dt']
        t_eval = np.arange(0, t_end + dt, dt)
        
        print(f"Generating training data:")
        print(f"  Time span: [0, {t_end}]")
        print(f"  Time step: {dt}")
        print(f"  Parameter sets: {len(training_params)}")
        
        # Generate training data
        data_gen = DataGenerator(self.solver, self.lifting)
        start_time = time.time()
        
        self.training_data = data_gen.generate_training_data(
            parameter_sets=training_params,
            t_span=(0, t_end),
            t_eval=t_eval,
            g_func=default_input_function
        )
        
        generation_time = time.time() - start_time
        
        print(f"Data generation completed in {generation_time:.2f} seconds")
        print(f"Generated snapshots:")
        print(f"  Original (S): {self.training_data['S'].shape}")
        print(f"  Lifted (W): {self.training_data['W'].shape}")
        print(f"  Parameters (P): {self.training_data['P'].shape}")
        
        # Visualize a sample solution
        if self.config.get('visualization', {}).get('show_training_data', True):
            # Show first parameter set solution
            alpha, beta = training_params[0]
            sample_solution = self.solver.solve(
                t_span=(0, min(2.0, t_end)),  # Shorter for visualization
                t_eval=np.linspace(0, min(2.0, t_end), 101),
                alpha=alpha,
                beta=beta,
                g_func=default_input_function
            )
            
            self.visualizer.plot_solution_evolution(
                sample_solution, 
                f"Sample Training Data (α={alpha}, β={beta})"
            )
    
    def run_phase_2(self) -> None:
        """Phase 2: Lifting transformation (already done in Phase 1)."""
        print("\n" + "="*60)
        print("PHASE 2: Lifting Transformation")
        print("="*60)
        
        if self.training_data is None:
            raise ValueError("Training data not generated. Run Phase 1 first.")
        
        print("Lifting transformation analysis:")
        
        # Verify lifting transformation
        n_test = 5
        test_indices = np.random.choice(self.training_data['S'].shape[1], n_test, replace=False)
        
        lifting_errors = []
        for i in test_indices:
            original = self.training_data['S'][:, i]
            lifted = self.lifting.lift(original)
            recovered = self.lifting.unlift(lifted)
            error = np.linalg.norm(original - recovered) / np.linalg.norm(original)
            lifting_errors.append(error)
        
        print(f"Lifting transformation verification:")
        print(f"  Average recovery error: {np.mean(lifting_errors):.2e}")
        print(f"  Maximum recovery error: {np.max(lifting_errors):.2e}")
        
        # Analyze lifted data properties
        W = self.training_data['W']
        print(f"Lifted data analysis:")
        print(f"  Shape: {W.shape}")
        print(f"  Mean: {np.mean(W):.2e}")
        print(f"  Std: {np.std(W):.2e}")
        print(f"  Min: {np.min(W):.2e}")
        print(f"  Max: {np.max(W):.2e}")
        
        # Check quadratic relationship w3 = w1^2
        nx = self.solver.nx
        w1_samples = W[:nx, test_indices]
        w3_samples = W[2*nx:3*nx, test_indices]
        w1_squared = w1_samples**2
        
        quadratic_error = np.linalg.norm(w3_samples - w1_squared, 'fro') / np.linalg.norm(w1_squared, 'fro')
        print(f"  Quadratic constraint error (w₃ = w₁²): {quadratic_error:.2e}")
    
    def run_phase_3(self) -> None:
        """Phase 3: POD dimensionality reduction."""
        print("\n" + "="*60)
        print("PHASE 3: POD Dimensionality Reduction")
        print("="*60)
        
        if self.training_data is None:
            raise ValueError("Training data not generated. Run Phase 1 first.")
        
        # Compute POD basis
        W = self.training_data['W']
        r = self.config['pod']['r']
        
        print(f"Computing POD basis for r = {r}")
        start_time = time.time()
        
        self.pod_data = self.pod.compute_pod_basis(W, r=r)
        
        pod_time = time.time() - start_time
        
        print(f"POD completed in {pod_time:.2f} seconds")
        print(f"Selected modes: {self.pod_data['r']}")
        print(f"Energy captured: {self.pod_data['cumulative_energy'][r-1]*100:.2f}%")
        
        # Visualize POD results
        if self.config.get('visualization', {}).get('show_pod_analysis', True):
            self.pod.plot_energy_spectrum(max_modes=50)
            self.pod.plot_pod_modes(self.solver.x, n_modes=6)
        
        # Test projection error
        test_error = self.pod.compute_projection_error(W[:, ::10])  # Subsample for speed
        print(f"POD projection error: {test_error:.2e}")
    
    def run_phase_4(self) -> None:
        """Phase 4: Operator inference."""
        print("\n" + "="*60)
        print("PHASE 4: Operator Inference")
        print("="*60)
        
        if self.training_data is None or self.pod_data is None:
            raise ValueError("Training data and POD basis required. Run Phases 1-3 first.")
        
        # Project training data to reduced space
        W = self.training_data['W']
        W_reduced = self.pod.project(W)
        
        print(f"Projected data shape: {W_reduced.shape}")
        
        # Initialize operator inference
        self.opi = OperatorInference(self.solver, self.lifting, self.pod)
        
        # Learn operators
        print("Learning reduced operators...")
        start_time = time.time()
        
        regularization = self.config.get('operator_inference', {}).get('regularization')
        self.operators = self.opi.learn_operators(
            reduced_states=W_reduced,
            parameters=self.training_data['P'],
            g_func=default_input_function,
            regularization=regularization
        )
        
        learning_time = time.time() - start_time
        
        print(f"Operator learning completed in {learning_time:.2f} seconds")
        print(f"Learned operators:")
        print(f"  A (linear): {self.operators['A'].shape}")
        print(f"  H (quadratic): {self.operators['H'].shape}")
        print(f"  B (input): {self.operators['B'].shape}")
        
        # Analyze operator properties
        A = self.operators['A']
        H = self.operators['H']
        B = self.operators['B']
        
        print(f"Operator statistics:")
        print(f"  A norm: {np.linalg.norm(A, 'fro'):.2e}")
        print(f"  H norm: {np.linalg.norm(H, 'fro'):.2e}")
        print(f"  B norm: {np.linalg.norm(B, 'fro'):.2e}")
        
        # Check operator conditioning
        A_cond = np.linalg.cond(A)
        print(f"  A condition number: {A_cond:.2e}")
        
        if A_cond > 1e12:
            print("  Warning: A matrix is poorly conditioned!")
    
    def run_phase_5(self) -> None:
        """Phase 5: Validation and verification."""
        print("\n" + "="*60)
        print("PHASE 5: Validation and Verification")
        print("="*60)
        
        if self.opi is None:
            raise ValueError("Operators not learned. Run Phases 1-4 first.")
        
        # Initialize validation suite
        validator = ValidationSuite(self.solver, self.lifting, self.pod, self.opi)
        
        # Test parameters (different from training)
        test_params = self.config['validation']['test_parameters']
        
        # Time settings for validation
        t_end = self.config['time']['t_end']
        dt_test = self.config['validation'].get('dt', self.config['time']['dt'])
        t_eval = np.arange(0, t_end + dt_test, dt_test)
        
        print(f"Validation settings:")
        print(f"  Test parameters: {len(test_params)}")
        print(f"  Time span: [0, {t_end}]")
        print(f"  Time step: {dt_test}")
        
        # Single prediction validation
        print("\nSingle prediction validation:")
        alpha_test, beta_test = test_params[0]
        
        validation_result = validator.validate_single_prediction(
            alpha=alpha_test,
            beta=beta_test,
            t_span=(0, t_end),
            t_eval=t_eval,
            plot=self.config.get('visualization', {}).get('show_validation', True)
        )
        
        if validation_result['success']:
            print(f"  Validation successful: error = {validation_result['error']:.2e}")
        else:
            print(f"  Validation failed!")
            return
        
        # Parameter sweep validation
        if len(test_params) > 1:
            print("\nParameter sweep validation:")
            alpha_range = (0.8, 1.2)
            beta_range = (0.8, 1.2)
            n_test = min(10, len(test_params))
            
            sweep_results = validator.validate_parameter_sweep(
                alpha_range=alpha_range,
                beta_range=beta_range,
                n_test=n_test,
                t_span=(0, t_end),
                t_eval=t_eval
            )
            
            print(f"  Success rate: {sweep_results['success_rate']:.1%}")
            if len(sweep_results['valid_errors']) > 0:
                print(f"  Mean error: {np.mean(sweep_results['valid_errors']):.2e}")
                print(f"  Median error: {np.median(sweep_results['valid_errors']):.2e}")
        
        # Reduced dimension study (reproduce Figure 2)
        if self.config.get('validation', {}).get('dimension_study', False):
            print("\nReduced dimension study:")
            r_values = self.config['validation']['r_values']
            
            dimension_results = validator.validate_reduced_dimensions(
                r_values=r_values,
                training_params=self.config['training']['parameter_sets'][:3],
                test_params=test_params[:3],
                t_span=(0, t_end),
                t_eval=t_eval
            )
            
            if self.config.get('visualization', {}).get('show_dimension_study', True):
                validator.plot_error_vs_dimension(dimension_results)
    
    def run_extension(self) -> None:
        """Extension: Noise robustness analysis."""
        print("\n" + "="*60)
        print("EXTENSION: Noise Robustness Analysis")
        print("="*60)
        
        if self.training_data is None:
            raise ValueError("Training data required. Run Phase 1 first.")
        
        # Initialize noise analysis
        noise_analyzer = NoiseAnalysis(self.solver, self.lifting)
        
        # Configuration for noise analysis
        noise_config = self.config.get('noise_analysis', {})
        
        if not noise_config.get('enabled', False):
            print("Noise analysis disabled in configuration.")
            return
        
        # Noise analysis parameters
        training_params = self.config['training']['parameter_sets'][:3]  # Subset for speed
        test_params = self.config['validation']['test_parameters'][:2]
        
        t_end = min(2.0, self.config['time']['t_end'])  # Shorter for extension
        dt = self.config['time']['dt']
        t_eval = np.arange(0, t_end + dt, dt)
        
        r = self.config['pod']['r']
        noise_levels = noise_config.get('noise_levels', [0.01, 0.05, 0.1])
        
        # Regularization methods to test
        regularization_methods = [
            {'type': None, 'name': 'no_regularization'},
            {'type': 'ridge', 'lambda': 1e-4, 'name': 'ridge_1e-4'},
            {'type': 'ridge', 'lambda': 1e-3, 'name': 'ridge_1e-3'},
            {'type': 'lasso', 'lambda': 1e-4, 'name': 'lasso_1e-4'},
        ]
        
        print(f"Noise robustness analysis:")
        print(f"  Noise levels: {noise_levels}")
        print(f"  Regularization methods: {len(regularization_methods)}")
        print(f"  Reduced dimension: {r}")
        
        # Run noise robustness analysis
        noise_results = noise_analyzer.analyze_noise_robustness(
            training_params=training_params,
            test_params=test_params,
            t_span=(0, t_end),
            t_eval=t_eval,
            r=r,
            noise_levels=noise_levels,
            regularization_methods=regularization_methods,
            n_trials=3  # Reduced for computational efficiency
        )
        
        # Visualize results
        if self.config.get('visualization', {}).get('show_noise_analysis', True):
            noise_analyzer.plot_noise_analysis_results(noise_results)
        
        # Regularization comparison
        print("\nRegularization parameter comparison:")
        lambda_values = [1e-5, 1e-4, 1e-3, 1e-2]
        
        reg_comparison = noise_analyzer.compare_regularization_methods(
            training_params=training_params[:2],
            test_params=test_params[:1],
            t_span=(0, t_end),
            t_eval=t_eval,
            r=r,
            noise_level=0.05,  # Fixed noise level
            lambda_values=lambda_values
        )
        
        if self.config.get('visualization', {}).get('show_regularization_comparison', True):
            noise_analyzer.plot_regularization_comparison(reg_comparison)
    
    def generate_report(self) -> None:
        """Generate final project report."""
        print("\n" + "="*60)
        print("GENERATING PROJECT REPORT")
        print("="*60)
        
        report = {
            'project_title': 'FitzHugh-Nagumo Lift & Learn Implementation',
            'configuration': self.config,
            'results': {}
        }
        
        if self.training_data is not None:
            report['results']['training_data'] = {
                'original_snapshots': self.training_data['S'].shape,
                'lifted_snapshots': self.training_data['W'].shape,
                'parameter_sets': len(self.config['training']['parameter_sets'])
            }
        
        if self.pod_data is not None:
            report['results']['pod_analysis'] = {
                'reduced_dimension': self.pod_data['r'],
                'energy_captured': float(self.pod_data['cumulative_energy'][self.pod_data['r']-1]),
                'singular_values': self.pod_data['singular_values'][:10].tolist()
            }
        
        if self.operators is not None:
            report['results']['operator_inference'] = {
                'linear_operator_norm': float(np.linalg.norm(self.operators['A'], 'fro')),
                'quadratic_operator_norm': float(np.linalg.norm(self.operators['H'], 'fro')),
                'input_operator_norm': float(np.linalg.norm(self.operators['B'], 'fro'))
            }
        
        # Save report
        report_path = 'fitzhugh_nagumo_lift_learn_report.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"Report saved to: {report_path}")
        print("\nProject Summary:")
        print(f"  Implementation: Complete Lift & Learn methodology")
        print(f"  System: FitzHugh-Nagumo PDEs")
        print(f"  Approach: Non-intrusive operator inference")
        print(f"  Extension: Noise robustness analysis")
        print(f"  Spatial resolution: {self.config['spatial']['nx']} points")
        print(f"  Time horizon: {self.config['time']['t_end']} seconds")
        
        if self.pod_data is not None:
            print(f"  POD dimension: {self.pod_data['r']}")
            print(f"  Energy captured: {self.pod_data['cumulative_energy'][self.pod_data['r']-1]*100:.1f}%")


def load_config(config_path: str = None) -> Dict:
    """Load configuration from file or use defaults."""
    
    default_config = {
        'spatial': {
            'nx': 100,
            'L': 1.0
        },
        'system': {
            'gamma': 0.01
        },
        'time': {
            't_end': 4.0,
            'dt': 0.01
        },
        'training': {
            'parameter_sets': [
                (0.8, 0.8), (0.8, 1.0), (0.8, 1.2),
                (1.0, 0.8), (1.0, 1.0), (1.0, 1.2),
                (1.2, 0.8), (1.2, 1.0), (1.2, 1.2)
            ]
        },
        'pod': {
            'r': 10
        },
        'operator_inference': {
            'regularization': None
        },
        'validation': {
            'test_parameters': [(0.9, 0.9), (1.1, 1.1), (0.85, 1.15)],
            'dt': 0.02,
            'dimension_study': False,
            'r_values': [3, 6, 10, 14, 20]
        },
        'noise_analysis': {
            'enabled': False,
            'noise_levels': [0.01, 0.05, 0.1, 0.2]
        },
        'visualization': {
            'show_training_data': True,
            'show_pod_analysis': True,
            'show_validation': True,
            'show_dimension_study': True,
            'show_noise_analysis': True,
            'show_regularization_comparison': True
        }
    }
    
    if config_path and os.path.exists(config_path):
        with open(config_path, 'r') as f:
            user_config = json.load(f)
        # Merge configurations
        def merge_dicts(default, user):
            result = default.copy()
            for key, value in user.items():
                if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                    result[key] = merge_dicts(result[key], value)
                else:
                    result[key] = value
            return result
        
        return merge_dicts(default_config, user_config)
    
    return default_config


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description='FitzHugh-Nagumo Lift & Learn Implementation')
    parser.add_argument('--phase', type=str, choices=['1', '2', '3', '4', '5', 'extension', 'all'],
                       default='all', help='Phase to execute')
    parser.add_argument('--config', type=str, help='Configuration file path')
    parser.add_argument('--output-dir', type=str, default='.', help='Output directory')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Create output directory
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    # Change to output directory
    original_dir = os.getcwd()
    os.chdir(args.output_dir)
    
    try:
        # Initialize pipeline
        pipeline = LiftLearnPipeline(config)
        
        print("FitzHugh-Nagumo Lift & Learn Implementation")
        print("="*60)
        print(f"Execution mode: {args.phase}")
        print(f"Output directory: {args.output_dir}")
        
        # Execute requested phases
        if args.phase in ['1', 'all']:
            pipeline.run_phase_1()
        
        if args.phase in ['2', 'all']:
            pipeline.run_phase_2()
        
        if args.phase in ['3', 'all']:
            pipeline.run_phase_3()
        
        if args.phase in ['4', 'all']:
            pipeline.run_phase_4()
        
        if args.phase in ['5', 'all']:
            pipeline.run_phase_5()
        
        if args.phase in ['extension', 'all']:
            pipeline.run_extension()
        
        # Generate final report
        pipeline.generate_report()
        
        print("\n" + "="*60)
        print("EXECUTION COMPLETED SUCCESSFULLY")
        print("="*60)
        
    except Exception as e:
        print(f"\nError during execution: {e}")
        raise
    
    finally:
        # Return to original directory
        os.chdir(original_dir)


if __name__ == "__main__":
    main()