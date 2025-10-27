# =============================================================================
# CENTRAL CONFIGURATION FILE - ALL EXPERIMENT PARAMETERS
# =============================================================================
"""
Configuration file for Metaheuristic Benchmark Analyzer

This file contains all parameters, settings, and configurations for
running comprehensive benchmarking experiments on metaheuristic algorithms.

All algorithm parameters follow the ORIGINAL PAPER recommendations unless
explicitly noted as modified.
"""

import numpy as np
import os
from datetime import datetime

# =============================================================================
# BASIC EXPERIMENT SETTINGS
# =============================================================================

# Core experiment parameters
POP_SIZE = 30          # Population size (standard for all algorithms)
MAX_ITER = 1000        # Maximum iterations (standard benchmark setting)
DIMENSION = 30         # Problem dimensionality (standard high-dimensional test)
NUM_RUNS = 30          # Independent runs per algorithm (for statistical significance)
RANDOM_SEED = 42       # Seed for reproducibility

# Note: NUM_RUNS = 30 is standard in metaheuristic research for statistical analysis

# =============================================================================
# ALGORITHM-SPECIFIC PARAMETERS (FOLLOWING ORIGINAL PAPERS)
# =============================================================================

# Particle Swarm Optimization (PSO) Parameters
# Reference: Shi & Eberhart (1998) - "A modified particle swarm optimizer"
PSO_PARAMS = {
    'w': 0.9,           # Initial inertia weight (linearly decreases to w_min)
    'w_min': 0.4,       # Minimum inertia weight
    'c1': 2.0,          # Cognitive coefficient (personal best influence)
    'c2': 2.0,          # Social coefficient (global best influence)
    'v_max_factor': 0.5 # Velocity limit as fraction of bounds
}

# Differential Evolution (DE) Parameters
# Reference: Storn & Price (1997) - DE/rand/1/bin strategy
DE_PARAMS = {
    'F': 0.5,           # Mutation factor (scaling factor)
    'CR': 0.9           # Crossover probability
}

# Marine Predators Algorithm (MPA) Parameters
# Reference: Faramarzi et al. (2020)
MPA_PARAMS = {
    'P': 0.5,           # Constant parameter
    'FADs': 0.2         # Fish Aggregating Devices effect
}

# Artificial Gorilla Troops Optimizer (GTO) Parameters
# Reference: Abdollahzadeh et al. (2021)
GTO_PARAMS = {
    'beta': 3.0,        # Exploration intensity parameter
    'p': 0.03           # Probability of random exploration
}

# Enhanced RIME Optimization Algorithm Parameters
# NOTE: This is an IMPROVED version with author's contributions
# Original RIME: Su et al. (2023) - "RIME: A physics-based optimization"
# Enhancements: Social interaction, adaptive noise, periodic restart
RIME_PARAMS = {
    'social_rate': 0.5,     # Social interaction probability (NEW)
    'noise_factor': 0.1,    # Noise injection strength (NEW)
    'reset_interval': 100   # Diversity injection frequency (NEW)
}

# Super RIME Parameters (Multi-Strategy Version)
# Advanced version with multiple complementary strategies
# Keeps original RIME formulas + adds PSO/DE/GTO-inspired strategies
SUPER_RIME_PARAMS = {
    'F': 0.5               # Differential weight for DE-style mutation
}

# =============================================================================
# ALGORITHMS TO RUN
# =============================================================================

ALGORITHMS = {
    'PSO': {
        'function': 'pso',
        'module': 'algorithms.pso',
        'params': PSO_PARAMS,
        'color': '#1f77b4',      # Blue
        'marker': 'o',
        'linestyle': '-',
        'description': 'Particle Swarm Optimization (Shi & Eberhart, 1998)'
    },
    'DE': {
        'function': 'de',
        'module': 'algorithms.de',
        'params': DE_PARAMS,
        'color': '#ff7f0e',      # Orange
        'marker': 's',
        'linestyle': '--',
        'description': 'Differential Evolution (Storn & Price, 1997)'
    },
    'MPA': {
        'function': 'mpa',
        'module': 'algorithms.mpa',
        'params': MPA_PARAMS,
        'color': '#2ca02c',      # Green
        'marker': '^',
        'linestyle': '-.',
        'description': 'Marine Predators Algorithm (Faramarzi et al., 2020)'
    },
    'GTO': {
        'function': 'gto',
        'module': 'algorithms.gto',
        'params': GTO_PARAMS,
        'color': '#d62728',      # Red
        'marker': 'D',
        'linestyle': ':',
        'description': 'Gorilla Troops Optimizer (Abdollahzadeh et al., 2021)'
    },
    'RIME': {
        'function': 'rime_enhanced',  # Enhanced version
        'module': 'algorithms.rime',
        'params': RIME_PARAMS,
        'color': '#9467bd',      # Purple
        'marker': 'v',
        'linestyle': '-',
        'description': 'Enhanced RIME (Su et al., 2023 + Improvements)'
    },
    'Super-RIME': {
        'function': 'rime_super',     # Multi-strategy version
        'module': 'algorithms.rime_super',
        'params': SUPER_RIME_PARAMS,
        'color': '#8c564b',      # Brown
        'marker': 'p',
        'linestyle': '-.',
        'description': 'Super RIME (Original RIME + Multi-Strategy Layer)'
    }
}

# =============================================================================
# BENCHMARK FUNCTIONS
# =============================================================================

BENCHMARK_FUNCTIONS = {
    'sphere': {
        'function': 'sphere',
        'module': 'benchmarks.sphere',
        'global_min': 0.0,
        'optimal_solution': np.zeros(DIMENSION),
        'bounds': (-100, 100),        # Standard bounds for Sphere
        'difficulty': 'unimodal',
        'description': 'Sphere Function - Convex, Unimodal, Separable'
    },
    'rastrigin': {
        'function': 'rastrigin',
        'module': 'benchmarks.rastrigin',
        'global_min': 0.0,
        'optimal_solution': np.zeros(DIMENSION),
        'bounds': (-5.12, 5.12),      # Standard bounds for Rastrigin
        'difficulty': 'highly_multimodal',
        'description': 'Rastrigin Function - Highly Multimodal, Many Local Optima'
    },
    'ackley': {
        'function': 'ackley',
        'module': 'benchmarks.ackley',
        'global_min': 0.0,
        'optimal_solution': np.zeros(DIMENSION),
        'bounds': (-32.768, 32.768),  # Standard bounds for Ackley
        'difficulty': 'multimodal',
        'description': 'Ackley Function - Multimodal, Nearly Flat Outer Region'
    },
    'rosenbrock': {
        'function': 'rosenbrock',
        'module': 'benchmarks.rosenbrock',
        'global_min': 0.0,
        'optimal_solution': np.ones(DIMENSION),
        'bounds': (-5, 10),           # Standard bounds for Rosenbrock
        'difficulty': 'valley_shaped',
        'description': 'Rosenbrock Function - Valley-Shaped, Global Optimum in Narrow Valley'
    },
    'schwefel': {
        'function': 'schwefel',
        'module': 'benchmarks.schwefel',
        'global_min': 0.0,
        'optimal_solution': np.full(DIMENSION, 420.9687),
        'bounds': (-500, 500),        # Standard bounds for Schwefel
        'difficulty': 'deceptive',
        'description': 'Schwefel Function - Deceptive, Global Optimum Far from Second Best'
    }
}

# =============================================================================
# FILE AND DIRECTORY SETTINGS
# =============================================================================

# Dynamic result folder creation with timestamp
EXPERIMENT_TIMESTAMP = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
RESULTS_DIR = os.path.join("results", f"experiment_{EXPERIMENT_TIMESTAMP}")
os.makedirs(RESULTS_DIR, exist_ok=True)

# Output file paths
CONVERGENCE_DIR = os.path.join(RESULTS_DIR, "convergence_plots")
BOXPLOT_DIR = os.path.join(RESULTS_DIR, "boxplots")
TIMING_FILE = os.path.join(RESULTS_DIR, "computation_timing.csv")

# Create subdirectories
os.makedirs(CONVERGENCE_DIR, exist_ok=True)
os.makedirs(BOXPLOT_DIR, exist_ok=True)

# Function-specific output files (created dynamically)
def get_summary_file(function_name):
    """Get summary CSV file path for specific benchmark function"""
    return os.path.join(RESULTS_DIR, f"performance_summary_{function_name}.csv")

def get_stats_file(function_name):
    """Get statistical analysis file path for specific benchmark function"""
    return os.path.join(RESULTS_DIR, f"statistical_analysis_{function_name}.txt")

# =============================================================================
# PLOT SETTINGS
# =============================================================================

# Individual plot parameters (for easy access)
FIGURE_SIZE = (12, 8)           # Figure dimensions (width, height) in inches
DPI = 300                       # Resolution for saved figures

# Comprehensive plot settings dictionary
PLOT_SETTINGS = {
    'figure_size': FIGURE_SIZE,
    'dpi': DPI,
    'font_size': 12,
    'title_fontsize': 14,
    'label_fontsize': 12,
    'legend_fontsize': 10,
    'tick_size': 10,
    'line_width': 2.0,
    'grid_alpha': 0.3,
    'save_format': 'png'  # Can be 'png', 'pdf', 'svg'
}

# =============================================================================
# STATISTICAL ANALYSIS SETTINGS
# =============================================================================

# Statistical test parameters
ALPHA = 0.05                    # Significance level for hypothesis testing
CONFIDENCE_LEVEL = 0.95         # Confidence level for intervals

# Performance metrics to track
METRICS = [
    'best_fitness',             # Best fitness found
    'mean_fitness',             # Mean fitness across runs
    'std_fitness',              # Standard deviation
    'median_fitness',           # Median fitness
    'worst_fitness',            # Worst fitness
    'success_rate',             # Percentage of successful runs
    'convergence_iteration',    # Iteration where convergence occurred
    'computation_time'          # Execution time in seconds
]

# Convergence criteria
CONVERGENCE_THRESHOLD = 1e-8    # Fitness improvement threshold
STAGNATION_GENERATIONS = 50     # Generations without improvement = stagnation
MAX_TIME_SECONDS = 7200         # Maximum runtime (2 hours per run)

# Success criteria (for success_rate metric)
SUCCESS_THRESHOLD = 1e-6        # Error from global optimum to be considered "success"

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def get_algorithm_params(algorithm_name):
    """
    Get parameters for specific algorithm
    
    Args:
        algorithm_name (str): Name of algorithm ('PSO', 'DE', etc.)
    
    Returns:
        dict: Algorithm parameters
    """
    if algorithm_name not in ALGORITHMS:
        raise ValueError(f"Unknown algorithm: {algorithm_name}")
    return ALGORITHMS[algorithm_name]['params'].copy()


def get_benchmark_bounds(function_name):
    """
    Get search space bounds for specific benchmark function
    
    Args:
        function_name (str): Name of benchmark function
    
    Returns:
        tuple: (lower_bound, upper_bound)
    """
    if function_name not in BENCHMARK_FUNCTIONS:
        raise ValueError(f"Unknown benchmark function: {function_name}")
    return BENCHMARK_FUNCTIONS[function_name]['bounds']


def get_results_directory():
    """
    Get the results directory path
    
    Returns:
        str: Path to results directory
    """
    return RESULTS_DIR


def get_algorithm_list():
    """
    Get list of algorithm names
    
    Returns:
        list: Algorithm names
    """
    return list(ALGORITHMS.keys())


def get_benchmark_list():
    """
    Get list of benchmark function names
    
    Returns:
        list: Benchmark function names
    """
    return list(BENCHMARK_FUNCTIONS.keys())


def validate_config():
    """
    Validate configuration settings
    
    Raises:
        AssertionError: If any configuration is invalid
    """
    # Validate basic parameters
    assert POP_SIZE > 0, "POP_SIZE must be positive"
    assert MAX_ITER > 0, "MAX_ITER must be positive"
    assert DIMENSION > 0, "DIMENSION must be positive"
    assert NUM_RUNS > 0, "NUM_RUNS must be positive"
    assert 0 < ALPHA < 1, "ALPHA must be between 0 and 1"
    
    # Validate algorithms
    for algo_name, algo_config in ALGORITHMS.items():
        assert 'function' in algo_config, f"{algo_name}: 'function' not defined"
        assert 'module' in algo_config, f"{algo_name}: 'module' not defined"
        assert 'params' in algo_config, f"{algo_name}: 'params' not defined"
        assert 'color' in algo_config, f"{algo_name}: 'color' not defined"
    
    # Validate benchmark functions
    for func_name, func_config in BENCHMARK_FUNCTIONS.items():
        assert 'function' in func_config, f"{func_name}: 'function' not defined"
        assert 'bounds' in func_config, f"{func_name}: 'bounds' not defined"
        assert 'global_min' in func_config, f"{func_name}: 'global_min' not defined"
        
        bounds = func_config['bounds']
        assert bounds[0] < bounds[1], f"{func_name}: Invalid bounds {bounds}"
    
    print("=" * 70)
    print("✓ Configuration validated successfully")
    print("=" * 70)
    print(f"  Algorithms: {len(ALGORITHMS)} ({', '.join(ALGORITHMS.keys())})")
    print(f"  Benchmarks: {len(BENCHMARK_FUNCTIONS)} ({', '.join(BENCHMARK_FUNCTIONS.keys())})")
    print(f"  Dimension: {DIMENSION}D")
    print(f"  Population: {POP_SIZE}")
    print(f"  Iterations: {MAX_ITER}")
    print(f"  Runs: {NUM_RUNS}")
    print(f"  Results Directory: {RESULTS_DIR}")
    print("=" * 70)


def print_experiment_info():
    """Print detailed experiment information"""
    print("\n" + "=" * 70)
    print(" " * 20 + "EXPERIMENT CONFIGURATION")
    print("=" * 70)
    
    print("\n📊 EXPERIMENT PARAMETERS:")
    print(f"  • Dimension:          {DIMENSION}D")
    print(f"  • Population Size:    {POP_SIZE}")
    print(f"  • Max Iterations:     {MAX_ITER}")
    print(f"  • Independent Runs:   {NUM_RUNS}")
    print(f"  • Random Seed:        {RANDOM_SEED}")
    
    print("\n🔬 ALGORITHMS TO TEST:")
    for name, config in ALGORITHMS.items():
        print(f"  • {name:<6} - {config['description']}")
    
    print("\n📈 BENCHMARK FUNCTIONS:")
    for name, config in BENCHMARK_FUNCTIONS.items():
        bounds = config['bounds']
        print(f"  • {name:<12} - {config['description']}")
        print(f"    {'':12}   Bounds: {bounds}, Difficulty: {config['difficulty']}")
    
    print("\n📁 OUTPUT:")
    print(f"  • Results Directory:  {RESULTS_DIR}")
    print(f"  • Convergence Plots:  {CONVERGENCE_DIR}")
    print(f"  • Box Plots:          {BOXPLOT_DIR}")
    
    print("\n" + "=" * 70)


# Auto-validate when imported (except when running as main)
if __name__ != "__main__":
    validate_config()
