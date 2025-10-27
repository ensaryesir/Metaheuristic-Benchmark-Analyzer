# =============================================================================
# MAIN EXPERIMENT RUNNER - METAHEURISTIC ALGORITHM BENCHMARK
# =============================================================================
"""
Main execution script for comprehensive metaheuristic algorithm benchmarking.

This script:
1. Runs multiple metaheuristic algorithms on various benchmark functions
2. Collects performance data over multiple independent runs
3. Performs statistical analysis (Wilcoxon signed-rank test)
4. Generates visualization (convergence plots, boxplots)
5. Exports results to CSV and text files

Author: Ensar Yesir
Date: October 2025
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import wilcoxon
import os
import warnings
import time
from datetime import datetime
import importlib
import traceback

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Import all configuration parameters
from config import (
    POP_SIZE, MAX_ITER, DIMENSION, NUM_RUNS, RANDOM_SEED,
    ALGORITHMS, BENCHMARK_FUNCTIONS,
    RESULTS_DIR, CONVERGENCE_DIR, BOXPLOT_DIR,
    ALPHA, METRICS, CONVERGENCE_THRESHOLD,
    get_algorithm_params, get_benchmark_bounds,
    get_summary_file, get_stats_file,
    PLOT_SETTINGS, FIGURE_SIZE, DPI,
    print_experiment_info
)



# =============================================================================
# DYNAMIC MODULE LOADING
# =============================================================================

def load_algorithm_function(algo_config):
    """
    Dynamically load algorithm function from module
    
    Args:
        algo_config (dict): Algorithm configuration from config.py
    
    Returns:
        callable: Algorithm function
    """
    try:
        module_path = algo_config['module']
        function_name = algo_config['function']
        
        module = importlib.import_module(module_path)
        return getattr(module, function_name)
    except Exception as e:
        print(f"    ⚠️  Warning: Could not load algorithm: {e}")
        return None


def load_benchmark_function(bench_config):
    """
    Dynamically load benchmark function from module
    
    Args:
        bench_config (dict): Benchmark configuration from config.py
    
    Returns:
        callable: Benchmark function
    """
    try:
        module_path = bench_config['module']
        function_name = bench_config['function']
        
        module = importlib.import_module(module_path)
        return getattr(module, function_name)
    except Exception as e:
        print(f"    ⚠️  Warning: Could not load benchmark: {e}")
        return None


# =============================================================================
# EXPERIMENT EXECUTION
# =============================================================================

def run_experiment():
    """
    Main experiment execution function
    
    Runs all configured algorithms on all benchmark functions for NUM_RUNS times.
    Collects fitness scores, solutions, convergence curves, and computation times.
    
    Returns:
        tuple: (results dict, algorithm functions dict, benchmark functions dict)
    """
    
    # Create results directory structure
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(CONVERGENCE_DIR, exist_ok=True)
    os.makedirs(BOXPLOT_DIR, exist_ok=True)
    
    # Set random seed for reproducibility
    np.random.seed(RANDOM_SEED)
    
    # Print experiment information
    print_experiment_info()
    
    # Load algorithm and benchmark functions dynamically
    print("\n🔧 Loading Algorithms and Benchmarks...")
    print("-" * 70)
    
    algorithms = {}
    for alg_name, alg_config in ALGORITHMS.items():
        func = load_algorithm_function(alg_config)
        if func:
            algorithms[alg_name] = func
            print(f"  ✓ Loaded: {alg_name} - {alg_config['description']}")
        else:
            print(f"  ✗ Failed: {alg_name}")
    
    benchmark_functions = {}
    for bench_name, bench_config in BENCHMARK_FUNCTIONS.items():
        func = load_benchmark_function(bench_config)
        if func:
            benchmark_functions[bench_name] = func
            print(f"  ✓ Loaded: {bench_name.upper()} - {bench_config['description']}")
        else:
            print(f"  ✗ Failed: {bench_name}")
    
    if not algorithms:
        raise RuntimeError("❌ No algorithms loaded successfully!")
    if not benchmark_functions:
        raise RuntimeError("❌ No benchmark functions loaded successfully!")
    
    # Results storage structure
    results = {}
    
    # Main experiment loop
    print("\n" + "=" * 70)
    print("🚀 STARTING EXPERIMENTS")
    print("=" * 70)
    
    total_start_time = time.time()
    total_experiments = len(benchmark_functions) * len(algorithms) * NUM_RUNS
    completed_experiments = 0
    
    for func_name, func in benchmark_functions.items():
        print(f"\n{'='*70}")
        print(f"🔍 BENCHMARK: {func_name.upper()}")
        func_config = BENCHMARK_FUNCTIONS[func_name]
        print(f"   Description: {func_config['description']}")
        print(f"   Bounds: {func_config['bounds']}")
        print(f"   Global Minimum: {func_config['global_min']}")
        print(f"   Difficulty: {func_config['difficulty'].upper()}")
        print(f"{'='*70}")
        
        results[func_name] = {}
        func_bounds = func_config['bounds']
        
        for alg_name, alg_func in algorithms.items():
            print(f"\n  ⚙️  Algorithm: {alg_name}")
            print(f"     {ALGORITHMS[alg_name]['description']}")
            
            results[func_name][alg_name] = {
                'best_scores': [],
                'best_solutions': [],
                'convergence_curves': [],
                'computation_times': [],
                'iterations_to_converge': [],
                'success_count': 0
            }
            
            # Get algorithm-specific parameters
            alg_params = get_algorithm_params(alg_name)
            
            for run_num in range(NUM_RUNS):
                completed_experiments += 1
                progress = (completed_experiments / total_experiments) * 100
                
                start_time = time.time()
                
                print(f"     Run {run_num + 1:2d}/{NUM_RUNS} ", end="")
                
                try:
                    # Execute algorithm
                    best_solution, best_fitness, convergence_curve = alg_func(
                        objective_func=func,
                        dim=DIMENSION,
                        bounds=func_bounds,
                        pop_size=POP_SIZE,
                        max_iter=MAX_ITER,
                        **alg_params
                    )
                    
                    computation_time = time.time() - start_time
                    
                    # Store results
                    results[func_name][alg_name]['best_scores'].append(best_fitness)
                    results[func_name][alg_name]['best_solutions'].append(best_solution)
                    results[func_name][alg_name]['convergence_curves'].append(convergence_curve)
                    results[func_name][alg_name]['computation_times'].append(computation_time)
                    
                    # Check convergence iteration
                    converged_at = MAX_ITER
                    for i in range(1, len(convergence_curve)):
                        if abs(convergence_curve[i] - convergence_curve[i-1]) < CONVERGENCE_THRESHOLD:
                            converged_at = i
                            break
                    results[func_name][alg_name]['iterations_to_converge'].append(converged_at)
                    
                    # Check success (reached near-optimal solution)
                    if abs(best_fitness - func_config['global_min']) < 1e-6:
                        results[func_name][alg_name]['success_count'] += 1
                    
                    # Print result with color coding
                    status = "✓" if best_fitness < 1e3 else "⚠" if best_fitness < 1e6 else "✗"
                    print(f"[{status}] Fitness: {best_fitness:12.4e} | Time: {computation_time:6.2f}s | Progress: {progress:5.1f}%")
                    
                except Exception as e:
                    computation_time = time.time() - start_time
                    print(f"[✗] ERROR: {str(e)[:50]}... | Time: {computation_time:6.2f}s")
                    
                    # Store error values
                    results[func_name][alg_name]['best_scores'].append(float('inf'))
                    results[func_name][alg_name]['best_solutions'].append(None)
                    results[func_name][alg_name]['convergence_curves'].append([])
                    results[func_name][alg_name]['computation_times'].append(computation_time)
                    results[func_name][alg_name]['iterations_to_converge'].append(MAX_ITER)
    
    total_time = time.time() - total_start_time
    
    print("\n" + "=" * 70)
    print(f"🎉 ALL EXPERIMENTS COMPLETED!")
    print(f"   Total Time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
    print(f"   Experiments: {completed_experiments}")
    print(f"   Average Time per Run: {total_time/completed_experiments:.2f}s")
    print("=" * 70)
    
    return results, algorithms, benchmark_functions

# =============================================================================
# RESULTS ANALYSIS
# =============================================================================

def analyze_results(results, algorithms, benchmark_functions):
    """
    Analyze and report experimental results
    
    Creates performance summary tables with statistical metrics for each
    benchmark function and saves them to CSV files.
    
    Args:
        results (dict): Experimental results
        algorithms (dict): Loaded algorithm functions
        benchmark_functions (dict): Loaded benchmark functions
    """
    
    print("\n" + "=" * 70)
    print("📊 RESULTS ANALYSIS AND REPORTING")
    print("=" * 70)
    
    # Timing data collection
    timing_data = []
    
    # Analyze each benchmark function
    for func_name in benchmark_functions.keys():
        print(f"\n{'='*70}")
        print(f"📈 Performance Summary: {func_name.upper()}")
        print(f"{'='*70}")
        
        # Prepare summary data
        summary_data = []
        
        for alg_name in algorithms.keys():
            scores = results[func_name][alg_name]['best_scores']
            times = results[func_name][alg_name]['computation_times']
            convergence_iters = results[func_name][alg_name]['iterations_to_converge']
            success_count = results[func_name][alg_name]['success_count']
            
            # Filter valid scores (exclude inf values)
            valid_scores = [s for s in scores if s != float('inf')]
            valid_times = [t for i, t in enumerate(times) if scores[i] != float('inf')]
            valid_iters = [it for i, it in enumerate(convergence_iters) if scores[i] != float('inf')]
            
            if valid_scores:
                # Collect timing statistics
                timing_data.append({
                    'Function': func_name,
                    'Algorithm': alg_name,
                    'Avg_Time': np.mean(valid_times),
                    'Std_Time': np.std(valid_times),
                    'Min_Time': np.min(valid_times),
                    'Max_Time': np.max(valid_times),
                    'Total_Time': np.sum(valid_times)
                })
                
                # Create summary statistics
                summary_data.append({
                    'Algorithm': alg_name,
                    'Best': np.min(valid_scores),
                    'Worst': np.max(valid_scores),
                    'Mean': np.mean(valid_scores),
                    'Median': np.median(valid_scores),
                    'Std': np.std(valid_scores),
                    'Success_Rate': f"{(success_count / NUM_RUNS) * 100:.1f}%",
                    'Avg_Convergence_Iter': int(np.mean(valid_iters)),
                    'Avg_Time_s': np.mean(valid_times),
                    'Valid_Runs': f"{len(valid_scores)}/{NUM_RUNS}"
                })
            else:
                # All runs failed
                summary_data.append({
                    'Algorithm': alg_name,
                    'Best': 'FAILED',
                    'Worst': 'FAILED',
                    'Mean': 'FAILED',
                    'Median': 'FAILED',
                    'Std': 'FAILED',
                    'Success_Rate': '0.0%',
                    'Avg_Convergence_Iter': 'N/A',
                    'Avg_Time_s': 'N/A',
                    'Valid_Runs': f"0/{NUM_RUNS}"
                })
        
        # Create and display DataFrame
        df = pd.DataFrame(summary_data)
        print("\n" + df.to_string(index=False))
        
        # Save to CSV
        csv_filename = get_summary_file(func_name)
        df.to_csv(csv_filename, index=False, float_format='%.6e')
        print(f"\n💾 Summary saved: {csv_filename}")
    
    # Save timing analysis
    if timing_data:
        timing_df = pd.DataFrame(timing_data)
        timing_file = os.path.join(RESULTS_DIR, "computation_timing.csv")
        timing_df.to_csv(timing_file, index=False, float_format='%.4f')
        print(f"\n💾 Timing analysis saved: {timing_file}")
        
        # Print timing summary
        print(f"\n{'='*70}")
        print("⏱️  COMPUTATION TIME SUMMARY")
        print(f"{'='*70}")
        print("\n" + timing_df.to_string(index=False))
    
    print(f"\n{'='*70}")


# =============================================================================
# VISUALIZATION - CONVERGENCE PLOTS
# =============================================================================

def create_convergence_plots(results, algorithms, benchmark_functions):
    """
    Create convergence plots for all algorithms on each benchmark
    
    Args:
        results (dict): Experimental results
        algorithms (dict): Loaded algorithm functions
        benchmark_functions (dict): Loaded benchmark functions
    """
    
    print("\n" + "=" * 70)
    print("📈 VISUALIZATION - Convergence Plots")
    print("=" * 70)
    
    for func_name in benchmark_functions.keys():
        plt.figure(figsize=FIGURE_SIZE)
        
        print(f"\n  Plotting: {func_name.upper()}")
        
        for alg_name in algorithms.keys():
            convergence_curves = results[func_name][alg_name]['convergence_curves']
            
            # Filter valid curves
            valid_curves = [curve for curve in convergence_curves if len(curve) > 0]
            
            if valid_curves:
                # Calculate average convergence curve
                min_length = min(len(curve) for curve in valid_curves)
                truncated_curves = [curve[:min_length] for curve in valid_curves]
                avg_convergence = np.mean(truncated_curves, axis=0)
                
                # Get algorithm visualization settings
                alg_config = ALGORITHMS.get(alg_name, {})
                color = alg_config.get('color', None)
                marker = alg_config.get('marker', None)
                linestyle = alg_config.get('linestyle', '-')
                
                # Plot average convergence
                plt.plot(avg_convergence, label=alg_name, linewidth=2, 
                        color=color, marker=marker, markevery=max(1, min_length//10), 
                        linestyle=linestyle, markersize=6)
        
        # Configure plot
        plt.title(f'{func_name.upper()} Function - Convergence Curves', 
                 fontsize=PLOT_SETTINGS['title_fontsize'], fontweight='bold')
        plt.xlabel('Iteration', fontsize=PLOT_SETTINGS['label_fontsize'])
        plt.ylabel('Best Fitness (log scale)', fontsize=PLOT_SETTINGS['label_fontsize'])
        plt.legend(fontsize=PLOT_SETTINGS['legend_fontsize'], loc='upper right')
        plt.grid(True, alpha=PLOT_SETTINGS['grid_alpha'], linestyle='--')
        plt.yscale('log')
        plt.tight_layout()
        
        # Save plot
        plot_filename = os.path.join(CONVERGENCE_DIR, f"{func_name}_convergence.png")
        plt.savefig(plot_filename, dpi=DPI, bbox_inches='tight')
        plt.close()
        print(f"    💾 Saved: {plot_filename}")


# =============================================================================
# VISUALIZATION - BOXPLOTS
# =============================================================================

def create_boxplots(results, algorithms, benchmark_functions):
    """
    Create boxplots showing performance distribution for each algorithm
    
    Args:
        results (dict): Experimental results
        algorithms (dict): Loaded algorithm functions
        benchmark_functions (dict): Loaded benchmark functions
    """
    
    print("\n" + "=" * 70)
    print("📊 VISUALIZATION - Box Plots")
    print("=" * 70)
    
    for func_name in benchmark_functions.keys():
        plt.figure(figsize=FIGURE_SIZE)
        
        print(f"\n  Plotting: {func_name.upper()}")
        
        # Prepare data
        data = []
        labels = []
        colors = []
        
        for alg_name in algorithms.keys():
            scores = results[func_name][alg_name]['best_scores']
            valid_scores = [s for s in scores if s != float('inf')]
            
            if valid_scores:
                data.append(valid_scores)
                labels.append(alg_name)
                colors.append(ALGORITHMS[alg_name].get('color', '#1f77b4'))
        
        # Create boxplot
        bp = plt.boxplot(data, labels=labels, patch_artist=True, 
                        showmeans=True, meanline=True)
        
        # Color the boxes
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        
        # Configure plot
        plt.title(f'{func_name.upper()} Function - Performance Distribution', 
                 fontsize=PLOT_SETTINGS['title_fontsize'], fontweight='bold')
        plt.xlabel('Algorithm', fontsize=PLOT_SETTINGS['label_fontsize'])
        plt.ylabel('Best Fitness (log scale)', fontsize=PLOT_SETTINGS['label_fontsize'])
        plt.xticks(rotation=45, ha='right')
        plt.grid(True, alpha=PLOT_SETTINGS['grid_alpha'], axis='y', linestyle='--')
        plt.yscale('log')
        plt.tight_layout()
        
        # Save plot
        plot_filename = os.path.join(BOXPLOT_DIR, f"{func_name}_boxplot.png")
        plt.savefig(plot_filename, dpi=DPI, bbox_inches='tight')
        plt.close()
        print(f"    💾 Saved: {plot_filename}")


# =============================================================================
# STATISTICAL ANALYSIS - WILCOXON TEST
# =============================================================================

def statistical_analysis(results, algorithms, benchmark_functions):
    """
    Perform statistical significance testing using Wilcoxon signed-rank test
    
    Compares all algorithms against the champion (best mean performer) for
    each benchmark function.
    
    Args:
        results (dict): Experimental results
        algorithms (dict): Loaded algorithm functions
        benchmark_functions (dict): Loaded benchmark functions
    """
    
    print("\n" + "=" * 70)
    print("� STATISTICAL ANALYSIS - Wilcoxon Signed-Rank Test")
    print("=" * 70)
    
    for func_name in benchmark_functions.keys():
        print(f"\n{'='*70}")
        print(f"Statistical Test Results: {func_name.upper()}")
        print(f"{'='*70}")
        
        # Find champion algorithm (best mean score)
        mean_scores = {}
        for alg_name in algorithms.keys():
            scores = results[func_name][alg_name]['best_scores']
            valid_scores = [s for s in scores if s != float('inf')]
            if valid_scores:
                mean_scores[alg_name] = np.mean(valid_scores)
            else:
                mean_scores[alg_name] = float('inf')
        
        if not mean_scores:
            print("  ⚠️  No valid results for statistical analysis")
            continue
        
        champion = min(mean_scores, key=mean_scores.get)
        champion_score = mean_scores[champion]
        
        print(f"\n🏆 Champion Algorithm: {champion}")
        print(f"   Mean Fitness: {champion_score:.6e}")
        print(f"\n{'-'*70}")
        print(f"{'Algorithm':<15} {'Mean Fitness':<15} {'p-value':<12} {'Significant?'}")
        print(f"{'-'*70}")
        
        # Perform Wilcoxon tests
        champion_scores = [s for s in results[func_name][champion]['best_scores'] 
                          if s != float('inf')]
        test_results = []
        
        for alg_name in sorted(algorithms.keys()):
            mean_fitness = mean_scores[alg_name]
            mean_str = f"{mean_fitness:.4e}" if mean_fitness != float('inf') else "FAILED"
            
            if alg_name == champion:
                print(f"{alg_name:<15} {mean_str:<15} {'---':<12} {'CHAMPION'}")
                test_results.append(f"{alg_name}: CHAMPION (Mean: {mean_str})")
            else:
                other_scores = [s for s in results[func_name][alg_name]['best_scores'] 
                               if s != float('inf')]
                
                if champion_scores and other_scores and len(champion_scores) == len(other_scores):
                    try:
                        # Check if scores are identical
                        if np.array_equal(champion_scores, other_scores):
                            print(f"{alg_name:<15} {mean_str:<15} {'---':<12} {'IDENTICAL'}")
                            test_results.append(f"{alg_name}: Identical to champion")
                        else:
                            statistic, p_value = wilcoxon(champion_scores, other_scores)
                            significant = "✅ YES" if p_value < ALPHA else "❌ NO"
                            print(f"{alg_name:<15} {mean_str:<15} {p_value:<12.6f} {significant}")
                            test_results.append(
                                f"{alg_name}: p={p_value:.6f}, Significant={significant}, Mean={mean_str}"
                            )
                    except Exception as e:
                        print(f"{alg_name:<15} {mean_str:<15} {'ERROR':<12} {str(e)[:20]}")
                        test_results.append(f"{alg_name}: Test failed - {str(e)}")
                else:
                    reason = "INCOMPLETE" if other_scores else "NO_DATA"
                    print(f"{alg_name:<15} {mean_str:<15} {'---':<12} {reason}")
                    test_results.append(f"{alg_name}: {reason}")
        
        print(f"{'-'*70}")
        print(f"Significance level: α = {ALPHA}")
        print(f"p < {ALPHA} indicates statistically significant difference from champion")
        
        # Save statistical results to file
        stats_filename = get_stats_file(func_name)
        with open(stats_filename, 'w', encoding='utf-8') as f:
            f.write("=" * 70 + "\n")
            f.write(f"STATISTICAL ANALYSIS: {func_name.upper()} FUNCTION\n")
            f.write("=" * 70 + "\n\n")
            f.write(f"Test: Wilcoxon Signed-Rank Test\n")
            f.write(f"Significance Level: α = {ALPHA}\n")
            f.write(f"Number of Runs: {NUM_RUNS}\n\n")
            f.write("-" * 70 + "\n")
            f.write(f"CHAMPION ALGORITHM: {champion}\n")
            f.write(f"Champion Mean Fitness: {champion_score:.6e}\n")
            f.write("-" * 70 + "\n\n")
            f.write("DETAILED RESULTS:\n")
            f.write("-" * 70 + "\n\n")
            for result in test_results:
                f.write(result + "\n")
            f.write("\n" + "-" * 70 + "\n")
            f.write("\nINTERPRETATION:\n")
            f.write("- p < 0.05: Statistically significant difference (✅ YES)\n")
            f.write("- p ≥ 0.05: No significant difference (❌ NO)\n")
            f.write("- Lower p-value = stronger evidence of difference\n")
            f.write("- Champion has the best (lowest) mean fitness\n")
            f.write("=" * 70 + "\n")
        
        print(f"\n💾 Statistical results saved: {stats_filename}")


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """
    Main execution function
    
    Orchestrates the entire benchmarking process:
    1. Runs experiments on all algorithm-benchmark combinations
    2. Analyzes results and computes statistics
    3. Generates visualizations (convergence and boxplots)
    4. Performs statistical significance testing
    5. Exports all results to files
    """
    
    print("\n" + "=" * 70)
    print(" " * 15 + "METAHEURISTIC ALGORITHM BENCHMARK")
    print(" " * 20 + "Comprehensive Performance Analysis")
    print("=" * 70)
    print(f"\nStart Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        # Run experiments
        print("\n" + "🔬" * 35)
        results, algorithms, benchmark_functions = run_experiment()
        
        # Analyze results
        print("\n" + "📊" * 35)
        analyze_results(results, algorithms, benchmark_functions)
        
        # Create visualizations
        print("\n" + "📈" * 35)
        create_convergence_plots(results, algorithms, benchmark_functions)
        create_boxplots(results, algorithms, benchmark_functions)
        
        # Statistical analysis
        print("\n" + "🔬" * 35)
        statistical_analysis(results, algorithms, benchmark_functions)
        
        # Final summary
        print("\n" + "=" * 70)
        print("✅ ALL ANALYSES COMPLETED SUCCESSFULLY!")
        print("=" * 70)
        print(f"\n📁 Results Location: {RESULTS_DIR}")
        print(f"   • Summary Tables:     {RESULTS_DIR}/performance_summary_*.csv")
        print(f"   • Statistical Tests:  {RESULTS_DIR}/statistical_analysis_*.txt")
        print(f"   • Timing Analysis:    {RESULTS_DIR}/computation_timing.csv")
        print(f"   • Convergence Plots:  {CONVERGENCE_DIR}/")
        print(f"   • Box Plots:          {BOXPLOT_DIR}/")
        print("\n" + "=" * 70)
        print(f"End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 70 + "\n")
        
    except Exception as e:
        print("\n" + "=" * 70)
        print("❌ CRITICAL ERROR OCCURRED")
        print("=" * 70)
        print(f"\nError: {str(e)}")
        print("\nStack Trace:")
        print("-" * 70)
        traceback.print_exc()
        print("-" * 70)
        print("\nPlease check:")
        print("  1. All algorithm files are present in algorithms/ directory")
        print("  2. All benchmark files are present in benchmarks/ directory")
        print("  3. config.py is properly configured")
        print("  4. Required packages are installed (numpy, scipy, pandas, matplotlib)")
        print("=" * 70 + "\n")
        raise


if __name__ == "__main__":
    # Set matplotlib backend for non-interactive plotting
    plt.switch_backend('Agg')
    
    # Run main function
    main()
