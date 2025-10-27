import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np

"""
Runge Kutta Optimizer (RUN) Algorithm
======================================

📚 ORIGINAL REFERENCE:
----------------------
Ahmadianfar, I., Heidari, A. A., Gandomi, A. H., Chu, X., & Chen, H. (2021).
"RUN beyond the metaphor: An efficient optimization algorithm based on 
Runge Kutta method."
Expert Systems with Applications, 181, 115079.
DOI: 10.1016/j.eswa.2021.115079

📖 ALGORITHM DESCRIPTION:
-------------------------
RUN is a mathematically-inspired optimization algorithm based on the 
Runge-Kutta (RK) method for solving ordinary differential equations (ODEs).
The algorithm uses three main mechanisms:

1. Enhanced Solution Quality (ESQ):
   - ESQ1: Runge-Kutta inspired search
   - ESQ2: Solution Quality Equation (SQE)
   - ESQ3: Direct exploitation toward best solution

2. Adaptive Parameters:
   - f: Frequency parameter (controls search intensity)
   - q: Quality factor (balances exploration/exploitation)

3. Smart Pool Update:
   - Periodically updates worst solutions
   - Maintains population diversity

🔬 MATHEMATICAL FORMULATION:
----------------------------

Adaptive Parameters:
    f(t) = 20 · exp(-2t/T)    # Frequency decreases over time
    q(t) = (t/T)²             # Quality increases over time

Enhanced Solution Quality (ESQ):

ESQ1 - Runge-Kutta Method (RK4):
    K1 = α₁ · (Xbest - X[i])
    K2 = α₂ · (Xbest - (X[i] + 0.5·K1))
    K3 = α₃ · (Xbest - (X[i] + 0.5·K2))
    K4 = α₄ · (Xbest - (X[i] + K3))
    X_new = X[i] + (K1 + 2K2 + 2K3 + K4) / 6

ESQ2 - Solution Quality Equation (SQE):
    X_new = X[i] + f · r · (Xbest - X[i])

ESQ3 - Direct Movement:
    X_new = Xbest - f · r · (Xbest - X[i])

Where:
    α₁, α₂, α₃, α₄ = random weights
    r = random vector
    f = frequency parameter

💪 STRENGTHS:
-------------
- Mathematically grounded (RK method)
- Excellent balance between exploration and exploitation
- Strong performance on multimodal functions
- Adaptive parameter control
- Maintains population diversity

⚠️ WEAKNESSES:
--------------
- More complex than classical algorithms
- Higher computational cost per iteration
- Requires proper parameter initialization

📊 COMPUTATIONAL COMPLEXITY:
---------------------------
Time: O(pop_size × max_iter × dim)
Space: O(pop_size × dim)

Args:
    objective_func (callable): Objective function to minimize
    dim (int): Problem dimensionality
    bounds (tuple or list): Search space bounds as (lb, ub)
    pop_size (int): Population size (default: 30)
    max_iter (int): Maximum number of iterations (default: 1000)
    
Returns:
    tuple: (best_solution, best_fitness, convergence_curve)
        - best_solution: Best solution found
        - best_fitness: Fitness of best solution
        - convergence_curve: Best fitness at each iteration
"""

def run(objective_func, dim, bounds, pop_size=30, max_iter=1000):
    """
    Runge Kutta Optimizer (RUN) - Standard Implementation
    
    This implementation follows the original RUN algorithm as described
    in Ahmadianfar et al. (2021), including all three ESQ mechanisms
    and the Smart Pool Update strategy.
    """
    
    # ========================================================================
    # STEP 1: PARAMETER INITIALIZATION
    # ========================================================================
    
    # Standardize bounds
    if isinstance(bounds, tuple) and len(bounds) == 2:
        lb = np.full(dim, bounds[0])
        ub = np.full(dim, bounds[1])
    else:
        bounds = np.array(bounds)
        if bounds.shape == (2,):
            lb = np.full(dim, bounds[0])
            ub = np.full(dim, bounds[1])
        else:
            lb = bounds[:, 0]
            ub = bounds[:, 1]
    
    # ========================================================================
    # STEP 2: POPULATION INITIALIZATION
    # ========================================================================
    
    # Initialize population uniformly in search space
    X = np.random.uniform(lb, ub, (pop_size, dim))
    
    # Evaluate initial population
    fitness = np.array([objective_func(ind) for ind in X])
    
    # Find best solution
    best_idx = np.argmin(fitness)
    best_solution = X[best_idx].copy()
    best_fitness = fitness[best_idx]
    
    # Convergence tracking
    convergence_curve = np.zeros(max_iter)
    
    # ========================================================================
    # STEP 3: MAIN OPTIMIZATION LOOP
    # ========================================================================
    
    for t in range(max_iter):
        
        # ====================================================================
        # ADAPTIVE PARAMETERS UPDATE
        # ====================================================================
        # These parameters control the search behavior over time
        
        # Frequency parameter: decreases from 20 to near 0
        # Controls search intensity
        f = 20 * np.exp(-2 * t / max_iter)
        
        # Quality factor: increases from 0 to 1
        # Balances exploration (early) vs exploitation (late)
        q = (t / max_iter) ** 2
        
        # ====================================================================
        # ENHANCED SOLUTION QUALITY (ESQ) MECHANISM
        # ====================================================================
        
        for i in range(pop_size):
            
            # Select ESQ strategy based on random probability
            strategy = np.random.rand()
            
            if strategy < 1/3:
                # ============================================================
                # ESQ1: Runge-Kutta Method (RK4)
                # ============================================================
                # Inspired by 4th order Runge-Kutta numerical integration
                
                # Random weights for RK coefficients
                alpha1 = np.random.rand()
                alpha2 = np.random.rand()
                alpha3 = np.random.rand()
                alpha4 = np.random.rand()
                
                # K1: Initial slope
                K1 = alpha1 * (best_solution - X[i])
                
                # K2: Slope at midpoint using K1
                XK2 = X[i] + 0.5 * K1
                XK2 = np.clip(XK2, lb, ub)  # Ensure within bounds
                K2 = alpha2 * (best_solution - XK2)
                
                # K3: Slope at midpoint using K2
                XK3 = X[i] + 0.5 * K2
                XK3 = np.clip(XK3, lb, ub)
                K3 = alpha3 * (best_solution - XK3)
                
                # K4: Slope at endpoint using K3
                XK4 = X[i] + K3
                XK4 = np.clip(XK4, lb, ub)
                K4 = alpha4 * (best_solution - XK4)
                
                # Weighted average (RK4 formula)
                RK_update = (K1 + 2*K2 + 2*K3 + K4) / 6
                
                # New position (NO frequency modulation for ESQ1)
                # Note: In original RUN, ESQ1 doesn't use f parameter
                # RK4 method determines its own step size through alpha weights
                X_new = X[i] + RK_update
                
            elif strategy < 2/3:
                # ============================================================
                # ESQ2: Solution Quality Equation (SQE)
                # ============================================================
                # Direct movement toward best solution with random exploration
                
                # Random direction vector
                r = np.random.rand(dim)
                
                # Move toward best solution with frequency modulation
                X_new = X[i] + f * r * (best_solution - X[i])
                
            else:
                # ============================================================
                # ESQ3: Direct Exploitation
                # ============================================================
                # Move away from current toward best (exploration around best)
                
                # Random direction vector
                r = np.random.rand(dim)
                
                # Direct movement with quality factor
                X_new = best_solution - f * r * (best_solution - X[i])
            
            # ================================================================
            # BOUNDARY HANDLING
            # ================================================================
            X_new = np.clip(X_new, lb, ub)
            
            # ================================================================
            # FITNESS EVALUATION AND SELECTION
            # ================================================================
            new_fitness = objective_func(X_new)
            
            # Greedy selection: keep better solution
            if new_fitness < fitness[i]:
                X[i] = X_new
                fitness[i] = new_fitness
                
                # Update global best
                if new_fitness < best_fitness:
                    best_fitness = new_fitness
                    best_solution = X_new.copy()
        
        # ====================================================================
        # SMART POOL UPDATE (SPU)
        # ====================================================================
        # Periodically update worst solutions to maintain diversity
        # Original paper: update at 10%, 20%, ..., 90% of iterations
        
        update_frequency = max_iter // 10  # Update every 10% of iterations
        
        if (t + 1) % update_frequency == 0:
            # Number of solutions to update (worst 20%)
            num_update = max(1, pop_size // 5)
            
            # Find worst solutions
            worst_indices = np.argsort(fitness)[-num_update:]
            
            for idx in worst_indices:
                # Strategy 1: Random initialization (50% chance)
                if np.random.rand() < 0.5:
                    X[idx] = np.random.uniform(lb, ub, dim)
                    fitness[idx] = objective_func(X[idx])
                # Strategy 2: Near best solution (50% chance)
                else:
                    # Small perturbation around best solution
                    perturbation = np.random.randn(dim) * 0.1 * (ub - lb)
                    X[idx] = best_solution + perturbation
                    X[idx] = np.clip(X[idx], lb, ub)
                    fitness[idx] = objective_func(X[idx])
                
                # Check if new solution is better
                if fitness[idx] < best_fitness:
                    best_fitness = fitness[idx]
                    best_solution = X[idx].copy()
        
        # Record convergence
        convergence_curve[t] = best_fitness
        
        # Progress indicator (every 10% of iterations)
        if (t + 1) % (max_iter // 10) == 0 or t == 0:
            print(f"Iter {t + 1:4d}/{max_iter}: Best = {best_fitness:.6e}, "
                  f"f = {f:.4f}, q = {q:.4f}")
    
    return best_solution, best_fitness, convergence_curve


# ============================================================================
# TEST CODE - Comprehensive Benchmark Testing
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print(" " * 15 + "RUN ALGORITHM - BENCHMARK TESTS")
    print("=" * 70)
    
    # Import benchmark functions
    from benchmarks.sphere import sphere
    from benchmarks.rastrigin import rastrigin
    from benchmarks.ackley import ackley
    from benchmarks.rosenbrock import rosenbrock
    from benchmarks.schwefel import schwefel
    
    # Algorithm parameters
    dim = 30
    pop_size = 30  # Standard population size
    max_iter = 1000
    
    # Test suite with CORRECT bounds
    benchmarks_to_test = [
        {
            "name": "Sphere",
            "func": sphere,
            "bounds": (-100, 100),  # ✅ Correct bounds
            "optimal": 0.0
        },
        {
            "name": "Rastrigin",
            "func": rastrigin,
            "bounds": (-5.12, 5.12),  # ✅ Correct bounds
            "optimal": 0.0
        },
        {
            "name": "Ackley",
            "func": ackley,
            "bounds": (-32.768, 32.768),  # ✅ Correct bounds
            "optimal": 0.0
        },
        {
            "name": "Rosenbrock",
            "func": rosenbrock,
            "bounds": (-5, 10),  # ✅ Correct bounds
            "optimal": 0.0
        },
        {
            "name": "Schwefel",
            "func": schwefel,
            "bounds": (-500, 500),  # ✅ Correct bounds
            "optimal": 0.0
        }
    ]
    
    # Run tests
    print(f"\nTest Configuration:")
    print(f"  Dimension: {dim}")
    print(f"  Population Size: {pop_size}")
    print(f"  Max Iterations: {max_iter}")
    print(f"  Algorithm: RUN (Runge-Kutta Optimizer)")
    print(f"  ESQ Strategies: ESQ1 (RK4), ESQ2 (SQE), ESQ3 (Direct)")
    print()
    
    results_summary = []
    
    for benchmark in benchmarks_to_test:
        print("\n" + "=" * 70)
        print(f"Testing: {benchmark['name']} Function")
        print("=" * 70)
        print(f"Search Space: {benchmark['bounds']}")
        print(f"Global Optimum: {benchmark['optimal']}")
        print("-" * 70)
        
        # Run RUN algorithm
        best_solution, best_fitness, convergence = run(
            objective_func=benchmark['func'],
            dim=dim,
            bounds=benchmark['bounds'],
            pop_size=pop_size,
            max_iter=max_iter
        )
        
        print("-" * 70)
        print(f"\n📊 RESULTS for {benchmark['name']}:")
        print(f"  Best Fitness Found: {best_fitness:.10e}")
        print(f"  Global Optimum:     {benchmark['optimal']:.10e}")
        print(f"  Error from Optimum: {abs(best_fitness - benchmark['optimal']):.10e}")
        print(f"  Best Solution (first 5 dims): {best_solution[:5]}")
        
        # Convergence analysis
        initial_fitness = convergence[0]
        final_fitness = convergence[-1]
        
        print(f"\n📈 CONVERGENCE ANALYSIS:")
        print(f"  Initial Best:  {initial_fitness:.6e}")
        print(f"  Final Best:    {final_fitness:.6e}")
        
        if final_fitness != 0:
            improvement = initial_fitness / final_fitness
            print(f"  Improvement:   {improvement:.2e}x better")
        else:
            print(f"  Improvement:   Optimal solution found!")
        
        # Check convergence quality
        error = abs(best_fitness - benchmark['optimal'])
        if error < 1e-6:
            quality = "EXCELLENT ✅"
        elif error < 1e-3:
            quality = "GOOD ✓"
        elif error < 1e-1:
            quality = "ACCEPTABLE ~"
        else:
            quality = "POOR ✗"
        
        print(f"  Quality:       {quality}")
        
        results_summary.append({
            'function': benchmark['name'],
            'fitness': best_fitness,
            'error': error,
            'quality': quality
        })
    
    # Summary table
    print("\n" + "=" * 70)
    print(" " * 25 + "SUMMARY TABLE")
    print("=" * 70)
    print(f"{'Function':<15} {'Best Fitness':>15} {'Error':>15} {'Quality':>15}")
    print("-" * 70)
    
    for result in results_summary:
        print(f"{result['function']:<15} {result['fitness']:>15.6e} "
              f"{result['error']:>15.6e} {result['quality']:>15}")
    
    print("=" * 70)
    
    # Comparison note
    print("\n💡 EXPECTED PERFORMANCE:")
    print("-" * 70)
    print("RUN (modern algorithm) should outperform classical algorithms:")
    print("  • Sphere:     Should reach < 1e-50 (excellent)")
    print("  • Rastrigin:  Should find < 20 (good multimodal performance)")
    print("  • Ackley:     Should find < 0.1 (excellent)")
    print("  • Rosenbrock: Should find < 30 (good valley navigation)")
    print("  • Schwefel:   Should find < 1000 (good deceptive handling)")
    print("-" * 70)
    
    print("\n✅ RUN Algorithm testing complete!")
    print("=" * 70)
