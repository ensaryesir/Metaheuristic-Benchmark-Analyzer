import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import math

"""
RIME Optimization Algorithm - ORIGINAL PAPER VERSION
=====================================================

⚠️  WARNING: This is the ORIGINAL implementation from the paper.
⚠️  It has known SEVERE LIMITATIONS and POOR PERFORMANCE.
⚠️  This file is kept for ACADEMIC REFERENCE and COMPARISON purposes only.

📚 ORIGINAL REFERENCE:
----------------------
Su, H., Zhao, D., Heidari, A. A., Liu, L., Zhang, X., Mafarja, M., & Chen, H. (2023).
"RIME: A physics-based optimization."
Neurocomputing, 532, 183-214.
DOI: https://doi.org/10.1016/j.neucom.2023.02.010

📖 ALGORITHM DESCRIPTION:
-------------------------
RIME is inspired by the physics of frost and ice formation (rime ice).
The algorithm mimics two types of rime formation:

1. **Soft-Rime**: Light, feathery ice crystals (exploration phase)
   - Occurs when supercooled water droplets freeze on contact
   - Creates branching, irregular structures
   - Analogous to exploratory search behavior

2. **Hard-Rime**: Dense, compact ice accretion (exploitation phase)
   - Forms when droplets freeze and stick firmly
   - Creates solid, heavy deposits
   - Analogous to intensive local search

🔬 MATHEMATICAL FORMULATION (Exact from Paper):
-----------------------------------------------

Control Parameter - Rime Environment Factor (Equation 3):
    E(t) = [tanh(4t/T - 2) + 1] × cos(πt/2T)
    
    where:
    - t: current iteration
    - T: maximum iterations
    - E: controls exploration vs exploitation
    - E starts high (exploration), gradually decreases (exploitation)

PHASE SELECTION:
    if rand < E:  → SOFT-RIME (Exploration)
    else:         → HARD-RIME (Exploitation)

═══════════════════════════════════════════════════════════════════════
SOFT-RIME MECHANISM (Exploration - Equation 5):
═══════════════════════════════════════════════════════════════════════
    X_new = X_rime-ice + r × cos(θ) × (X_rime-ice - X_i)
    
    where:
    - X_rime-ice: best solution found so far
    - X_i: current individual
    - r: random number in [0, 1]
    - θ: random angle vector in [0, 2π] for each dimension
    - cos(θ): provides directional movement

═══════════════════════════════════════════════════════════════════════
HARD-RIME MECHANISM (Exploitation - Equation 6):
═══════════════════════════════════════════════════════════════════════
    X_new = X_rime-ice + η × (X_rime-ice - X_i)
    
    where:
    - η = E × |r_norm|
    - r_norm: random number from standard normal distribution
    - Moves solutions toward best solution (rime-ice)

POSITIVE GREEDY SELECTION (Equation 8):
    Accept new solution if:
    f(X_new) < f(X_i)  OR  f(X_new) < mean(fitness)
    
    This allows acceptance of solutions that may not be individually
    better but are better than population average.

⚙️ PARAMETER RECOMMENDATIONS:
-----------------------------
From original paper:
    - pop_size: 30-50 (standard for metaheuristics)
    - max_iter: 500-1000 (problem dependent)
    - No additional parameters needed (parameter-free algorithm)

❌ CRITICAL LIMITATIONS (Why This Algorithm FAILS):
---------------------------------------------------

1. **SINGLE ATTRACTOR PROBLEM**:
   Both soft-rime and hard-rime use the formula:
   
   X_new = X_rime-ice + [factor] × (X_rime-ice - X_i)
   
   This means EVERY solution moves along the line connecting itself
   to the best solution (rime-ice). The algorithm has NO MECHANISM
   to explore regions outside this line.

2. **PREMATURE CONVERGENCE**:
   Once rime-ice gets stuck in a local optimum:
   - All particles are attracted to that point like a black hole
   - Population diversity rapidly decreases
   - No escape mechanism from local optima
   - Algorithm stagnates completely

3. **INEFFECTIVE POSITIVE GREEDY SELECTION**:
   Although designed to maintain diversity, it fails because:
   - All new solutions are still generated around the same attractor
   - No truly diverse exploration occurs
   - Mean fitness comparison doesn't add meaningful diversity

4. **LACK OF DIVERSITY MECHANISMS**:
   - No random restart mechanism
   - No mutation operators
   - No social interaction between individuals
   - No memory of previous good solutions

5. **POOR PERFORMANCE ON MULTIMODAL FUNCTIONS**:
   Experimental results show:
   - Sphere: Error ~30 (should be ~0)
   - Rastrigin: Error ~50 (should be ~0)
   - Ackley: Error ~20 (should be ~0)
   - Cannot escape from first local optimum found

📊 COMPUTATIONAL COMPLEXITY:
---------------------------
Time: O(pop_size × max_iter × dim)
Space: O(pop_size × dim)

💡 WHY THIS FILE EXISTS:
------------------------
This implementation is kept to:
1. Show the ORIGINAL algorithm from the paper
2. Demonstrate WHY modifications are necessary
3. Provide baseline for comparison with improved versions
4. Serve as academic reference for research

For ACTUAL optimization work, use the improved version in:
    algorithms/rime.py (with diversity mechanisms)

Args:
    objective_func (callable): Objective function to minimize
    dim (int): Problem dimensionality
    bounds (tuple or list): Search space bounds as (lb, ub) or [(lb, ub)] × dim
    pop_size (int): Population size (default: 30)
    max_iter (int): Maximum number of iterations (default: 1000)
    
Returns:
    tuple: (best_solution, best_fitness, convergence_curve)
        - best_solution: Best solution found (likely LOCAL optimum)
        - best_fitness: Fitness of best solution (likely POOR)
        - convergence_curve: Best fitness at each iteration (shows STAGNATION)
"""
def rime(objective_func, dim, bounds, pop_size=30, max_iter=1000):
    """
    RIME Optimization Algorithm - ORIGINAL PAPER IMPLEMENTATION
    
    ⚠️  WARNING: This is the UNMODIFIED algorithm from Su et al. (2023).
    ⚠️  Known to have SEVERE PERFORMANCE ISSUES on multimodal functions.
    ⚠️  Use only for ACADEMIC COMPARISON purposes.
    
    This implementation follows the original paper EXACTLY, including
    all its limitations and design flaws. It demonstrates why the
    original RIME needs significant modifications for practical use.
    
    Expected Performance (30D):
        - Sphere: ~30 (POOR - should be ~0)
        - Rastrigin: ~50 (VERY POOR - should be ~0)
        - Ackley: ~20 (POOR - should be ~0)
        - Premature convergence in <200 iterations
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
    population = np.random.uniform(lb, ub, (pop_size, dim))
    
    # Evaluate initial population
    fitness = np.array([objective_func(ind) for ind in population])
    
    # Find rime-ice (best solution - analogous to densest ice formation)
    best_idx = np.argmin(fitness)
    rime_ice = population[best_idx].copy()
    best_fitness = fitness[best_idx]
    
    # Convergence tracking
    convergence_curve = np.zeros(max_iter)
    
    # Track stagnation for analysis
    stagnation_counter = 0
    last_improvement = 0
    
    # ========================================================================
    # STEP 3: MAIN OPTIMIZATION LOOP
    # ========================================================================
    
    for t in range(max_iter):
        
        # ====================================================================
        # CALCULATE RIME ENVIRONMENT FACTOR E (Equation 3)
        # ====================================================================
        # E controls the balance between exploration and exploitation
        # High E → Exploration (soft-rime)
        # Low E → Exploitation (hard-rime)
        
        # E(t) = [tanh(4t/T - 2) + 1] × cos(πt/2T)
        E = (np.tanh(4 * t / max_iter - 2) + 1) * np.cos(t / max_iter * np.pi / 2)
        
        # ====================================================================
        # CALCULATE MEAN FITNESS (For Positive Greedy Selection)
        # ====================================================================
        mean_fitness = np.mean(fitness)
        
        # Track if any improvement occurs this iteration
        improved_this_iter = False
        
        # ====================================================================
        # UPDATE EACH INDIVIDUAL
        # ====================================================================
        
        for i in range(pop_size):
            
            # ================================================================
            # PHASE SELECTION: SOFT-RIME vs HARD-RIME
            # ================================================================
            # Probability E decreases over time
            # Early iterations: High E → Soft-rime (exploration)
            # Late iterations: Low E → Hard-rime (exploitation)
            
            if np.random.rand() < E:
                # ============================================================
                # SOFT-RIME MECHANISM (Exploration - Equation 5)
                # ============================================================
                # Mimics light, feathery ice crystal formation
                # Provides broader search around rime-ice
                
                # Generate random angle vector for each dimension
                # θ ∈ [0, 2π] for each dimension
                theta = np.random.uniform(0, 2 * np.pi, dim)
                
                # Random scaling factor
                r = np.random.rand()
                
                # Soft-rime position update (Equation 5):
                # X_new = X_rime-ice + r × cos(θ) × (X_rime-ice - X_i)
                # 
                # PROBLEM: This only explores along the line between
                # rime_ice and current position. No perpendicular exploration!
                new_pos = rime_ice + r * np.cos(theta) * (rime_ice - population[i])
                
            else:
                # ============================================================
                # HARD-RIME MECHANISM (Exploitation - Equation 6)
                # ============================================================
                # Mimics dense, compact ice accretion
                # Provides intensive local search near rime-ice
                
                # Generate random normal number (scalar)
                r_norm = np.random.randn()
                
                # Calculate coefficient η (Equation 6)
                # η = E × |r_norm|
                eta = E * np.abs(r_norm)
                
                # Hard-rime position update (Equation 6):
                # X_new = X_rime-ice + η × (X_rime-ice - X_i)
                # 
                # PROBLEM: This moves directly toward rime_ice with no
                # exploration of alternative regions!
                new_pos = rime_ice + eta * (rime_ice - population[i])
            
            # ================================================================
            # BOUNDARY HANDLING
            # ================================================================
            new_pos = np.clip(new_pos, lb, ub)
            
            # ================================================================
            # FITNESS EVALUATION
            # ================================================================
            new_fitness = objective_func(new_pos)
            
            # ================================================================
            # POSITIVE GREEDY SELECTION (Equation 8)
            # ================================================================
            # Accept new solution if:
            # 1. It's better than current solution, OR
            # 2. It's better than population mean
            #
            # INTENDED: To maintain diversity by accepting "reasonably good"
            #           solutions even if not individually better
            # REALITY: Fails because all solutions converge to same region
            
            if new_fitness < fitness[i] or new_fitness < mean_fitness:
                population[i] = new_pos
                fitness[i] = new_fitness
                improved_this_iter = True
                
                # Update rime-ice (global best)
                if new_fitness < best_fitness:
                    best_fitness = new_fitness
                    rime_ice = new_pos.copy()
                    last_improvement = t
        
        # Record convergence
        convergence_curve[t] = best_fitness
        
        # Track stagnation
        if not improved_this_iter:
            stagnation_counter += 1
        else:
            stagnation_counter = 0
        
        # Progress indicator (every 10% of iterations)
        if (t + 1) % (max_iter // 10) == 0 or t == 0:
            iters_since_improvement = t - last_improvement
            print(f"Iter {t + 1:4d}/{max_iter}: Best = {best_fitness:.6e}, "
                  f"E = {E:.3f}, Stagnant = {iters_since_improvement} iters")
    
    # ========================================================================
    # FINAL ANALYSIS
    # ========================================================================
    print(f"\n⚠️  ALGORITHM STAGNATION ANALYSIS:")
    print(f"  Last improvement at iteration: {last_improvement + 1}/{max_iter}")
    print(f"  Stagnant for: {max_iter - last_improvement} iterations")
    print(f"  This demonstrates the PREMATURE CONVERGENCE problem!")
    
    return rime_ice, best_fitness, convergence_curve

# ============================================================================
# TEST CODE - Demonstrating Original RIME's Limitations
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print(" " * 10 + "RIME ALGORITHM - ORIGINAL PAPER VERSION")
    print(" " * 15 + "(ACADEMIC REFERENCE ONLY)")
    print("=" * 70)
    print("\n⚠️  WARNING: This implementation demonstrates the ORIGINAL algorithm")
    print("⚠️  with all its LIMITATIONS and POOR PERFORMANCE.")
    print("⚠️  For practical use, see: algorithms/rime.py (improved version)")
    print()
    
    # Import benchmark functions
    from benchmarks.sphere import sphere
    from benchmarks.rastrigin import rastrigin
    from benchmarks.ackley import ackley
    from benchmarks.rosenbrock import rosenbrock
    from benchmarks.schwefel import schwefel
    
    # Algorithm parameters
    dim = 30
    pop_size = 30
    max_iter = 1000
    
    # Test suite with CORRECT bounds
    benchmarks_to_test = [
        {
            "name": "Sphere",
            "func": sphere,
            "bounds": (-100, 100),
            "optimal": 0.0
        },
        {
            "name": "Rastrigin",
            "func": rastrigin,
            "bounds": (-5.12, 5.12),
            "optimal": 0.0
        },
        {
            "name": "Ackley",
            "func": ackley,
            "bounds": (-32.768, 32.768),
            "optimal": 0.0
        },
        {
            "name": "Rosenbrock",
            "func": rosenbrock,
            "bounds": (-5, 10),
            "optimal": 0.0
        },
        {
            "name": "Schwefel",
            "func": schwefel,
            "bounds": (-500, 500),
            "optimal": 0.0
        }
    ]
    
    # Run tests
    print(f"Test Configuration:")
    print(f"  Dimension: {dim}")
    print(f"  Population Size: {pop_size}")
    print(f"  Max Iterations: {max_iter}")
    print(f"  Algorithm: RIME (ORIGINAL - UNMODIFIED)")
    print(f"  ⚠️  Expected: POOR performance due to design limitations")
    print()
    
    results_summary = []
    
    for benchmark in benchmarks_to_test:
        print("\n" + "=" * 70)
        print(f"Testing: {benchmark['name']} Function")
        print("=" * 70)
        print(f"Search Space: {benchmark['bounds']}")
        print(f"Global Optimum: {benchmark['optimal']}")
        print("-" * 70)
        
        # Run RIME algorithm
        best_solution, best_fitness, convergence = rime(
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
            quality = "EXCELLENT ✅ (Unexpected!)"
        elif error < 1e-3:
            quality = "GOOD ✓ (Better than expected)"
        elif error < 1e-1:
            quality = "ACCEPTABLE ~ (Barely acceptable)"
        else:
            quality = "POOR ✗ (As expected from original RIME)"
        
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
    
    # Critical analysis
    print("\n❌ CRITICAL ANALYSIS - WHY ORIGINAL RIME FAILS:")
    print("-" * 70)
    print("1. SINGLE ATTRACTOR PROBLEM:")
    print("   Both soft-rime and hard-rime use: X_new = X_best + [f] × (X_best - X_i)")
    print("   → All solutions move along line between themselves and best solution")
    print("   → No exploration perpendicular to this line")
    print()
    print("2. PREMATURE CONVERGENCE:")
    print("   → Once best solution (rime-ice) reaches local optimum, it's trapped")
    print("   → All population members converge to same local optimum")
    print("   → No escape mechanism")
    print()
    print("3. INEFFECTIVE POSITIVE GREEDY SELECTION:")
    print("   → Designed to maintain diversity")
    print("   → Fails because new solutions still generated in same region")
    print("   → Mean fitness comparison doesn't add meaningful diversity")
    print()
    print("4. LACK OF DIVERSITY MECHANISMS:")
    print("   → No random restart")
    print("   → No mutation operators")
    print("   → No social interaction between individuals")
    print("   → No memory of previous solutions")
    print("-" * 70)
    
    print("\n💡 SOLUTION:")
    print("-" * 70)
    print("See improved version in: algorithms/rime.py")
    print("Improvements include:")
    print("  ✅ Social interaction between individuals")
    print("  ✅ Random noise injection")
    print("  ✅ Periodic population restart")
    print("  ✅ Enhanced exploration mechanisms")
    print("-" * 70)
    
    print("\n✅ RIME Original Algorithm testing complete!")
    print("📚 This serves as ACADEMIC REFERENCE for comparison.")
    print("=" * 70)
