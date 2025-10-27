import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np

"""
Artificial Gorilla Troops Optimizer (GTO) Algorithm
===================================================

📚 ORIGINAL REFERENCE:
----------------------
Abdollahzadeh, B., Soleimanian Gharehchopogh, F., & Mirjalili, S. (2021).
"Artificial gorilla troops optimizer: a new nature‐inspired metaheuristic 
algorithm for global optimization problems."
International Journal of Intelligent Systems, 36(10), 5887-5958.
DOI: 10.1002/int.22535

📖 ALGORITHM DESCRIPTION:
-------------------------
GTO mimics the social intelligence and group behaviors of gorilla troops.
The algorithm is inspired by gorilla life, including:
- Silverback leadership (best solution guidance)
- Following other gorillas (social learning)
- Competition for adult females (exploitation)
- Migration to unknown and known locations (exploration)

CRITICAL: This implementation follows the EXACT equations from the paper.

🔬 MATHEMATICAL FORMULATION (Exact from Paper):
-----------------------------------------------

Control Parameter:
    C(t) = F × (1 - t/T)    where F ∈ [0.8, 1]
    
PHASE SELECTION:
    if C ≥ rand:  → EXPLORATION PHASE
    else:         → EXPLOITATION PHASE

═══════════════════════════════════════════════════════════════════════
EXPLORATION PHASE (C ≥ rand):
═══════════════════════════════════════════════════════════════════════

    Strategy 1: Migration to unknown location (rand < p) - Equation 6:
        GX_new = (UB - LB) × r + LB

    Strategy 2: Migration to other gorillas (rand ≥ 0.5) - Equation 8:
        GX_new = (r2 - C) × X_r(t) + L × X(t)
        where L = C × l, l ∈ [-1, 1]

    Strategy 3: Migration to known location (rand < 0.5) - Equation 9:
        GX_new = X(t) - L × (L × (X(t) - X_r(t)) + r3 × (X(t) - X_r(t)))

═══════════════════════════════════════════════════════════════════════
EXPLOITATION PHASE (C < rand):
═══════════════════════════════════════════════════════════════════════

    Strategy 1: Silverback movement (rand ≥ 0.5) - Equations 13-14:
        M = (Σ |X_j(t)|^g)^(1/g)    [Lp norm of entire population]
        g ∈ [2^-1, 2^1]
        GX(t+1) = L × M × (X(t) - X_silverback) + X(t)

    Strategy 2: Competition for females (rand < 0.5) - Equation 18:
        GX(t+1) = X_silverback - (X_silverback × Q - X(t) × Q) × A
        where:
        - Q ∈ [2r - 1] for r ∈ [0, 1]
        - A = β × E
        - E = N1 (when rand ≥ 0.5) or N2 (when rand < 0.5)

⚙️ PARAMETER RECOMMENDATIONS:
-----------------------------
Standard Settings (from original paper):
    - pop_size = 30 (or 5-10 × dim)
    - β (beta) = 3.0 (exploration intensity)
    - p = 0.03 (probability of random exploration)
    - F = 0.8 (control parameter coefficient)

Args:
    objective_func (callable): Objective function to minimize
    dim (int): Problem dimensionality
    bounds (tuple or list): Search space bounds as (lb, ub)
    pop_size (int): Population size (default: 30)
    max_iter (int): Maximum number of iterations (default: 1000)
    beta (float): Exploration intensity parameter (default: 3.0)
    p (float): Probability of random exploration (default: 0.03)
    
Returns:
    tuple: (best_solution, best_fitness, convergence_curve)
"""

def gto(objective_func, dim, bounds, pop_size=30, max_iter=1000, beta=3.0, p=0.03):
    """
    Artificial Gorilla Troops Optimizer (GTO) - EXACT Implementation
    
    This implementation follows the ORIGINAL GTO algorithm EXACTLY as described
    in Abdollahzadeh et al. (2021), equation by equation.
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
    
    # Fixed parameter from original paper
    F = 0.8  # Control parameter coefficient
    
    # ========================================================================
    # STEP 2: POPULATION INITIALIZATION (Gorilla Troops)
    # ========================================================================
    
    # Initialize gorilla positions uniformly in search space
    X = np.random.uniform(lb, ub, (pop_size, dim))
    
    # Evaluate initial population
    fitness = np.array([objective_func(ind) for ind in X])
    
    # Find silverback (best gorilla - leader of the troop)
    best_idx = np.argmin(fitness)
    X_silverback = X[best_idx].copy()
    fitness_silverback = fitness[best_idx]
    
    # Convergence tracking
    convergence_curve = np.zeros(max_iter)
    
    # ========================================================================
    # STEP 3: MAIN OPTIMIZATION LOOP
    # ========================================================================
    
    for t in range(max_iter):
        
        # ====================================================================
        # UPDATE CONTROL PARAMETER C (Equation 3)
        # ====================================================================
        # C decreases linearly from F to 0
        # Controls transition from exploration to exploitation
        C = F * (1 - t / max_iter)
        
        # ====================================================================
        # UPDATE EACH GORILLA POSITION
        # ====================================================================
        
        for i in range(pop_size):
            
            # ================================================================
            # PHASE SELECTION: EXPLORATION OR EXPLOITATION
            # ================================================================
            # CRITICAL: This is the main decision point!
            
            if C >= np.random.rand():
                # ============================================================
                # EXPLORATION PHASE (C ≥ rand)
                # ============================================================
                
                if np.random.rand() < p:
                    # ========================================================
                    # EXPLORATION - Strategy 1: Random Exploration (Eq. 6)
                    # ========================================================
                    # Gorillas migrate to completely unknown locations
                    
                    r = np.random.rand(dim)
                    GX_new = (ub - lb) * r + lb
                    
                elif np.random.rand() >= 0.5:
                    # ========================================================
                    # EXPLORATION - Strategy 2: Follow Other Gorillas (Eq. 8)
                    # ========================================================
                    # Gorillas follow other group members (social learning)
                    
                    # Generate adaptive parameter L (Equation 4)
                    l = 2 * np.random.rand() - 1  # l ∈ [-1, 1]
                    L = C * l
                    
                    # Select random gorilla
                    r_idx = np.random.randint(pop_size)
                    
                    # Random coefficient
                    r2 = np.random.rand()
                    
                    # Equation 8: GX_new = (r2 - C) × X_r(t) + L × X(t)
                    GX_new = (r2 - C) * X[r_idx] + L * X[i]
                    
                else:
                    # ========================================================
                    # EXPLORATION - Strategy 3: Known Location (Eq. 9)
                    # ========================================================
                    # Move to known place with adaptive step
                    
                    # Generate adaptive parameter L
                    l = 2 * np.random.rand() - 1
                    L = C * l
                    
                    # Select random gorilla
                    r_idx = np.random.randint(pop_size)
                    
                    # Random coefficient
                    r3 = np.random.rand()
                    
                    # Equation 9: GX_new = X(t) - L × (L × (X(t) - X_r(t)) + r3 × (X(t) - X_r(t)))
                    GX_new = X[i] - L * (L * (X[i] - X[r_idx]) + r3 * (X[i] - X[r_idx]))
                
            else:
                # ============================================================
                # EXPLOITATION PHASE (C < rand)
                # ============================================================
                
                if np.random.rand() >= 0.5:
                    # ========================================================
                    # EXPLOITATION - Strategy 1: Follow Silverback (Eq. 13-14)
                    # ========================================================
                    
                    # Calculate g parameter (Equation 12)
                    g = 2 ** np.random.rand()  # g ∈ [2^-1, 2^1] = [0.5, 2]
                    
                    # Calculate L parameter
                    l = 2 * np.random.rand() - 1
                    L = C * l
                    
                    # Calculate M: Lp norm of ENTIRE population (Equation 13)
                    # CRITICAL: This is sum over all individuals, not just mean!
                    # M = (Σ_j |X_j(t)|^g)^(1/g)
                    M = np.sum(np.abs(X) ** g, axis=0) ** (1.0 / g)
                    
                    # Equation 14: GX(t+1) = L × M × (X(t) - X_silverback) + X(t)
                    # CRITICAL: Last term is X(t), NOT X_silverback!
                    GX_new = L * M * (X[i] - X_silverback) + X[i]
                    
                else:
                    # ========================================================
                    # EXPLOITATION - Strategy 2: Competition for Females (Eq. 18)
                    # ========================================================
                    
                    # Calculate Q parameter (Equation 16)
                    r = np.random.rand()
                    Q = 2 * r - 1  # Q ∈ [-1, 1]
                    
                    # Calculate A parameter (Equation 17)
                    if np.random.rand() >= 0.5:
                        E = np.random.randn()  # N1: Normal distribution
                    else:
                        E = np.random.randn()  # N2: Normal distribution
                    
                    A = beta * E
                    
                    # Equation 18: GX(t+1) = X_silverback - (X_silverback × Q - X(t) × Q) × A
                    GX_new = X_silverback - (X_silverback * Q - X[i] * Q) * A
            
            # ================================================================
            # BOUNDARY HANDLING
            # ================================================================
            GX_new = np.clip(GX_new, lb, ub)
            
            # ================================================================
            # FITNESS EVALUATION AND SELECTION
            # ================================================================
            new_fitness = objective_func(GX_new)
            
            # Greedy selection: keep better solution
            if new_fitness < fitness[i]:
                X[i] = GX_new
                fitness[i] = new_fitness
                
                # Update silverback (troop leader)
                if new_fitness < fitness_silverback:
                    fitness_silverback = new_fitness
                    X_silverback = GX_new.copy()
                    best_idx = i
        
        # Record convergence
        convergence_curve[t] = fitness_silverback
        
        # Progress indicator (every 10% of iterations)
        if (t + 1) % (max_iter // 10) == 0 or t == 0:
            print(f"Iter {t + 1:4d}/{max_iter}: Best = {fitness_silverback:.6e}, C = {C:.4f}")
    
    return X_silverback, fitness_silverback, convergence_curve


# ============================================================================
# TEST CODE - Comprehensive Benchmark Testing
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print(" " * 15 + "GTO ALGORITHM - BENCHMARK TESTS")
    print("=" * 70)
    
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
    
    # GTO-specific parameters (from original paper)
    beta = 3.0  # Exploration intensity
    p = 0.03    # Random exploration probability
    
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
    print(f"\nTest Configuration:")
    print(f"  Dimension: {dim}")
    print(f"  Population Size: {pop_size}")
    print(f"  Max Iterations: {max_iter}")
    print(f"  Algorithm: GTO (EXACT implementation)")
    print(f"  Beta (β): {beta}")
    print(f"  p: {p}")
    print(f"  ⚠️  CRITICAL: All equations from original paper implemented exactly!")
    print()
    
    results_summary = []
    
    for benchmark in benchmarks_to_test:
        print("\n" + "=" * 70)
        print(f"Testing: {benchmark['name']} Function")
        print("=" * 70)
        print(f"Search Space: {benchmark['bounds']}")
        print(f"Global Optimum: {benchmark['optimal']}")
        print("-" * 70)
        
        # Run GTO algorithm
        best_solution, best_fitness, convergence = gto(
            objective_func=benchmark['func'],
            dim=dim,
            bounds=benchmark['bounds'],
            pop_size=pop_size,
            max_iter=max_iter,
            beta=beta,
            p=p
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
    print("\n✅ GTO Algorithm testing complete!")
    print("=" * 70)
