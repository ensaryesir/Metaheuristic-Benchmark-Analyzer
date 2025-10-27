import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import math

"""
Marine Predators Algorithm (MPA)
=================================

📚 ORIGINAL REFERENCE:
----------------------
Faramarzi, A., Heidarinejad, M., Mirjalili, S., & Gandomi, A. H. (2020).
"Marine Predators Algorithm: A nature-inspired metaheuristic."
Expert Systems with Applications, 152, 113377.
DOI: 10.1016/j.eswa.2020.113377

📖 ALGORITHM DESCRIPTION:
-------------------------
MPA is inspired by widespread foraging strategies namely Lévy and Brownian 
movements in ocean predators along with optimal encounter rate policy in 
biological interaction between predator and prey.

The algorithm mimics:
1. High-velocity ratio (exploration) - Brownian motion
2. Unit-velocity ratio (exploitation) - Brownian + Lévy
3. Low-velocity ratio (exploitation) - Lévy motion

Key Features:
- Three-phase velocity adjustment mechanism
- Lévy and Brownian motion strategies
- FADs (Fish Aggregating Devices) effect for diversity
- Elite matrix for memory saving
- Eddy formation and FADs effect

🔬 MATHEMATICAL FORMULATION (Exact from Paper):
-----------------------------------------------

OPTIMIZATION PHASES (Based on velocity ratios):

Phase 1: Exploration (iter < max_iter/3) - High velocity ratio
    When Prey moves faster than Predator:
    
    stepsize_i = RB ⊗ (Elite_i - RB ⊗ Prey_i)          [Eq. 6]
    Prey_i = Prey_i + P × R ⊗ stepsize_i                [Eq. 7]
    
    where:
    - RB: vector of random numbers (Brownian motion)
    - ⊗: element-wise multiplication
    - Elite: best positions (predators)
    - P: constant (0.5)
    - R: random number [0,1]

Phase 2: Transition (max_iter/3 ≤ iter < 2×max_iter/3) - Unit velocity
    When Predator and Prey move at similar velocities:
    
    For first half of population:
        stepsize_i = RB ⊗ (RB ⊗ Elite_i - Prey_i)       [Eq. 9]
        Prey_i = Prey_i + P × CF × stepsize_i            [Eq. 10]
    
    For second half of population:
        stepsize_i = RL ⊗ (Elite_i - RL ⊗ Prey_i)       [Eq. 11]
        Prey_i = Prey_i + P × CF × stepsize_i            [Eq. 12]
    
    where:
    - RL: vector of random numbers (Lévy flight)
    - CF: adaptive parameter = (1 - iter/max_iter)^(2×iter/max_iter)

Phase 3: Exploitation (iter ≥ 2×max_iter/3) - Low velocity ratio
    When Predator moves faster than Prey:
    
    stepsize_i = RL ⊗ (RL ⊗ Elite_i - Prey_i)           [Eq. 13]
    Prey_i = Elite_i + P × CF × stepsize_i               [Eq. 14]

FADs Effect (Fish Aggregating Devices):
    Simulates environmental effects and prey swarming:
    
    If rand < FADs:
        If rand < 0.2:
            Prey_i = Prey_i + CF × [X_min + R ⊗ (X_max - X_min)] ⊗ U  [Eq. 15]
        else:
            Prey_i = Prey_i + [FADs(1-r) + r] × (Prey_r1 - Prey_r2)   [Eq. 16]
    
    where:
    - U: binary vector
    - r, r1, r2: random indices

Marine Memory Saving:
    Elite_i is updated when Prey_i finds better position
    Preserves best-found positions throughout optimization

⚙️ PARAMETER RECOMMENDATIONS:
-----------------------------
Standard Settings (from original paper):
    - pop_size = 30 (for small problems) or 50-100 (for complex problems)
    - P = 0.5 (constant parameter)
    - FADs = 0.2 (Fish Aggregating Devices effect)
    - No additional parameters needed

Parameter Tuning Guidelines:
    - P controls step size (typically kept at 0.5)
    - FADs controls diversity (0.2 is optimal from experiments)
    - Higher FADs → more exploration
    - Lower FADs → more exploitation

💪 STRENGTHS:
-------------
- Excellent balance between exploration and exploitation
- Adaptive phase transitions
- Lévy flight for better exploration
- Memory-based elite preservation
- FADs effect prevents premature convergence
- Good performance on multimodal functions

⚠️ WEAKNESSES:
--------------
- More complex than basic algorithms (PSO, DE)
- Requires careful implementation of Lévy flight
- Multiple random components increase stochasticity
- May be slower per iteration due to complexity

📊 COMPUTATIONAL COMPLEXITY:
---------------------------
Time: O(pop_size × max_iter × dim)
Space: O(pop_size × dim)

Args:
    objective_func (callable): Objective function to minimize
    dim (int): Problem dimensionality
    bounds (tuple or list): Search space bounds as (lb, ub) or [(lb, ub)] × dim
    pop_size (int): Population size (default: 30)
    max_iter (int): Maximum number of iterations (default: 1000)
    P (float): Constant parameter (default: 0.5, from original paper)
    FADs (float): Fish Aggregating Devices effect (default: 0.2, from paper)
    
Returns:
    tuple: (best_solution, best_fitness, convergence_curve)
        - best_solution: Best solution found
        - best_fitness: Fitness of best solution
        - convergence_curve: Best fitness at each iteration
"""

def mpa(objective_func, dim, bounds, pop_size=30, max_iter=1000, P=0.5, FADs=0.2):
    """
    Marine Predators Algorithm - EXACT Implementation
    
    This implementation follows the ORIGINAL MPA algorithm as described
    in Faramarzi et al. (2020), equation by equation.
    
    The algorithm uses three-phase optimization strategy based on velocity
    ratios between predator and prey, combined with Lévy and Brownian
    movement patterns.
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
    # STEP 2: POPULATION INITIALIZATION (Prey - Candidate Solutions)
    # ========================================================================
    
    # Initialize prey population uniformly in search space
    prey = np.random.uniform(lb, ub, (pop_size, dim))
    
    # Evaluate initial population
    fitness = np.array([objective_func(ind) for ind in prey])
    
    # Initialize Elite matrix (Top predators - best solutions)
    # Elite matrix represents predators (best solutions found so far)
    elite = prey.copy()
    elite_fitness = fitness.copy()
    
    # Find global best (top predator)
    best_idx = np.argmin(fitness)
    best_solution = prey[best_idx].copy()
    best_fitness = fitness[best_idx]
    
    # Convergence tracking
    convergence_curve = np.zeros(max_iter)
    
    # Step size matrix
    stepsize = np.zeros((pop_size, dim))
    
    # ========================================================================
    # STEP 3: MAIN OPTIMIZATION LOOP
    # ========================================================================
    
    for iteration in range(max_iter):
        
        # ====================================================================
        # CALCULATE ADAPTIVE PARAMETER CF (Equation 8)
        # ====================================================================
        # CF controls the step size adaptively
        # Decreases non-linearly from 1 to 0
        CF = (1 - iteration / max_iter) ** (2 * iteration / max_iter)
        
        # ====================================================================
        # GENERATE MOVEMENT PATTERNS
        # ====================================================================
        # RL: Lévy flight random vector (for exploration)
        RL = 0.05 * levy_flight(pop_size, dim)
        
        # RB: Brownian motion random vector (for exploitation)
        RB = np.random.randn(pop_size, dim)
        
        # ====================================================================
        # PHASE SELECTION BASED ON ITERATION
        # ====================================================================
        
        if iteration < max_iter / 3:
            # ================================================================
            # PHASE 1: EXPLORATION (High velocity ratio - iter < max_iter/3)
            # ================================================================
            # Prey moves faster than Predator
            # Uses Brownian motion for exploration
            
            for i in range(pop_size):
                # Random coefficient for this prey
                R = np.random.rand(dim)
                
                # Calculate step size (Equation 6)
                # stepsize = RB ⊗ (Elite - RB ⊗ Prey)
                stepsize[i] = RB[i] * (elite[i] - RB[i] * prey[i])
                
                # Update prey position (Equation 7)
                # Prey = Prey + P × R ⊗ stepsize
                prey[i] = prey[i] + P * R * stepsize[i]
        
        elif iteration < 2 * max_iter / 3:
            # ================================================================
            # PHASE 2: TRANSITION (Unit velocity - max_iter/3 ≤ iter < 2×max_iter/3)
            # ================================================================
            # Predator and Prey move at similar velocities
            # Half population uses Brownian, half uses Lévy
            
            for i in range(pop_size):
                if i < pop_size / 2:
                    # ========================================================
                    # First half: Use BROWNIAN motion (Equations 9-10)
                    # ========================================================
                    
                    # Calculate step size (Equation 9)
                    # stepsize = RB ⊗ (RB ⊗ Elite - Prey)
                    stepsize[i] = RB[i] * (RB[i] * elite[i] - prey[i])
                    
                    # Update prey position (Equation 10)
                    # Prey = Prey + P × CF × stepsize
                    prey[i] = prey[i] + P * CF * stepsize[i]
                
                else:
                    # ========================================================
                    # Second half: Use LÉVY flight (Equations 11-12)
                    # ========================================================
                    
                    # Calculate step size (Equation 11)
                    # stepsize = RL ⊗ (Elite - RL ⊗ Prey)
                    stepsize[i] = RL[i] * (elite[i] - RL[i] * prey[i])
                    
                    # Update prey position (Equation 12)
                    # Prey = Prey + P × CF × stepsize
                    prey[i] = prey[i] + P * CF * stepsize[i]
        
        else:
            # ================================================================
            # PHASE 3: EXPLOITATION (Low velocity - iter ≥ 2×max_iter/3)
            # ================================================================
            # Predator moves faster than Prey
            # Uses Lévy flight for final exploitation
            
            for i in range(pop_size):
                # Calculate step size (Equation 13)
                # stepsize = RL ⊗ (RL ⊗ Elite - Prey)
                stepsize[i] = RL[i] * (RL[i] * elite[i] - prey[i])
                
                # Update prey position (Equation 14)
                # Prey = Elite + P × CF × stepsize
                # NOTE: Prey moves TOWARDS elite position
                prey[i] = elite[i] + P * CF * stepsize[i]
        
        # ====================================================================
        # BOUNDARY HANDLING
        # ====================================================================
        prey = np.clip(prey, lb, ub)
        
        # ====================================================================
        # FITNESS EVALUATION AND ELITE UPDATE
        # ====================================================================
        
        for i in range(pop_size):
            new_fitness = objective_func(prey[i])
            
            # Update if new position is better
            if new_fitness < fitness[i]:
                fitness[i] = new_fitness
                
                # Update Elite matrix (Marine Memory Saving)
                # Elite remembers best positions found
                elite[i] = prey[i].copy()
                elite_fitness[i] = new_fitness
                
                # Update global best
                if new_fitness < best_fitness:
                    best_fitness = new_fitness
                    best_solution = prey[i].copy()
        
        # ====================================================================
        # FADs EFFECT (Fish Aggregating Devices) - Equations 15-16
        # ====================================================================
        # Simulates environmental effects on prey behavior
        # Adds diversity and prevents premature convergence
        
        if np.random.rand() < FADs:
            # Binary matrix for selective update
            U = np.random.rand(pop_size, dim) < FADs
            
            for i in range(pop_size):
                if np.random.rand() < 0.2:
                    # ========================================================
                    # Strategy 1: Random position (Equation 15)
                    # ========================================================
                    # Prey explores random locations (eddy formation)
                    
                    prey[i] = prey[i] + CF * (lb + np.random.rand(dim) * (ub - lb)) * U[i]
                
                else:
                    # ========================================================
                    # Strategy 2: Follow other prey (Equation 16)
                    # ========================================================
                    # Prey interacts with other prey (swarming behavior)
                    
                    # Select two random prey
                    r1, r2 = np.random.randint(0, pop_size, 2)
                    
                    # FADs coefficient
                    r = np.random.rand()
                    
                    # Update position
                    prey[i] = prey[i] + (FADs * (1 - r) + r) * (prey[r1] - prey[r2]) * U[i]
                
                # Ensure within bounds after FADs effect
                prey[i] = np.clip(prey[i], lb, ub)
        
        # Record convergence
        convergence_curve[iteration] = best_fitness
        
        # Progress indicator (every 10% of iterations)
        if (iteration + 1) % (max_iter // 10) == 0 or iteration == 0:
            print(f"Iter {iteration + 1:4d}/{max_iter}: Best = {best_fitness:.6e}, "
                  f"CF = {CF:.4f}, Phase = {1 if iteration < max_iter/3 else (2 if iteration < 2*max_iter/3 else 3)}")
    
    return best_solution, best_fitness, convergence_curve

def levy_flight(n, dim, beta=1.5):
    """
    Generate Lévy flight random walk
    
    Lévy flight is a random walk where step lengths have a probability
    distribution that is heavy-tailed. This creates a mixture of short
    and occasional long steps, which is beneficial for exploration.
    
    Mathematical Formulation (Mantegna's algorithm):
    ------------------------------------------------
    Step = u / |v|^(1/β)
    
    where:
    - u ~ N(0, σ_u²)
    - v ~ N(0, 1)
    - σ_u = [Γ(1+β) × sin(πβ/2) / (Γ((1+β)/2) × β × 2^((β-1)/2))]^(1/β)
    - β: Lévy exponent (typically 1.5)
    
    Args:
        n (int): Number of samples
        dim (int): Dimensionality
        beta (float): Lévy exponent (default: 1.5)
    
    Returns:
        numpy.ndarray: Lévy flight random steps of shape (n, dim)
    
    Reference:
        Mantegna, R. N. (1994). "Fast, accurate algorithm for numerical 
        simulation of Levy stable stochastic processes."
        Physical Review E, 49(4), 4677.
    """
    
    # Calculate sigma_u using Gamma function (Mantegna's formula)
    numerator = math.gamma(1 + beta) * np.sin(np.pi * beta / 2)
    denominator = math.gamma((1 + beta) / 2) * beta * (2 ** ((beta - 1) / 2))
    sigma_u = (numerator / denominator) ** (1 / beta)
    
    # Generate random samples from normal distributions
    u = np.random.normal(0, sigma_u, (n, dim))
    v = np.random.normal(0, 1, (n, dim))
    
    # Calculate Lévy flight step
    # Add small epsilon to avoid division by zero
    step = u / (np.abs(v) ** (1 / beta) + 1e-10)
    
    return step

# ============================================================================
# TEST CODE - Comprehensive Benchmark Testing
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print(" " * 15 + "MPA ALGORITHM - BENCHMARK TESTS")
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
    
    # MPA-specific parameters (from original paper)
    P = 0.5      # Constant parameter
    FADs = 0.2   # Fish Aggregating Devices effect
    
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
    print(f"  Algorithm: MPA (EXACT implementation)")
    print(f"  P: {P} (constant parameter)")
    print(f"  FADs: {FADs} (Fish Aggregating Devices)")
    print(f"  ⚠️  CRITICAL: Three-phase velocity strategy implemented exactly!")
    print()
    
    results_summary = []
    
    for benchmark in benchmarks_to_test:
        print("\n" + "=" * 70)
        print(f"Testing: {benchmark['name']} Function")
        print("=" * 70)
        print(f"Search Space: {benchmark['bounds']}")
        print(f"Global Optimum: {benchmark['optimal']}")
        print("-" * 70)
        
        # Run MPA algorithm
        best_solution, best_fitness, convergence = mpa(
            objective_func=benchmark['func'],
            dim=dim,
            bounds=benchmark['bounds'],
            pop_size=pop_size,
            max_iter=max_iter,
            P=P,
            FADs=FADs
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
    
    # Analysis note
    print("\n📝 PERFORMANCE ANALYSIS:")
    print("-" * 70)
    print("MPA's Three-Phase Strategy Performance:")
    print("  Phase 1 (0-33%):   Exploration using Brownian motion")
    print("  Phase 2 (33-66%):  Transition using Brownian + Lévy")
    print("  Phase 3 (66-100%): Exploitation using Lévy flight")
    print("\nKey Features:")
    print("  ✅ Adaptive CF parameter controls step size")
    print("  ✅ FADs effect maintains diversity")
    print("  ✅ Elite matrix preserves best solutions")
    print("  ✅ Lévy flight enables better exploration")
    print("-" * 70)
    
    print("\n💡 NOTE FOR PAPER:")
    print("-" * 70)
    print("MPA demonstrates effective balance between exploration and")
    print("exploitation through its three-phase velocity strategy,")
    print("combined with Lévy flight and FADs effect for diversity.")
    print("-" * 70)
    
    print("\n✅ MPA Algorithm testing complete!")
    print("=" * 70)
