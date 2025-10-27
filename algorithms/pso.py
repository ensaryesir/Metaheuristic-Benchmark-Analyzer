import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np

"""
Particle Swarm Optimization (PSO) Algorithm
===========================================

📚 ORIGINAL REFERENCES:
-----------------------
Primary Paper:
    Kennedy, J., & Eberhart, R. (1995). 
    "Particle swarm optimization."
    Proceedings of ICNN'95-international conference on neural networks, 
    IEEE, Vol. 4, pp. 1942-1948.

Inertia Weight:
    Shi, Y., & Eberhart, R. (1998).
    "A modified particle swarm optimizer."
    IEEE international conference on evolutionary computation proceedings, 
    pp. 69-73.

Constriction Factor:
    Clerc, M., & Kennedy, J. (2002).
    "The particle swarm-explosion, stability, and convergence in a 
    multidimensional complex space."
    IEEE transactions on Evolutionary Computation, 6(1), 58-73.

📖 ALGORITHM DESCRIPTION:
-------------------------
PSO simulates the social behavior of bird flocking or fish schooling.
Each particle represents a potential solution and moves through the 
search space influenced by:
1. Its own best known position (cognitive component)
2. The swarm's best known position (social component)
3. Its current velocity (inertia)

🔬 MATHEMATICAL FORMULATION:
----------------------------
Velocity Update:
    v[i](t+1) = w·v[i](t) + c1·r1·(pbest[i] - x[i](t)) + c2·r2·(gbest - x[i](t))

Position Update:
    x[i](t+1) = x[i](t) + v[i](t+1)

Where:
    w  = inertia weight (balances exploration vs exploitation)
    c1 = cognitive coefficient (particle's confidence in itself)
    c2 = social coefficient (particle's confidence in swarm)
    r1, r2 = random vectors in [0,1]^D for each dimension
    pbest[i] = personal best position of particle i
    gbest = global best position found by swarm

⚙️ PARAMETER RECOMMENDATIONS:
-----------------------------
Standard PSO (Shi & Eberhart, 1998):
    - w: 0.9 → 0.4 (linear decrease)
    - c1 = 2.0 (cognitive coefficient)
    - c2 = 2.0 (social coefficient)
    - pop_size: 20-40 (typically 30)
    - v_max: 0.5-1.0 × (ub - lb)

Exploration-focused:
    - w: 0.9 (high inertia)
    - c1 = 2.5, c2 = 0.5

Exploitation-focused:
    - w: 0.4 (low inertia)
    - c1 = 0.5, c2 = 2.5

💪 STRENGTHS:
-------------
- Simple implementation
- Few parameters to tune
- Fast convergence
- Good for continuous optimization
- Effective for unimodal functions

⚠️ WEAKNESSES:
--------------
- Premature convergence in multimodal functions
- Sensitive to parameter settings
- May get trapped in local optima
- Performance degrades in high dimensions

📊 COMPUTATIONAL COMPLEXITY:
---------------------------
Time: O(pop_size × max_iter × dim)
Space: O(pop_size × dim)

Args:
    objective_func (callable): Objective function to minimize
    dim (int): Problem dimensionality
    bounds (tuple or list): Search space bounds as (lb, ub) or [(lb, ub)] × dim
    pop_size (int): Population size (number of particles)
    max_iter (int): Maximum number of iterations
    w (float): Initial inertia weight (default: 0.9)
    w_min (float): Minimum inertia weight (default: 0.4)
    c1 (float): Cognitive coefficient (default: 2.0)
    c2 (float): Social coefficient (default: 2.0)
    v_max_factor (float): Velocity limit as fraction of bounds (default: 0.5)
    
Returns:
    tuple: (best_position, best_fitness, convergence_curve)
        - best_position: Best solution found
        - best_fitness: Fitness of best solution
        - convergence_curve: Best fitness at each iteration
"""

def pso(objective_func, dim, bounds, pop_size=30, max_iter=1000, 
        w=0.9, w_min=0.4, c1=2.0, c2=2.0, v_max_factor=0.5):
    """
    Particle Swarm Optimization - Standard Implementation
    
    This implementation follows the canonical PSO algorithm with
    linearly decreasing inertia weight as proposed by Shi & Eberhart (1998).
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
    
    # Velocity limits (typically 50%-100% of position range)
    # Shi & Eberhart suggest v_max = 0.5 * (ub - lb)
    v_max = v_max_factor * (ub - lb)
    v_min = -v_max
    
    # ========================================================================
    # STEP 2: POPULATION INITIALIZATION
    # ========================================================================
    
    # Initialize positions uniformly in search space
    positions = np.random.uniform(lb, ub, (pop_size, dim))
    
    # Initialize velocities uniformly within velocity limits
    velocities = np.random.uniform(v_min, v_max, (pop_size, dim))
    
    # Initialize personal best positions and scores
    pbest_positions = positions.copy()
    pbest_scores = np.array([objective_func(pos) for pos in positions])
    
    # Initialize global best
    gbest_idx = np.argmin(pbest_scores)
    gbest_position = pbest_positions[gbest_idx].copy()
    gbest_score = pbest_scores[gbest_idx]
    
    # Convergence tracking
    convergence_curve = np.zeros(max_iter)
    
    # ========================================================================
    # STEP 3: MAIN OPTIMIZATION LOOP
    # ========================================================================
    
    for iteration in range(max_iter):
        
        # Update inertia weight (linear decrease from w to w_min)
        current_w = w - (w - w_min) * iteration / max_iter
        
        # Generate random matrices for cognitive and social components
        # IMPORTANT: Different random values for each particle AND each dimension
        r1 = np.random.rand(pop_size, dim)  # Cognitive random matrix
        r2 = np.random.rand(pop_size, dim)  # Social random matrix
        
        # ====================================================================
        # VELOCITY UPDATE (Vectorized)
        # ====================================================================
        # v(t+1) = w·v(t) + c1·r1·(pbest - x) + c2·r2·(gbest - x)
        
        cognitive_component = c1 * r1 * (pbest_positions - positions)
        social_component = c2 * r2 * (gbest_position - positions)
        
        velocities = (current_w * velocities + 
                     cognitive_component + 
                     social_component)
        
        # Apply velocity limits (clamping)
        velocities = np.clip(velocities, v_min, v_max)
        
        # ====================================================================
        # POSITION UPDATE (Vectorized)
        # ====================================================================
        # x(t+1) = x(t) + v(t+1)
        
        positions = positions + velocities
        
        # ====================================================================
        # BOUNDARY HANDLING
        # ====================================================================
        # When particle hits boundary, reflect velocity (damping approach)
        
        # Check which particles are out of bounds
        out_of_lower = positions < lb
        out_of_upper = positions > ub
        
        # Reflect velocities for out-of-bounds particles
        velocities[out_of_lower] = -0.5 * velocities[out_of_lower]
        velocities[out_of_upper] = -0.5 * velocities[out_of_upper]
        
        # Clamp positions to bounds
        positions = np.clip(positions, lb, ub)
        
        # ====================================================================
        # FITNESS EVALUATION
        # ====================================================================
        
        for i in range(pop_size):
            current_score = objective_func(positions[i])
            
            # Update personal best
            if current_score < pbest_scores[i]:
                pbest_scores[i] = current_score
                pbest_positions[i] = positions[i].copy()
                
                # Update global best
                if current_score < gbest_score:
                    gbest_score = current_score
                    gbest_position = positions[i].copy()
        
        # Record convergence
        convergence_curve[iteration] = gbest_score
        
        # Optional: Progress indicator (every 10% of iterations)
        if (iteration + 1) % (max_iter // 10) == 0 or iteration == 0:
            print(f"Iter {iteration + 1:4d}/{max_iter}: Best = {gbest_score:.6e}, w = {current_w:.3f}")
    
    return gbest_position, gbest_score, convergence_curve


def pso_with_early_stopping(objective_func, dim, bounds, pop_size=30, max_iter=1000,
                           w=0.9, w_min=0.4, c1=2.0, c2=2.0, v_max_factor=0.5,
                           tolerance=1e-8, patience=50):
    """
    PSO with Early Stopping - Extended Version
    
    Includes early stopping mechanism to terminate when convergence is detected.
    Useful for saving computational resources when solution is found.
    
    Additional Args:
        tolerance (float): Minimum fitness improvement to continue
        patience (int): Number of iterations without improvement before stopping
    """
    
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
    
    v_max = v_max_factor * (ub - lb)
    v_min = -v_max
    
    # Initialize
    positions = np.random.uniform(lb, ub, (pop_size, dim))
    velocities = np.random.uniform(v_min, v_max, (pop_size, dim))
    
    pbest_positions = positions.copy()
    pbest_scores = np.array([objective_func(pos) for pos in positions])
    
    gbest_idx = np.argmin(pbest_scores)
    gbest_position = pbest_positions[gbest_idx].copy()
    gbest_score = pbest_scores[gbest_idx]
    
    convergence_curve = np.zeros(max_iter)
    no_improvement_count = 0
    
    # Main loop
    for iteration in range(max_iter):
        current_w = w - (w - w_min) * iteration / max_iter
        
        r1 = np.random.rand(pop_size, dim)
        r2 = np.random.rand(pop_size, dim)
        
        cognitive_component = c1 * r1 * (pbest_positions - positions)
        social_component = c2 * r2 * (gbest_position - positions)
        
        velocities = current_w * velocities + cognitive_component + social_component
        velocities = np.clip(velocities, v_min, v_max)
        
        positions = positions + velocities
        
        out_of_lower = positions < lb
        out_of_upper = positions > ub
        velocities[out_of_lower] = -0.5 * velocities[out_of_lower]
        velocities[out_of_upper] = -0.5 * velocities[out_of_upper]
        positions = np.clip(positions, lb, ub)
        
        for i in range(pop_size):
            current_score = objective_func(positions[i])
            
            if current_score < pbest_scores[i]:
                pbest_scores[i] = current_score
                pbest_positions[i] = positions[i].copy()
                
                if current_score < gbest_score:
                    gbest_score = current_score
                    gbest_position = positions[i].copy()
        
        convergence_curve[iteration] = gbest_score
        
        # Early stopping check
        if iteration > 0:
            improvement = abs(convergence_curve[iteration-1] - convergence_curve[iteration])
            if improvement < tolerance:
                no_improvement_count += 1
            else:
                no_improvement_count = 0
            
            if no_improvement_count >= patience:
                print(f"Early stopping at iteration {iteration + 1} (no improvement for {patience} iterations)")
                convergence_curve = convergence_curve[:iteration+1]
                break
        
        if (iteration + 1) % (max_iter // 10) == 0 or iteration == 0:
            print(f"Iter {iteration + 1:4d}/{max_iter}: Best = {gbest_score:.6e}")
    
    return gbest_position, gbest_score, convergence_curve


# ============================================================================
# TEST CODE - Comprehensive Benchmark Testing
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print(" " * 15 + "PSO ALGORITHM - BENCHMARK TESTS")
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
    
    # PSO-specific parameters (standard settings)
    w = 0.9           # Initial inertia weight
    w_min = 0.4       # Final inertia weight
    c1 = 2.0          # Cognitive coefficient
    c2 = 2.0          # Social coefficient
    v_max_factor = 0.5  # Velocity limit factor
    
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
    print(f"  Inertia Weight: {w} → {w_min}")
    print(f"  Cognitive Coeff (c1): {c1}")
    print(f"  Social Coeff (c2): {c2}")
    print(f"  Velocity Limit Factor: {v_max_factor}")
    print()
    
    results_summary = []
    
    for benchmark in benchmarks_to_test:
        print("\n" + "=" * 70)
        print(f"Testing: {benchmark['name']} Function")
        print("=" * 70)
        print(f"Search Space: {benchmark['bounds']}")
        print(f"Global Optimum: {benchmark['optimal']}")
        print("-" * 70)
        
        # Run PSO
        best_solution, best_fitness, convergence = pso(
            objective_func=benchmark['func'],
            dim=dim,
            bounds=benchmark['bounds'],
            pop_size=pop_size,
            max_iter=max_iter,
            w=w,
            w_min=w_min,
            c1=c1,
            c2=c2,
            v_max_factor=v_max_factor
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
    print("✅ PSO Algorithm testing complete!")
    print("=" * 70)
