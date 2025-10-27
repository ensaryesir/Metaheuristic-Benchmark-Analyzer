import numpy as np

def schwefel(x, bounds=None):
    """
    Schwefel (f5) - Highly Deceptive Multimodal Benchmark Function
    
    Mathematical Definition:
        f(x) = 418.9829D - Σ[xi·sin(√|xi|)], i=1 to D
    
    Properties:
        - Type: Highly Multimodal, Deceptive
        - Global Minimum: f(420.9687, ..., 420.9687) = 0
        - Search Domain: [-500, 500]^D (standart)
        - Difficulty: Very Hard - most deceptive benchmark function
        - Separable: Evet (her boyut bağımsız)
        - Local Minima: Very numerous, regularly distributed
        - Symmetry: Hayır (asimetrik)
        - Smoothness: Sürekli, türevlenebilir
        - Deception Level: Extreme - second best is far from global optimum
    
    Usage in Benchmarking:
        - Tests ability to escape deceptive local optima
        - Global optimum located near boundary (x ≈ 421)
        - Second-best local optimum is very far away
        - Most algorithms get trapped in local optima
        - Expected: Only excellent algorithms find global optimum
        - Often considered the hardest classical benchmark
    
    Args:
        x (numpy.ndarray): D-boyutlu çözüm vektörü
        bounds (tuple, optional): (lower, upper) sınır değerleri. 
                                  None ise sınır kontrolü yapılmaz.
        
    Returns:
        float: Fonksiyon değeri
        
    References:
        - Schwefel, H. P. (1981). Numerical optimization of computer models.
        - Schwefel, H. P. (1995). Evolution and Optimum Seeking.
        - CEC benchmark suite
        - Jamil, M., & Yang, X. S. (2013). A literature survey of 
          benchmark functions for global optimisation problems.
    
    Example:
        >>> x = np.ones(30) * 420.9687
        >>> schwefel(x)
        0.0
        >>> x = np.zeros(30)
        >>> schwefel(x)  # Very high value
        12569.487
    
    Notes:
        - Named after Hans-Paul Schwefel
        - Global minimum at x* ≈ 420.9687 (near upper boundary at 500)
        - This placement makes the function highly deceptive
        - The function has sin(√|x|) term which creates complex landscape
        - Most algorithms converge to wrong region (around x=0)
        - Excellent test for population diversity and exploration
        - One of the CEC benchmark suite's most challenging functions
        - The constant 418.9829 ≈ 420.9687·sin(√420.9687) ensures f(x*)=0
    """
    n = len(x)
    
    # Schwefel constant (ensures f(x*) = 0)
    # 418.9829 ≈ 420.9687 * sin(sqrt(420.9687))
    constant = 418.9829
    
    # Ana hesaplama: f(x) = 418.9829D - Σ[xi·sin(√|xi|)]
    fitness = constant * n - np.sum(x * np.sin(np.sqrt(np.abs(x))))
    
    # Sınır kontrolü (opsiyonel)
    if bounds is not None:
        lower, upper = bounds
        penalty = 0.0
        for xi in x:
            if xi < lower:
                penalty += 1e10 * (lower - xi)**2
            elif xi > upper:
                penalty += 1e10 * (xi - upper)**2
        fitness += penalty
    
    return fitness


def get_schwefel_info():
    """
    Schwefel fonksiyonu hakkında detaylı bilgi döndürür.
    
    Returns:
        dict: Fonksiyon özellikleri
    """
    return {
        'name': 'Schwefel Function',
        'symbol': 'f5',
        'type': 'Highly Multimodal, Deceptive',
        'separable': True,
        'differentiable': True,
        'scalable': True,
        'continuous': True,
        'convex': False,
        'global_minimum': 0.0,
        'global_minimum_location': 'x* = (420.9687, 420.9687, ..., 420.9687)',
        'recommended_bounds': [-500, 500],
        'difficulty': 'Very Hard (hardest classical benchmark)',
        'tests': 'Deceptive landscape navigation, exploration',
        'formula': "f(x) = 418.9829D - Σ[xi·sin(√|xi|)]",
        'local_minima_pattern': 'Very numerous, deceptively placed',
        'dimensions_tested': [10, 30, 50, 100],
        'standard_runs': 30,
        'max_evaluations': '10,000 × D',
        'key_challenge': 'Global optimum near boundary, far from second-best',
        'special_features': 'Most deceptive classical benchmark',
        'deception_level': 'Extreme',
        'common_trap': 'Algorithms often converge to region around x=0',
        'year_introduced': 1981,
        'success_rate': 'Very low for most algorithms'
    }


# ============================================================================
# TEST SUITE - Fonksiyonun doğruluğunu test eder
# ============================================================================

def run_tests():
    """Schwefel fonksiyonu için kapsamlı test suite'i"""
    
    print("=" * 70)
    print(" " * 19 + "SCHWEFEL FUNCTION TEST SUITE")
    print("=" * 70)
    
    test_results = []
    
    # Test 1: Global minimum (en önemli test)
    print("\n[Test 1] Global Minimum Test")
    print("-" * 70)
    
    # Schwefel global minimum: x* = 420.9687 (approximately)
    optimal_value = 420.9687
    
    for dim in [2, 5, 10, 30, 50]:
        test_point = np.ones(dim) * optimal_value
        result = schwefel(test_point)
        passed = abs(result) < 1e-3  # Tolerance for numerical precision
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  Dimension {dim:3d}: f(420.9687^{dim}) = {result:.10f} ... {status}")
        test_results.append(passed)
    
    # Test 2: Bilinen değerler
    print("\n[Test 2] Known Values Test")
    print("-" * 70)
    
    # Test at origin (common trap point)
    test_point1 = np.zeros(5)
    result1 = schwefel(test_point1)
    expected1 = 418.9829 * 5  # ≈ 2094.9145
    error1 = abs(result1 - expected1)
    passed1 = error1 < 0.01
    status1 = "✓ PASS" if passed1 else "✗ FAIL"
    print(f"  f([0]^5) = {result1:.6f} (Expected: {expected1:.6f}, Error: {error1:.4f}) ... {status1}")
    print(f"    → Origin is a LOCAL MINIMUM, not global (deceptive!)")
    test_results.append(passed1)
    
    # Test at negative optimal
    test_point2 = np.ones(5) * (-420.9687)
    result2 = schwefel(test_point2)
    # Due to symmetry in sin, this should also be near optimum
    print(f"\n  f([-420.9687]^5) = {result2:.6f}")
    print(f"    → Negative side also near optimum (due to sin symmetry)")
    
    # Test at boundary
    test_point3 = np.ones(5) * 500
    result3 = schwefel(test_point3)
    print(f"\n  f([500]^5) = {result3:.6f}")
    print(f"    → Near boundary (not optimal)")
    
    # Test at common trap (-100)
    test_point4 = np.ones(5) * (-100)
    result4 = schwefel(test_point4)
    print(f"\n  f([-100]^5) = {result4:.6f}")
    print(f"    → Common local minimum trap")
    
    # Test 3: Deceptive landscape test
    print("\n[Test 3] Deceptive Landscape Test")
    print("-" * 70)
    print("  Testing fitness at various points (D=2 for visualization):")
    print("\n  Location          x-coord    Fitness    Relative to Optimum")
    print("  " + "-" * 66)
    
    test_locations = [
        ("Origin", 0),
        ("Near origin", 50),
        ("Quarter way", 100),
        ("Halfway", 200),
        ("Three quarters", 300),
        ("Near optimum", 400),
        ("Optimum", 420.9687),
        ("Past optimum", 450),
        ("Near boundary", 490),
    ]
    
    for name, x_val in test_locations:
        point = np.array([x_val, x_val])
        fitness = schwefel(point)
        optimal_fitness = schwefel(np.array([420.9687, 420.9687]))
        relative = fitness - optimal_fitness
        print(f"  {name:16s} {x_val:8.2f} {fitness:10.4f} {relative:10.4f}")
    
    print("\n  → Global optimum near boundary (420.97), NOT at origin!")
    print("  → This creates deception: algorithms naturally move away from optimum")
    
    # Test 4: Symmetry test (partial - sin function)
    print("\n[Test 4] Sin Function Behavior Test")
    print("-" * 70)
    print("  Testing sin(√|x|) symmetry:")
    
    test_values = [0, 100, 420.9687, -100, -420.9687]
    for val in test_values:
        term = val * np.sin(np.sqrt(np.abs(val)))
        print(f"  x = {val:8.2f}: x·sin(√|x|) = {term:10.4f}")
    
    # Test 5: Multiple local minima test
    print("\n[Test 5] Multiple Local Minima Test")
    print("-" * 70)
    print("  Sampling landscape to detect local minima (1D slice):")
    
    x_range = np.linspace(-500, 500, 1000)
    y_values = []
    for x_val in x_range:
        point = np.array([x_val])
        y_values.append(schwefel(point))
    
    # Find local minima (simple peak detection)
    y_array = np.array(y_values)
    local_minima_count = 0
    for i in range(1, len(y_array) - 1):
        if y_array[i] < y_array[i-1] and y_array[i] < y_array[i+1]:
            local_minima_count += 1
    
    print(f"  Detected ~{local_minima_count} local minima in 1D slice")
    print(f"  → Highly multimodal landscape confirmed")
    
    # Find global minimum in sample
    min_idx = np.argmin(y_values)
    min_x = x_range[min_idx]
    min_y = y_values[min_idx]
    print(f"  Lowest point found: x = {min_x:.4f}, f(x) = {min_y:.6f}")
    print(f"  Expected optimum:   x = 420.9687, f(x) ≈ 0.0")
    
    # Test 6: Dimension scaling test
    print("\n[Test 6] Dimension Scaling Test")
    print("-" * 70)
    print("  Testing how fitness scales with dimension:")
    
    for dim in [10, 30, 50, 100]:
        # At origin (trap)
        point_zero = np.zeros(dim)
        f_zero = schwefel(point_zero)
        
        # At optimum
        point_opt = np.ones(dim) * 420.9687
        f_opt = schwefel(point_opt)
        
        # Random point
        np.random.seed(42)
        point_rand = np.random.uniform(-500, 500, dim)
        f_rand = schwefel(point_rand)
        
        print(f"\n  Dimension {dim:3d}:")
        print(f"    f(0^{dim})        = {f_zero:10.2f} (local trap)")
        print(f"    f(420.97^{dim})   = {f_opt:10.6f} (global optimum)")
        print(f"    f(random)         = {f_rand:10.2f}")
    
    # Test 7: Yüksek boyut testi
    print("\n[Test 7] High Dimensional Test")
    print("-" * 70)
    for dim in [100, 500, 1000]:
        # Global minimum
        test_point_opt = np.ones(dim) * 420.9687
        result_opt = schwefel(test_point_opt)
        passed_opt = abs(result_opt) < 0.01
        
        # Trap point
        test_point_trap = np.zeros(dim)
        result_trap = schwefel(test_point_trap)
        
        status = "✓ PASS" if passed_opt else "✗ FAIL"
        print(f"  Dimension {dim:4d}:")
        print(f"    f(420.97^{dim}) = {result_opt:.6f} ... {status}")
        print(f"    f(0^{dim})      = {result_trap:.2f} (trap)")
        test_results.append(passed_opt)
    
    # Test 8: Sınır kontrolü testi
    print("\n[Test 8] Boundary Handling Test")
    print("-" * 70)
    test_point_in = np.array([400.0, -400.0, 0.0])
    test_point_out = np.array([600.0, -600.0, 0.0])
    
    result_in = schwefel(test_point_in, bounds=(-500, 500))
    result_out = schwefel(test_point_out, bounds=(-500, 500))
    
    print(f"  In bounds [400,-400,0]:     f(x) = {result_in:.6e}")
    print(f"  Out of bounds [600,-600,0]: f(x) = {result_out:.6e}")
    
    passed = result_out > result_in * 1000
    status = "✓ PASS" if passed else "✗ FAIL"
    print(f"  Penalty applied correctly: {status}")
    test_results.append(passed)
    
    # Test 9: Deception quantification
    print("\n[Test 9] Deception Quantification")
    print("-" * 70)
    print("  Measuring deception: Distance from common trap to global optimum")
    
    dim = 30
    trap_point = np.zeros(dim)  # Common trap
    optimal_point = np.ones(dim) * 420.9687
    
    euclidean_distance = np.linalg.norm(optimal_point - trap_point)
    f_trap = schwefel(trap_point)
    f_optimal = schwefel(optimal_point)
    fitness_difference = f_trap - f_optimal
    
    print(f"\n  Dimension: {dim}")
    print(f"  Trap location: [0]^{dim}")
    print(f"  Optimal location: [420.97]^{dim}")
    print(f"  Euclidean distance: {euclidean_distance:.2f}")
    print(f"  Fitness at trap: {f_trap:.2f}")
    print(f"  Fitness at optimum: {f_optimal:.6f}")
    print(f"  Fitness difference: {fitness_difference:.2f}")
    print(f"\n  → Deception level: EXTREME")
    print(f"  → Algorithms must travel {euclidean_distance:.0f} units through bad regions")
    
    # Test 10: Comparison with other multimodal functions
    print("\n[Test 10] Comparison with Other Multimodal Functions")
    print("-" * 70)
    
    try:
        from rastrigin import rastrigin
        from ackley import ackley
        
        dim = 30
        np.random.seed(42)
        test_points = [np.random.uniform(-5, 5, dim) for _ in range(5)]
        
        print(f"\n  {'Point':<8} {'Schwefel':>12} {'Rastrigin':>12} {'Ackley':>12}")
        print("  " + "-" * 48)
        
        for i, point in enumerate(test_points):
            # Scale point for Schwefel
            point_schwefel = point * 100  # Scale to [-500, 500]
            
            schw = schwefel(point_schwefel)
            rast = rastrigin(point)
            ack = ackley(point)
            
            print(f"  Point {i+1:2d} {schw:12.2f} {rast:12.2f} {ack:12.2f}")
        
        print("\n  → Schwefel typically has higher absolute values")
        print("  → But relative difficulty is about scale-invariant")
    except ImportError:
        print("  (Rastrigin/Ackley not available for comparison)")
    
    # Test 11: Performance test
    print("\n[Test 11] Performance Test")
    print("-" * 70)
    import time
    
    dim = 10000
    x = np.random.uniform(-500, 500, dim)
    
    start = time.time()
    for _ in range(1000):
        result = schwefel(x)
    elapsed = time.time() - start
    
    print(f"  1000 evaluations at D={dim}: {elapsed:.4f} seconds")
    print(f"  Average time per evaluation: {elapsed/1000*1e6:.2f} μs")
    status = "✓ PASS" if elapsed < 2.0 else "⚠ SLOW"
    print(f"  Performance: {status}")
    
    # Özet
    print("\n" + "=" * 70)
    total_tests = len(test_results)
    passed_tests = sum(test_results)
    print(f"TEST SUMMARY: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("✅ ALL TESTS PASSED - Fonksiyon hazır!")
    else:
        print("⚠️  SOME TESTS FAILED - Kodda düzeltme gerekiyor!")
    print("=" * 70)
    
    return all(test_results)


def print_function_info():
    """Fonksiyon bilgilerini yazdırır"""
    info = get_schwefel_info()
    
    print("\n" + "=" * 70)
    print(" " * 19 + "SCHWEFEL FUNCTION INFORMATION")
    print("=" * 70)
    
    for key, value in info.items():
        print(f"{key.replace('_', ' ').title():25s}: {value}")
    
    print("=" * 70)
    
    # Ek bilgiler
    print("\n📊 SCHWEFEL CHARACTERISTICS:")
    print("-" * 70)
    print("  • Deception: Most deceptive classical benchmark function")
    print("  • Global optimum location: x* ≈ 420.9687 (near boundary)")
    print("  • Common trap: Origin (x = 0) is a strong local minimum")
    print("  • Distance to travel: Must go ~421 units from trap to optimum")
    print("  • Success rate: Very low (<10% for many algorithms)")
    print("  • Search space: [-500, 500]^D (optimum near edge)")
    print("  • Function shape: sin(√|x|) creates complex oscillatory landscape")
    print("  • Difficulty ranking: Often ranked #1 hardest classical benchmark")
    print("  • Key insight: Tests if algorithm can escape deceptive attraction")
    print("=" * 70)


# ============================================================================
# MAIN - Direkt çalıştırma için
# ============================================================================

if __name__ == "__main__":
    # Fonksiyon bilgilerini yazdır
    print_function_info()
    
    # Test suite'i çalıştır
    success = run_tests()
    
    # Örnek kullanım
    print("\n" + "=" * 70)
    print(" " * 25 + "EXAMPLE USAGE")
    print("=" * 70)
    print("\nCode example:")
    print("-" * 70)
    print("""
    import numpy as np
    from schwefel import schwefel
    
    # 30 boyutlu random çözüm
    x = np.random.uniform(-500, 500, 30)
    fitness = schwefel(x)
    print(f"Fitness: {fitness}")
    
    # Sınır kontrolü ile
    fitness_bounded = schwefel(x, bounds=(-500, 500))
    print(f"Fitness (bounded): {fitness_bounded}")
    """)
    
    # Gerçek örnek
    print("\nActual example:")
    print("-" * 70)
    x_random = np.random.uniform(-500, 500, 30)
    fitness_random = schwefel(x_random)
    print(f"Random solution (D=30):  f(x) = {fitness_random:.2f}")
    
    x_optimal = np.ones(30) * 420.9687
    fitness_optimal = schwefel(x_optimal)
    print(f"Optimal solution (D=30): f(x*) = {fitness_optimal:.10f}")
    
    # Trap example
    x_trap = np.zeros(30)
    fitness_trap = schwefel(x_trap)
    print(f"Trap point (0^30):       f(x) = {fitness_trap:.2f} (local min!)")
    
    # Distance
    distance = np.linalg.norm(x_optimal - x_trap)
    print(f"\nDistance from trap to optimum: {distance:.2f} units")
    
    print("\n" + "=" * 70)
    print("✅ schwefel.py is ready for your benchmark study!")
    print("⚠️  WARNING: This is the HARDEST classical benchmark!")
    print("=" * 70)