import numpy as np

def rosenbrock(x, bounds=None):
    """
    Rosenbrock (f4) - Banana/Valley Function - Non-Separable Unimodal
    
    Mathematical Definition:
        f(x) = Σ[100(x[i+1] - x[i]²)² + (x[i] - 1)²], i=1 to D-1
    
    Properties:
        - Type: Unimodal (tek global minimum)
        - Global Minimum: f(1, 1, ..., 1) = 0
        - Search Domain: [-5, 10]^D or [-2.048, 2.048]^D (standart)
        - Difficulty: Zor - dar, kavisli vadi içinde optimum
        - Separable: Hayır (ardışık boyutlar bağımlı)
        - Valley Shape: Parabolic, very narrow and curved
        - Gradient: Nearly flat inside valley, steep on sides
        - Smoothness: Sürekli, türevlenebilir, ill-conditioned
    
    Usage in Benchmarking:
        - Tests exploitation in narrow valley
        - Tests algorithm's ability to follow curved path
        - Classic test for convergence speed in difficult terrain
        - Expected: Slow convergence even for good algorithms
        - Poor algorithms get stuck on valley walls
    
    Args:
        x (numpy.ndarray): D-boyutlu çözüm vektörü
        bounds (tuple, optional): (lower, upper) sınır değerleri. 
                                  None ise sınır kontrolü yapılmaz.
        
    Returns:
        float: Fonksiyon değeri
        
    References:
        - Rosenbrock, H. H. (1960). An automatic method for finding the 
          greatest or least value of a function. The Computer Journal, 3(3).
        - Dixon, L. C. W., & Szegö, G. P. (1978). The global optimization 
          problem: an introduction.
        - CEC benchmark suite
        - Jamil, M., & Yang, X. S. (2013). A literature survey of 
          benchmark functions for global optimisation problems.
    
    Example:
        >>> x = np.ones(30)
        >>> rosenbrock(x)
        0.0
        >>> x = np.array([1.0, 1.0, 1.0])
        >>> rosenbrock(x)
        0.0
        >>> x = np.zeros(3)
        >>> rosenbrock(x)  # Should be high: 100 + 1 + 100 + 1 = 202
        202.0
    
    Notes:
        - Named "banana function" due to its shape in 2D
        - Valley follows parabola: x[i+1] = x[i]²
        - Optimum lies in a long, narrow, parabolic valley
        - Easy to find valley, hard to converge to minimum
        - Excellent test for algorithms with adaptive step sizes
        - Classic benchmark since 1960s
        - One of the most famous test functions in optimization
    """
    n = len(x)
    
    # Ana hesaplama: f(x) = Σ[100(x[i+1] - x[i]²)² + (x[i] - 1)²]
    fitness = 0.0
    for i in range(n - 1):
        fitness += 100.0 * (x[i+1] - x[i]**2)**2 + (x[i] - 1.0)**2
    
    # Alternatif vectorized implementation (daha hızlı):
    # fitness = np.sum(100.0 * (x[1:] - x[:-1]**2)**2 + (x[:-1] - 1.0)**2)
    
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


def rosenbrock_vectorized(x, bounds=None):
    """
    Rosenbrock fonksiyonunun vectorized (optimize edilmiş) versiyonu.
    Büyük boyutlarda daha hızlı çalışır.
    """
    fitness = np.sum(100.0 * (x[1:] - x[:-1]**2)**2 + (x[:-1] - 1.0)**2)
    
    if bounds is not None:
        lower, upper = bounds
        penalty = 0.0
        violations = np.concatenate([
            np.maximum(0, lower - x),
            np.maximum(0, x - upper)
        ])
        penalty = 1e10 * np.sum(violations**2)
        fitness += penalty
    
    return fitness


def get_rosenbrock_info():
    """
    Rosenbrock fonksiyonu hakkında detaylı bilgi döndürür.
    
    Returns:
        dict: Fonksiyon özellikleri
    """
    return {
        'name': 'Rosenbrock Function (Banana Function)',
        'symbol': 'f4',
        'type': 'Unimodal',
        'separable': False,
        'differentiable': True,
        'scalable': True,
        'continuous': True,
        'convex': False,
        'global_minimum': 0.0,
        'global_minimum_location': 'x* = (1, 1, ..., 1)',
        'recommended_bounds': [-5, 10],  # or [-2.048, 2.048]
        'difficulty': 'Hard (despite being unimodal)',
        'tests': 'Convergence in narrow curved valley',
        'formula': "f(x) = Σ[100(x[i+1] - x[i]²)² + (x[i] - 1)²]",
        'valley_shape': 'Parabolic, narrow, curved',
        'dimensions_tested': [10, 30, 50, 100],
        'standard_runs': 30,
        'max_evaluations': '10,000 × D',
        'key_challenge': 'Following curved path in narrow valley',
        'special_features': 'Easy to find valley, hard to converge',
        'alternative_names': 'Banana Function, Rosenbrock Valley',
        'year_introduced': 1960,
        'conditioning': 'Ill-conditioned (ratio ~10⁶)'
    }


# ============================================================================
# TEST SUITE - Fonksiyonun doğruluğunu test eder
# ============================================================================

def run_tests():
    """Rosenbrock fonksiyonu için kapsamlı test suite'i"""
    
    print("=" * 70)
    print(" " * 18 + "ROSENBROCK FUNCTION TEST SUITE")
    print("=" * 70)
    
    test_results = []
    
    # Test 1: Global minimum (en önemli test)
    print("\n[Test 1] Global Minimum Test")
    print("-" * 70)
    for dim in [2, 3, 5, 10, 30, 50]:
        test_point = np.ones(dim)
        result = rosenbrock(test_point)
        passed = abs(result) < 1e-10
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  Dimension {dim:3d}: f(1^{dim}) = {result:.10e} ... {status}")
        test_results.append(passed)
    
    # Test 2: Bilinen değerler (manuel hesaplama)
    print("\n[Test 2] Known Values Test")
    print("-" * 70)
    
    # Test case 1: f([0, 0])
    # f([0,0]) = 100*(0-0²)² + (0-1)² = 100*0 + 1 = 1
    # But wait, for D=2: f([0,0]) = 100*(x₂-x₁²)² + (x₁-1)²
    #                              = 100*(0-0)² + (0-1)² = 0 + 1 = 1
    test_point1 = np.array([0.0, 0.0])
    result1 = rosenbrock(test_point1)
    expected1 = 1.0
    error1 = abs(result1 - expected1)
    passed1 = error1 < 1e-10
    status1 = "✓ PASS" if passed1 else "✗ FAIL"
    print(f"  f([0, 0]) = {result1:.10f} (Expected: {expected1}, Error: {error1:.2e}) ... {status1}")
    test_results.append(passed1)
    
    # Test case 2: f([0, 0, 0])
    # f([0,0,0]) = 100*(0-0²)² + (0-1)² + 100*(0-0²)² + (0-1)²
    #            = 0 + 1 + 0 + 1 = 2
    test_point2 = np.array([0.0, 0.0, 0.0])
    result2 = rosenbrock(test_point2)
    expected2 = 2.0
    error2 = abs(result2 - expected2)
    passed2 = error2 < 1e-10
    status2 = "✓ PASS" if passed2 else "✗ FAIL"
    print(f"  f([0, 0, 0]) = {result2:.10f} (Expected: {expected2}, Error: {error2:.2e}) ... {status2}")
    test_results.append(passed2)
    
    # Test case 3: f([1, 1, 0])
    # f([1,1,0]) = 100*(1-1²)² + (1-1)² + 100*(0-1²)² + (1-1)²
    #            = 100*0 + 0 + 100*1 + 0 = 100
    test_point3 = np.array([1.0, 1.0, 0.0])
    result3 = rosenbrock(test_point3)
    expected3 = 100.0
    error3 = abs(result3 - expected3)
    passed3 = error3 < 1e-10
    status3 = "✓ PASS" if passed3 else "✗ FAIL"
    print(f"  f([1, 1, 0]) = {result3:.10f} (Expected: {expected3}, Error: {error3:.2e}) ... {status3}")
    test_results.append(passed3)
    
    # Test case 4: 2D classic example
    test_point4 = np.array([2.0, 4.0])
    # f([2,4]) = 100*(4-2²)² + (2-1)² = 100*0 + 1 = 1
    result4 = rosenbrock(test_point4)
    expected4 = 1.0
    error4 = abs(result4 - expected4)
    passed4 = error4 < 1e-10
    status4 = "✓ PASS" if passed4 else "✗ FAIL"
    print(f"  f([2, 4]) = {result4:.10f} (Expected: {expected4}, lies on valley) ... {status4}")
    test_results.append(passed4)
    
    # Test 3: Valley path test (parabola x[i+1] = x[i]²)
    print("\n[Test 3] Valley Path Test")
    print("-" * 70)
    print("  Points on valley (x[i+1] = x[i]²) should have lower fitness:")
    
    valley_points = [
        np.array([1.0, 1.0]),           # Optimum
        np.array([0.5, 0.25]),          # On valley
        np.array([1.5, 2.25]),          # On valley
        np.array([0.8, 0.64]),          # On valley
    ]
    
    off_valley_points = [
        np.array([1.0, 2.0]),           # Off valley
        np.array([0.5, 0.5]),           # Off valley
        np.array([1.5, 1.5]),           # Off valley
    ]
    
    print("\n  On Valley:")
    valley_fitness = []
    for point in valley_points:
        fitness = rosenbrock(point)
        valley_fitness.append(fitness)
        print(f"    f({point}) = {fitness:.6f}")
    
    print("\n  Off Valley:")
    off_valley_fitness = []
    for point in off_valley_points:
        fitness = rosenbrock(point)
        off_valley_fitness.append(fitness)
        print(f"    f({point}) = {fitness:.6f}")
    
    avg_valley = np.mean(valley_fitness)
    avg_off = np.mean(off_valley_fitness)
    print(f"\n  Average on valley: {avg_valley:.6f}")
    print(f"  Average off valley: {avg_off:.6f}")
    print(f"  Valley is better: {'✓' if avg_valley < avg_off else '✗'}")
    
    # Test 4: Non-separability test
    print("\n[Test 4] Non-Separability Test")
    print("-" * 70)
    print("  Rosenbrock is non-separable (consecutive dimensions coupled)")
    
    # For non-separable: optimizing each dimension independently won't work
    x1 = np.array([1.0, 0.0, 1.0])
    x2 = np.array([1.0, 1.0, 0.0])
    x3 = np.array([1.0, 1.0, 1.0])
    
    f1 = rosenbrock(x1)
    f2 = rosenbrock(x2)
    f3 = rosenbrock(x3)
    
    print(f"  f([1, 0, 1]) = {f1:.6f}")
    print(f"  f([1, 1, 0]) = {f2:.6f}")
    print(f"  f([1, 1, 1]) = {f3:.6f} (optimal)")
    print(f"\n  Changing one dimension affects neighbors → Non-separable ✓")
    
    # Test 5: Ill-conditioning test
    print("\n[Test 5] Ill-Conditioning Test")
    print("-" * 70)
    print("  Testing convergence difficulty (steep valley walls):")
    
    # Points near optimum with small perturbations
    perturbations = [0.0, 0.001, 0.01, 0.1, 0.5]
    print("\n  Perturbation from optimal [1,1,...,1]:")
    for delta in perturbations:
        test_point = np.ones(5) + delta
        fitness = rosenbrock(test_point)
        print(f"    δ = {delta:5.3f}: f(1+δ) = {fitness:12.6f}")
    
    print("\n  → Small perturbations cause large fitness changes (ill-conditioned)")
    
    # Test 6: Vectorized vs loop implementation
    print("\n[Test 6] Vectorized Implementation Test")
    print("-" * 70)
    
    dim = 100
    x = np.random.uniform(-2, 2, dim)
    
    result_loop = rosenbrock(x)
    result_vectorized = rosenbrock_vectorized(x)
    
    error = abs(result_loop - result_vectorized)
    passed = error < 1e-10
    status = "✓ PASS" if passed else "✗ FAIL"
    
    print(f"  Loop version:       {result_loop:.10f}")
    print(f"  Vectorized version: {result_vectorized:.10f}")
    print(f"  Difference: {error:.2e} ... {status}")
    test_results.append(passed)
    
    # Test 7: Yüksek boyut testi
    print("\n[Test 7] High Dimensional Test")
    print("-" * 70)
    for dim in [100, 500, 1000]:
        # Global minimum
        test_point_opt = np.ones(dim)
        result_opt = rosenbrock(test_point_opt)
        passed_opt = abs(result_opt) < 1e-10
        
        # Random point (should be >> 0)
        test_point_rand = np.random.uniform(-2, 2, dim)
        result_rand = rosenbrock(test_point_rand)
        
        status = "✓ PASS" if passed_opt else "✗ FAIL"
        print(f"  Dimension {dim:4d}:")
        print(f"    f(1^{dim}) = {result_opt:.10e} ... {status}")
        print(f"    f(random) = {result_rand:.6e} (>> 0 expected)")
        test_results.append(passed_opt)
    
    # Test 8: Sınır kontrolü testi
    print("\n[Test 8] Boundary Handling Test")
    print("-" * 70)
    test_point_in = np.array([4.0, -4.0, 0.0])
    test_point_out = np.array([12.0, -12.0, 0.0])
    
    result_in = rosenbrock(test_point_in, bounds=(-5, 10))
    result_out = rosenbrock(test_point_out, bounds=(-5, 10))
    
    print(f"  In bounds [4, -4, 0]:    f(x) = {result_in:.6e}")
    print(f"  Out of bounds [12,-12,0]: f(x) = {result_out:.6e}")
    
    passed = result_out > result_in * 1000
    status = "✓ PASS" if passed else "✗ FAIL"
    print(f"  Penalty applied correctly: {status}")
    test_results.append(passed)
    
    # Test 9: Performance comparison
    print("\n[Test 9] Performance Test (Loop vs Vectorized)")
    print("-" * 70)
    import time
    
    dim = 10000
    x = np.random.uniform(-2, 2, dim)
    
    # Loop version
    start = time.time()
    for _ in range(100):
        result = rosenbrock(x)
    time_loop = time.time() - start
    
    # Vectorized version
    start = time.time()
    for _ in range(100):
        result = rosenbrock_vectorized(x)
    time_vec = time.time() - start
    
    print(f"  Loop version:       {time_loop:.4f} seconds (100 evals)")
    print(f"  Vectorized version: {time_vec:.4f} seconds (100 evals)")
    print(f"  Speedup: {time_loop/time_vec:.2f}x")
    
    status = "✓ FAST" if time_vec < 1.0 else "⚠ SLOW"
    print(f"  Performance: {status}")
    
    # Test 10: Comparison with other unimodal functions
    print("\n[Test 10] Comparison with Sphere (Unimodal Baseline)")
    print("-" * 70)
    
    from sphere import sphere
    
    dim = 30
    np.random.seed(42)
    test_points = [np.random.uniform(-2, 2, dim) for _ in range(5)]
    
    print(f"\n  {'Point':<8} {'Rosenbrock':>14} {'Sphere':>12} {'Ratio':>10}")
    print("  " + "-" * 48)
    for i, point in enumerate(test_points):
        rosen = rosenbrock(point)
        sph = sphere(point)
        ratio = rosen / sph if sph > 0 else float('inf')
        print(f"  Point {i+1:2d} {rosen:14.2f} {sph:12.2f} {ratio:10.2f}x")
    
    print("\n  → Rosenbrock typically harder than Sphere despite being unimodal")
    
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
    info = get_rosenbrock_info()
    
    print("\n" + "=" * 70)
    print(" " * 18 + "ROSENBROCK FUNCTION INFORMATION")
    print("=" * 70)
    
    for key, value in info.items():
        print(f"{key.replace('_', ' ').title():25s}: {value}")
    
    print("=" * 70)
    
    # Ek bilgiler
    print("\n📊 ROSENBROCK CHARACTERISTICS:")
    print("-" * 70)
    print("  • Valley shape: Parabolic curve x[i+1] = x[i]²")
    print("  • Valley width: Very narrow (high curvature)")
    print("  • Gradient inside valley: Nearly flat → slow convergence")
    print("  • Gradient on valley walls: Very steep → easy to find valley")
    print("  • Conditioning number: ~10⁶ (ill-conditioned)")
    print("  • Historical significance: One of oldest test functions (1960)")
    print("  • Alternative name: 'Banana Function' (due to 2D contour shape)")
    print("  • Challenge: Easy to find valley, hard to follow it to optimum")
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
    from rosenbrock import rosenbrock
    
    # 30 boyutlu random çözüm
    x = np.random.uniform(-5, 10, 30)
    fitness = rosenbrock(x)
    print(f"Fitness: {fitness}")
    
    # Sınır kontrolü ile
    fitness_bounded = rosenbrock(x, bounds=(-5, 10))
    print(f"Fitness (bounded): {fitness_bounded}")
    
    # Vectorized version (daha hızlı)
    from rosenbrock import rosenbrock_vectorized
    fitness_fast = rosenbrock_vectorized(x)
    """)
    
    # Gerçek örnek
    print("\nActual example:")
    print("-" * 70)
    x_random = np.random.uniform(-5, 10, 30)
    fitness_random = rosenbrock(x_random)
    print(f"Random solution (D=30):  f(x) = {fitness_random:.6e}")
    
    x_optimal = np.ones(30)
    fitness_optimal = rosenbrock(x_optimal)
    print(f"Optimal solution (D=30): f(x*) = {fitness_optimal:.10e}")
    
    # Valley point example
    x_valley = np.array([0.5, 0.25, 0.0625, 0.00390625])
    fitness_valley = rosenbrock(x_valley)
    print(f"On valley path:          f(x) = {fitness_valley:.6f}")
    
    print("\n" + "=" * 70)
    print("✅ rosenbrock.py is ready for your benchmark study!")
    print("=" * 70)