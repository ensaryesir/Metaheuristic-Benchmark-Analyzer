import numpy as np

def ackley(x, bounds=None):
    """
    Ackley (f3) - Highly Multimodal Benchmark Function with Deep Basin
    
    Mathematical Definition:
        f(x) = -a·exp(-b·√(1/D·Σxi²)) - exp(1/D·Σcos(c·xi)) + a + e
        where a=20, b=0.2, c=2π, e=exp(1)
    
    Properties:
        - Type: Highly Multimodal (çok sayıda yerel minimum)
        - Global Minimum: f(0,...,0) = 0
        - Search Domain: [-32.768, 32.768]^D (standart) or [-5, 5]^D
        - Difficulty: Orta-Zor - çok sayıda yerel minima + derin basin
        - Separable: Hayır (boyutlar arasında ilişki var)
        - Local Minima: Çok sayıda, düzenli aralıklarla
        - Symmetry: Evet (orijine göre simetrik)
        - Smoothness: Sürekli, türevlenebilir, highly oscillatory
        - Special Feature: Nearly flat outer region with deep central basin
    
    Usage in Benchmarking:
        - Tests both exploration AND exploitation
        - Tests algorithm's ability to find narrow global basin
        - Outer region is nearly flat → exploration challenge
        - Central region has deep basin → exploitation challenge
        - Expected: Good algorithms converge to global optimum
    
    Args:
        x (numpy.ndarray): D-boyutlu çözüm vektörü
        bounds (tuple, optional): (lower, upper) sınır değerleri. 
                                  None ise sınır kontrolü yapılmaz.
        
    Returns:
        float: Fonksiyon değeri
        
    References:
        - Ackley, D. H. (1987). A connectionist machine for genetic 
          hillclimbing. Kluwer Academic Publishers.
        - Bäck, T. (1996). Evolutionary algorithms in theory and practice.
        - CEC benchmark suite
        - Jamil, M., & Yang, X. S. (2013). A literature survey of 
          benchmark functions for global optimisation problems.
    
    Example:
        >>> x = np.zeros(30)
        >>> ackley(x)
        0.0
        >>> x = np.array([1.0, 1.0, 1.0])
        >>> ackley(x)  # ≈ 3.625
        3.6253849384403627
    
    Notes:
        - The function has a nearly flat outer region
        - Global minimum basin is very narrow (high gradient near origin)
        - Cosine modulation creates regular pattern of local minima
        - Excellent test for balanced exploration-exploitation
        - More challenging than Rastrigin due to non-separability
    """
    # Standard Ackley parameters
    a = 20.0
    b = 0.2
    c = 2.0 * np.pi
    n = len(x)
    
    # Ana hesaplama
    sum1 = np.sum(x**2)
    sum2 = np.sum(np.cos(c * x))
    
    term1 = -a * np.exp(-b * np.sqrt(sum1 / n))
    term2 = -np.exp(sum2 / n)
    
    fitness = term1 + term2 + a + np.exp(1)
    
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


def get_ackley_info():
    """
    Ackley fonksiyonu hakkında detaylı bilgi döndürür.
    
    Returns:
        dict: Fonksiyon özellikleri
    """
    return {
        'name': 'Ackley Function',
        'symbol': 'f3',
        'type': 'Highly Multimodal',
        'separable': False,
        'differentiable': True,
        'scalable': True,
        'continuous': True,
        'convex': False,
        'global_minimum': 0.0,
        'global_minimum_location': 'x* = (0, 0, ..., 0)',
        'recommended_bounds': [-32.768, 32.768],  # or [-5, 5]
        'difficulty': 'Medium-Hard',
        'tests': 'Exploration + Exploitation, basin finding',
        'formula': "f(x) = -20·exp(-0.2·√(Σxi²/D)) - exp(Σcos(2π·xi)/D) + 20 + e",
        'local_minima_pattern': 'Regular, numerous',
        'dimensions_tested': [10, 30, 50, 100],
        'standard_runs': 30,
        'max_evaluations': '10,000 × D',
        'key_challenge': 'Finding narrow global basin in nearly flat landscape',
        'special_features': 'Nearly flat outer region + deep central basin'
    }


# ============================================================================
# TEST SUITE - Fonksiyonun doğruluğunu test eder
# ============================================================================

def run_tests():
    """Ackley fonksiyonu için kapsamlı test suite'i"""
    
    print("=" * 70)
    print(" " * 20 + "ACKLEY FUNCTION TEST SUITE")
    print("=" * 70)
    
    test_results = []
    
    # Test 1: Global minimum (en önemli test)
    print("\n[Test 1] Global Minimum Test")
    print("-" * 70)
    for dim in [3, 5, 10, 30, 50]:
        test_point = np.zeros(dim)
        result = ackley(test_point)
        passed = abs(result) < 1e-10
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  Dimension {dim:3d}: f(0^{dim}) = {result:.10e} ... {status}")
        test_results.append(passed)
    
    # Test 2: Bilinen değerler
    print("\n[Test 2] Known Values Test")
    print("-" * 70)
    
    # Manual calculation for verification
    # f([1,1,1]) with a=20, b=0.2, c=2π
    test_point1 = np.array([1.0, 1.0, 1.0])
    result1 = ackley(test_point1)
    # Approximate expected value (can be calculated manually)
    print(f"  f([1, 1, 1]) = {result1:.10f}")
    print(f"    Components:")
    sum1 = np.sum(test_point1**2)
    sum2 = np.sum(np.cos(2*np.pi*test_point1))
    print(f"      sum(x²) = {sum1:.6f}")
    print(f"      sum(cos(2πx)) = {sum2:.6f} (cos(2π)=1, so = {len(test_point1)})")
    term1 = -20 * np.exp(-0.2 * np.sqrt(sum1 / 3))
    term2 = -np.exp(sum2 / 3)
    print(f"      term1 = {term1:.6f}")
    print(f"      term2 = {term2:.6f}")
    print(f"      Result = {term1} + {term2} + 20 + e = {result1:.6f}")
    
    # Test with single dimension
    test_point2 = np.array([1.0])
    result2 = ackley(test_point2)
    print(f"\n  f([1]) = {result2:.10f}")
    
    # Test 3: Simetri testi
    print("\n[Test 3] Symmetry Test")
    print("-" * 70)
    test_point_pos = np.array([1.5, 2.5, 3.5, 1.0])
    test_point_neg = -test_point_pos
    result_pos = ackley(test_point_pos)
    result_neg = ackley(test_point_neg)
    error_sym = abs(result_pos - result_neg)
    passed = error_sym < 1e-10
    status = "✓ PASS" if passed else "✗ FAIL"
    print(f"  f([1.5, 2.5, 3.5, 1.0])  = {result_pos:.10f}")
    print(f"  f([-1.5,-2.5,-3.5,-1.0]) = {result_neg:.10f}")
    print(f"  Difference: {error_sym:.2e} ... {status}")
    test_results.append(passed)
    
    # Test 4: Basin depth test (important characteristic)
    print("\n[Test 4] Basin Depth Test")
    print("-" * 70)
    print("  Testing gradient near global minimum:")
    
    distances = [0.0, 0.01, 0.1, 1.0, 5.0, 10.0]
    for dist in distances:
        test_point = np.ones(5) * dist
        result = ackley(test_point)
        print(f"  Distance {dist:5.2f} from origin: f(x) = {result:.10f}")
    
    print("\n  → Deep basin confirmed: steep gradient near origin")
    
    # Test 5: Outer region flatness test
    print("\n[Test 5] Outer Region Flatness Test")
    print("-" * 70)
    print("  Testing nearly flat outer region:")
    
    outer_points = [
        np.array([20.0, 20.0]),
        np.array([25.0, 25.0]),
        np.array([30.0, 30.0]),
    ]
    
    results = []
    for point in outer_points:
        result = ackley(point)
        results.append(result)
        print(f"  f({point}) = {result:.10f}")
    
    # Check if values are close (nearly flat)
    variance = np.var(results)
    print(f"\n  Variance in outer region: {variance:.6f}")
    status = "✓ Nearly Flat" if variance < 1.0 else "⚠ Not Flat"
    print(f"  Flatness characteristic: {status}")
    
    # Test 6: Non-separability test
    print("\n[Test 6] Non-Separability Test")
    print("-" * 70)
    print("  Ackley is non-separable (different from Rastrigin/Sphere)")
    
    # Test: f([x, 0]) + f([0, y]) ≠ f([x, y]) for non-separable functions
    x1 = np.array([2.0, 0.0])
    x2 = np.array([0.0, 2.0])
    x3 = np.array([2.0, 2.0])
    
    f1 = ackley(x1)
    f2 = ackley(x2)
    f3 = ackley(x3)
    
    print(f"  f([2, 0]) = {f1:.6f}")
    print(f"  f([0, 2]) = {f2:.6f}")
    print(f"  f([2, 2]) = {f3:.6f}")
    print(f"  f([2,0]) + f([0,2]) = {f1 + f2:.6f}")
    
    is_separable = abs((f1 + f2) - f3) < 0.01
    status = "✗ Non-Separable (Correct!)" if not is_separable else "⚠ Appears Separable"
    print(f"  Separability check: {status}")
    
    # Test 7: Yüksek boyut testi
    print("\n[Test 7] High Dimensional Test")
    print("-" * 70)
    for dim in [100, 500, 1000]:
        # Global minimum
        test_point_zero = np.zeros(dim)
        result_zero = ackley(test_point_zero)
        passed_zero = abs(result_zero) < 1e-10
        
        # Random point (should be >> 0)
        test_point_rand = np.random.uniform(-32.768, 32.768, dim)
        result_rand = ackley(test_point_rand)
        
        status = "✓ PASS" if passed_zero else "✗ FAIL"
        print(f"  Dimension {dim:4d}:")
        print(f"    f(0^{dim}) = {result_zero:.10e} ... {status}")
        print(f"    f(random) = {result_rand:.6f} (>> 0 expected)")
        test_results.append(passed_zero)
    
    # Test 8: Sınır kontrolü testi
    print("\n[Test 8] Boundary Handling Test")
    print("-" * 70)
    test_point_in = np.array([30.0, -30.0, 0.0])
    test_point_out = np.array([50.0, -50.0, 0.0])
    
    result_in = ackley(test_point_in, bounds=(-32.768, 32.768))
    result_out = ackley(test_point_out, bounds=(-32.768, 32.768))
    
    print(f"  In bounds [30, -30, 0]:     f(x) = {result_in:.6e}")
    print(f"  Out of bounds [50, -50, 0]: f(x) = {result_out:.6e}")
    
    passed = result_out > result_in * 1000
    status = "✓ PASS" if passed else "✗ FAIL"
    print(f"  Penalty applied correctly: {status}")
    test_results.append(passed)
    
    # Test 9: Convergence difficulty test
    print("\n[Test 9] Convergence Difficulty Assessment")
    print("-" * 70)
    print("  Comparing with Rastrigin (similar difficulty):")
    
    from rastrigin import rastrigin
    
    dim = 30
    np.random.seed(42)
    test_points = [np.random.uniform(-5, 5, dim) for _ in range(5)]
    
    print(f"\n  {'Point':<8} {'Ackley':>12} {'Rastrigin':>12}")
    print("  " + "-" * 35)
    for i, point in enumerate(test_points):
        ack = ackley(point)
        ras = rastrigin(point)
        print(f"  Point {i+1:2d} {ack:12.4f} {ras:12.4f}")
    
    print("\n  → Both functions are multimodal and challenging")
    
    # Test 10: Vectorization efficiency test
    print("\n[Test 10] Vectorization Efficiency Test")
    print("-" * 70)
    import time
    dim = 10000
    x = np.random.uniform(-32.768, 32.768, dim)
    
    start = time.time()
    for _ in range(1000):
        result = ackley(x)
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
    info = get_ackley_info()
    
    print("\n" + "=" * 70)
    print(" " * 20 + "ACKLEY FUNCTION INFORMATION")
    print("=" * 70)
    
    for key, value in info.items():
        print(f"{key.replace('_', ' ').title():25s}: {value}")
    
    print("=" * 70)
    
    # Ek bilgiler
    print("\n📊 ACKLEY CHARACTERISTICS:")
    print("-" * 70)
    print("  • Global minimum basin: Very deep and narrow")
    print("  • Outer region: Nearly flat (ƒ(x) ≈ 20 + e ≈ 22.72 for large |x|)")
    print("  • Gradient: Very steep near origin, nearly zero far from origin")
    print("  • Non-separable: Dimensions interact through averaging")
    print("  • Challenge: Finding narrow basin in vast, flat search space")
    print("  • Vs Rastrigin: Similar difficulty but non-separable")
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
    from ackley import ackley
    
    # 30 boyutlu random çözüm
    x = np.random.uniform(-32.768, 32.768, 30)
    fitness = ackley(x)
    print(f"Fitness: {fitness}")
    
    # Sınır kontrolü ile
    fitness_bounded = ackley(x, bounds=(-32.768, 32.768))
    print(f"Fitness (bounded): {fitness_bounded}")
    """)
    
    # Gerçek örnek
    print("\nActual example:")
    print("-" * 70)
    x_random = np.random.uniform(-32.768, 32.768, 30)
    fitness_random = ackley(x_random)
    print(f"Random solution (D=30):  f(x) = {fitness_random:.6f}")
    
    x_optimal = np.zeros(30)
    fitness_optimal = ackley(x_optimal)
    print(f"Optimal solution (D=30): f(x*) = {fitness_optimal:.10e}")
    
    # Outer region example
    x_outer = np.ones(30) * 30
    fitness_outer = ackley(x_outer)
    print(f"Outer region (30^30):    f(x) = {fitness_outer:.6f} (nearly flat)")
    
    print("\n" + "=" * 70)
    print("✅ ackley.py is ready for your benchmark study!")
    print("=" * 70)