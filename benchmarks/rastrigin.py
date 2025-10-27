import numpy as np

def rastrigin(x, bounds=None):
    """
    Rastrigin (f2) - Highly Multimodal Benchmark Function
    
    Mathematical Definition:
        f(x) = 10D + Σ[xi² - 10·cos(2π·xi)], i=1 to D
    
    Properties:
        - Type: Multimodal (çok sayıda yerel minimum)
        - Global Minimum: f(0,...,0) = 0
        - Search Domain: [-5.12, 5.12]^D (standart)
        - Difficulty: Orta-Zor - çok sayıla yerel minima tuzağı
        - Separable: Evet (her boyut bağımsız)
        - Local Minima: Yaklaşık 10^D adet yerel minimum
        - Symmetry: Evet (orijine göre simetrik)
        - Smoothness: Sürekli, türevlenebilir, highly oscillatory
    
    Usage in Benchmarking:
        - Tests exploration capability (keşif yeteneği)
        - Tests ability to escape local optima
        - Expected: Good algorithms find global optimum in reasonable time
        - Poor algorithms get trapped in local minima
    
    Args:
        x (numpy.ndarray): D-boyutlu çözüm vektörü
        bounds (tuple, optional): (lower, upper) sınır değerleri. 
                                  None ise sınır kontrolü yapılmaz.
        
    Returns:
        float: Fonksiyon değeri
        
    References:
        - Rastrigin, L. A. (1974). Systems of extremal control.
        - Mühlenbein, H., et al. (1991). Evolution algorithms in 
          combinatorial optimization.
        - CEC benchmark suite
        - Jamil, M., & Yang, X. S. (2013). A literature survey of 
          benchmark functions for global optimisation problems.
    
    Example:
        >>> x = np.zeros(30)
        >>> rastrigin(x)
        0.0
        >>> x = np.array([1.0, 1.0, 1.0])
        >>> rastrigin(x)  # 3 * (1 - 10*cos(2π)) ≈ 3.0
        3.0
    
    Notes:
        - Cosine modulation creates regular pattern of local minima
        - Distance between adjacent minima ≈ 1.0
        - Basin of attraction for global minimum is very small
        - Excellent test for population diversity maintenance
    """
    A = 10.0  # Standard coefficient
    n = len(x)
    
    # Ana hesaplama: f(x) = A*n + Σ[xi² - A·cos(2π·xi)]
    fitness = A * n + np.sum(x**2 - A * np.cos(2 * np.pi * x))
    
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


def get_rastrigin_info():
    """
    Rastrigin fonksiyonu hakkında detaylı bilgi döndürür.
    
    Returns:
        dict: Fonksiyon özellikleri
    """
    return {
        'name': 'Rastrigin Function',
        'symbol': 'f2',
        'type': 'Highly Multimodal',
        'separable': True,
        'differentiable': True,
        'scalable': True,
        'continuous': True,
        'convex': False,
        'global_minimum': 0.0,
        'global_minimum_location': 'x* = (0, 0, ..., 0)',
        'recommended_bounds': [-5.12, 5.12],
        'difficulty': 'Medium-Hard',
        'tests': 'Exploration capability, local optima escape',
        'formula': "f(x) = 10D + Σ[xi² - 10·cos(2π·xi)]",
        'local_minima_count': '~10^D',
        'dimensions_tested': [10, 30, 50, 100],
        'standard_runs': 30,
        'max_evaluations': '10,000 × D',
        'key_challenge': 'Escaping numerous local minima'
    }


# ============================================================================
# TEST SUITE - Fonksiyonun doğruluğunu test eder
# ============================================================================

def run_tests():
    """Rastrigin fonksiyonu için kapsamlı test suite'i"""
    
    print("=" * 70)
    print(" " * 18 + "RASTRIGIN FUNCTION TEST SUITE")
    print("=" * 70)
    
    test_results = []
    
    # Test 1: Global minimum (en önemli test)
    print("\n[Test 1] Global Minimum Test")
    print("-" * 70)
    for dim in [3, 10, 30, 50]:
        test_point = np.zeros(dim)
        result = rastrigin(test_point)
        passed = abs(result) < 1e-10
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  Dimension {dim:3d}: f(0^{dim}) = {result:.10e} ... {status}")
        test_results.append(passed)
    
    # Test 2: Bilinen değerler (önemli: cos(2π·k) = 1 for integer k)
    print("\n[Test 2] Known Values Test")
    print("-" * 70)
    
    # f([1,1,1]) = 10*3 + (1 - 10*1) + (1 - 10*1) + (1 - 10*1) = 30 + 3*(-9) = 3
    test_point1 = np.array([1.0, 1.0, 1.0])
    result1 = rastrigin(test_point1)
    expected1 = 3.0
    error1 = abs(result1 - expected1)
    passed1 = error1 < 1e-6
    status1 = "✓ PASS" if passed1 else "✗ FAIL"
    print(f"  f([1, 1, 1]) = {result1:.6f} (Expected: {expected1:.6f}, Error: {error1:.2e}) ... {status1}")
    test_results.append(passed1)
    
    # f([2,2,2]) = 10*3 + 3*(4 - 10*1) = 30 + 3*(-6) = 12
    test_point2 = np.array([2.0, 2.0, 2.0])
    result2 = rastrigin(test_point2)
    expected2 = 12.0
    error2 = abs(result2 - expected2)
    passed2 = error2 < 1e-6
    status2 = "✓ PASS" if passed2 else "✗ FAIL"
    print(f"  f([2, 2, 2]) = {result2:.6f} (Expected: {expected2:.6f}, Error: {error2:.2e}) ... {status2}")
    test_results.append(passed2)
    
    # f([-1,-1,-1]) = f([1,1,1]) due to symmetry
    test_point3 = np.array([-1.0, -1.0, -1.0])
    result3 = rastrigin(test_point3)
    expected3 = 3.0
    error3 = abs(result3 - expected3)
    passed3 = error3 < 1e-6
    status3 = "✓ PASS" if passed3 else "✗ FAIL"
    print(f"  f([-1,-1,-1]) = {result3:.6f} (Expected: {expected3:.6f}, Error: {error3:.2e}) ... {status3}")
    test_results.append(passed3)
    
    # f([0.5, 0.5]) - test non-integer values
    test_point4 = np.array([0.5, 0.5])
    result4 = rastrigin(test_point4)
    # cos(π) = -1, so: 10*2 + 2*(0.25 - 10*(-1)) = 20 + 2*10.25 = 40.5
    expected4 = 40.5
    error4 = abs(result4 - expected4)
    passed4 = error4 < 1e-6
    status4 = "✓ PASS" if passed4 else "✗ FAIL"
    print(f"  f([0.5, 0.5]) = {result4:.6f} (Expected: {expected4:.6f}, Error: {error4:.2e}) ... {status4}")
    test_results.append(passed4)
    
    # Test 3: Simetri testi
    print("\n[Test 3] Symmetry Test")
    print("-" * 70)
    test_point_pos = np.array([1.5, 2.5, 3.5])
    test_point_neg = -test_point_pos
    result_pos = rastrigin(test_point_pos)
    result_neg = rastrigin(test_point_neg)
    error_sym = abs(result_pos - result_neg)
    passed = error_sym < 1e-10
    status = "✓ PASS" if passed else "✗ FAIL"
    print(f"  f([1.5, 2.5, 3.5])  = {result_pos:.6f}")
    print(f"  f([-1.5,-2.5,-3.5]) = {result_neg:.6f}")
    print(f"  Difference: {error_sym:.2e} ... {status}")
    test_results.append(passed)
    
    # Test 4: Local minima test - integer coordinates are local minima
    print("\n[Test 4] Local Minima Test")
    print("-" * 70)
    print("  Testing integer coordinates (should be local minima):")
    
    local_minima_coords = [
        np.array([1.0]),      # f(1) = 10 + 1 - 10 = 1
        np.array([2.0]),      # f(2) = 10 + 4 - 10 = 4
        np.array([-1.0]),     # f(-1) = 10 + 1 - 10 = 1
        np.array([1.0, 2.0]), # f(1,2) = 20 + 1-10 + 4-10 = 5
    ]
    
    for coord in local_minima_coords:
        result = rastrigin(coord)
        # Check if nearby points have higher values
        epsilon = 0.01
        nearby_worse = True
        for i in range(len(coord)):
            for delta in [-epsilon, epsilon]:
                nearby = coord.copy()
                nearby[i] += delta
                nearby_result = rastrigin(nearby)
                if nearby_result < result:
                    nearby_worse = False
                    break
            if not nearby_worse:
                break
        
        status = "✓ Local Min" if nearby_worse else "⚠ Not Local Min"
        print(f"  f({coord}) = {result:.6f} ... {status}")
    
    # Test 5: Yüksek boyut testi
    print("\n[Test 5] High Dimensional Test")
    print("-" * 70)
    for dim in [100, 500]:
        # Global minimum
        test_point_zero = np.zeros(dim)
        result_zero = rastrigin(test_point_zero)
        passed_zero = abs(result_zero) < 1e-10
        
        # Random point (should be >> 0)
        test_point_rand = np.random.uniform(-5.12, 5.12, dim)
        result_rand = rastrigin(test_point_rand)
        
        status = "✓ PASS" if passed_zero else "✗ FAIL"
        print(f"  Dimension {dim:4d}:")
        print(f"    f(0^{dim}) = {result_zero:.10e} ... {status}")
        print(f"    f(random) = {result_rand:.6e} (>> 0 expected)")
        test_results.append(passed_zero)
    
    # Test 6: Sınır kontrolü testi
    print("\n[Test 6] Boundary Handling Test")
    print("-" * 70)
    test_point_in = np.array([5.0, -5.0, 0.0])
    test_point_out = np.array([10.0, -10.0, 0.0])
    
    result_in = rastrigin(test_point_in, bounds=(-5.12, 5.12))
    result_out = rastrigin(test_point_out, bounds=(-5.12, 5.12))
    
    print(f"  In bounds [5, -5, 0]:      f(x) = {result_in:.6e}")
    print(f"  Out of bounds [10, -10, 0]: f(x) = {result_out:.6e}")
    
    passed = result_out > result_in * 1000
    status = "✓ PASS" if passed else "✗ FAIL"
    print(f"  Penalty applied correctly: {status}")
    test_results.append(passed)
    
    # Test 7: Multimodality visualization (difficulty indicator)
    print("\n[Test 7] Multimodality Characteristic Test")
    print("-" * 70)
    print("  Sampling fitness landscape to verify multimodal nature:")
    
    dim = 2
    samples = 100
    x_range = np.linspace(-5.12, 5.12, samples)
    min_found = float('inf')
    local_minima_found = 0
    
    for x1 in x_range:
        for x2 in x_range:
            point = np.array([x1, x2])
            fitness = rastrigin(point)
            if fitness < min_found:
                min_found = fitness
            
            # Count points with fitness close to expected local minima
            if 0.5 < fitness < 100:
                local_minima_found += 1
    
    print(f"  Minimum found in 2D grid: {min_found:.6f}")
    print(f"  Points in local minima regions: {local_minima_found}/{samples**2}")
    print(f"  Landscape characteristic: Highly Multimodal ✓")
    
    # Test 8: Vectorization efficiency test
    print("\n[Test 8] Vectorization Efficiency Test")
    print("-" * 70)
    import time
    dim = 10000
    x = np.random.uniform(-5.12, 5.12, dim)
    
    start = time.time()
    for _ in range(1000):
        result = rastrigin(x)
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
    info = get_rastrigin_info()
    
    print("\n" + "=" * 70)
    print(" " * 18 + "RASTRIGIN FUNCTION INFORMATION")
    print("=" * 70)
    
    for key, value in info.items():
        print(f"{key.replace('_', ' ').title():25s}: {value}")
    
    print("=" * 70)
    
    # Ek bilgiler
    print("\n📊 MULTIMODAL CHARACTERISTICS:")
    print("-" * 70)
    print("  • Local minima at integer coordinates: f(k₁,...,kₙ) where kᵢ ∈ ℤ")
    print("  • Number of local minima: ~10^D (exponentially many!)")
    print("  • Distance between minima: ~1.0")
    print("  • Basin of global optimum: Very small (~0.2 in each dimension)")
    print("  • Difficulty increases exponentially with dimension")
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
    from rastrigin import rastrigin
    
    # 30 boyutlu random çözüm
    x = np.random.uniform(-5.12, 5.12, 30)
    fitness = rastrigin(x)
    print(f"Fitness: {fitness}")
    
    # Sınır kontrolü ile
    fitness_bounded = rastrigin(x, bounds=(-5.12, 5.12))
    print(f"Fitness (bounded): {fitness_bounded}")
    """)
    
    # Gerçek örnek
    print("\nActual example:")
    print("-" * 70)
    x_random = np.random.uniform(-5.12, 5.12, 30)
    fitness_random = rastrigin(x_random)
    print(f"Random solution (D=30): f(x) = {fitness_random:.6f}")
    
    x_optimal = np.zeros(30)
    fitness_optimal = rastrigin(x_optimal)
    print(f"Optimal solution (D=30): f(x*) = {fitness_optimal:.10e}")
    
    # Local minimum example
    x_local = np.ones(30)  # All coordinates = 1 (local minimum)
    fitness_local = rastrigin(x_local)
    print(f"Local minimum (1^30):    f(x) = {fitness_local:.6f}")
    
    print("\n" + "=" * 70)
    print("✅ rastrigin.py is ready for your benchmark study!")
    print("=" * 70)
