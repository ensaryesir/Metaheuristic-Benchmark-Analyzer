import numpy as np

def sphere(x, bounds=None):
    """
    Sphere (f1) - Unimodal Benchmark Function
    
    Mathematical Definition:
        f(x) = Σ(xi²), i=1 to D
    
    Properties:
        - Type: Unimodal (tek optimal nokta)
        - Global Minimum: f(0,...,0) = 0
        - Search Domain: [-100, 100]^D (standart)
        - Difficulty: Kolay - gradyan tabanlı algoritmalar için ideal
        - Separable: Evet (her boyut bağımsız optimize edilebilir)
        - Symmetry: Evet (orijine göre simetrik)
        - Smoothness: Sürekli, türevlenebilir, konveks
    
    Usage in Benchmarking:
        - Tests exploitation capability (sömürü yeteneği)
        - Baseline for algorithm comparison
        - Expected: All decent algorithms should converge to near-zero
    
    Args:
        x (numpy.ndarray): D-boyutlu çözüm vektörü
        bounds (tuple, optional): (lower, upper) sınır değerleri. 
                                  None ise sınır kontrolü yapılmaz.
        
    Returns:
        float: Fonksiyon değeri
        
    References:
        - Yang, X. S. (2010). Test problems in optimization
        - CEC benchmark suite
        - Jamil, M., & Yang, X. S. (2013). A literature survey of 
          benchmark functions for global optimisation problems.
    
    Example:
        >>> x = np.zeros(30)
        >>> sphere(x)
        0.0
        >>> x = np.array([1.0, 2.0, 3.0])
        >>> sphere(x)
        14.0
    """
    # Ana hesaplama
    fitness = np.sum(x**2)
    
    # Sınır kontrolü (opsiyonel)
    if bounds is not None:
        lower, upper = bounds
        # Sınır dışına çıkan değerleri penalize et
        penalty = 0.0
        for xi in x:
            if xi < lower:
                penalty += 1e10 * (lower - xi)**2
            elif xi > upper:
                penalty += 1e10 * (xi - upper)**2
        fitness += penalty
    
    return fitness


def get_sphere_info():
    """
    Sphere fonksiyonu hakkında detaylı bilgi döndürür.
    
    Returns:
        dict: Fonksiyon özellikleri
    """
    return {
        'name': 'Sphere Function',
        'symbol': 'f1',
        'type': 'Unimodal',
        'separable': True,
        'differentiable': True,
        'scalable': True,
        'continuous': True,
        'convex': True,
        'global_minimum': 0.0,
        'global_minimum_location': 'x* = (0, 0, ..., 0)',
        'recommended_bounds': [-100, 100],
        'difficulty': 'Easy',
        'tests': 'Exploitation capability',
        'formula': 'f(x) = Σ(xi²)',
        'dimensions_tested': [10, 30, 50, 100],
        'standard_runs': 30,
        'max_evaluations': '10,000 × D'
    }


# ============================================================================
# TEST SUITE - Fonksiyonun doğruluğunu test eder
# ============================================================================

def run_tests():
    """Sphere fonksiyonu için kapsamlı test suite'i"""
    
    print("=" * 70)
    print(" " * 20 + "SPHERE FUNCTION TEST SUITE")
    print("=" * 70)
    
    test_results = []
    
    # Test 1: Global minimum (en önemli test)
    print("\n[Test 1] Global Minimum Test")
    print("-" * 70)
    for dim in [3, 10, 30, 50]:
        test_point = np.zeros(dim)
        result = sphere(test_point)
        passed = abs(result) < 1e-10
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  Dimension {dim:3d}: f(0^{dim}) = {result:.10e} ... {status}")
        test_results.append(passed)
    
    # Test 2: Bilinen değerler
    print("\n[Test 2] Known Values Test")
    print("-" * 70)
    test_cases = [
        (np.array([1.0, 2.0, 3.0]), 14.0, "[1, 2, 3]"),
        (np.array([2.0, 2.0, 2.0]), 12.0, "[2, 2, 2]"),
        (np.array([1.0, 1.0, 1.0, 1.0, 1.0]), 5.0, "[1]×5"),
        (np.array([3.0, -4.0]), 25.0, "[3, -4]"),
    ]
    
    for test_point, expected, label in test_cases:
        result = sphere(test_point)
        error = abs(result - expected)
        passed = error < 1e-10
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  f({label:12s}) = {result:.6f} (Expected: {expected:.6f}, Error: {error:.2e}) ... {status}")
        test_results.append(passed)
    
    # Test 3: Simetri testi (önemli özellik)
    print("\n[Test 3] Symmetry Test")
    print("-" * 70)
    test_point_pos = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    test_point_neg = -test_point_pos
    result_pos = sphere(test_point_pos)
    result_neg = sphere(test_point_neg)
    passed = abs(result_pos - result_neg) < 1e-10
    status = "✓ PASS" if passed else "✗ FAIL"
    print(f"  f([1,2,3,4,5])  = {result_pos:.6f}")
    print(f"  f([-1,-2,-3,-4,-5]) = {result_neg:.6f}")
    print(f"  Symmetry check: {status}")
    test_results.append(passed)
    
    # Test 4: Yüksek boyut performans testi
    print("\n[Test 4] High Dimensional Test")
    print("-" * 70)
    for dim in [100, 500, 1000]:
        test_point = np.random.uniform(-10, 10, dim)
        expected = np.sum(test_point**2)
        result = sphere(test_point)
        error = abs(result - expected) / expected if expected > 0 else 0
        passed = error < 1e-10
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  Dimension {dim:4d}: f(x) = {result:.6e} (Relative error: {error:.2e}) ... {status}")
        test_results.append(passed)
    
    # Test 5: Sınır kontrolü testi
    print("\n[Test 5] Boundary Handling Test")
    print("-" * 70)
    test_point_in = np.array([50.0, -50.0, 0.0])
    test_point_out = np.array([150.0, -150.0, 0.0])
    
    result_in = sphere(test_point_in, bounds=(-100, 100))
    result_out = sphere(test_point_out, bounds=(-100, 100))
    
    print(f"  In bounds [50, -50, 0]:   f(x) = {result_in:.6e}")
    print(f"  Out of bounds [150, -150, 0]: f(x) = {result_out:.6e}")
    
    passed = result_out > result_in * 1000  # Penaltı çok büyük olmalı
    status = "✓ PASS" if passed else "✗ FAIL"
    print(f"  Penalty applied correctly: {status}")
    test_results.append(passed)
    
    # Test 6: Vectorization efficiency test
    print("\n[Test 6] Vectorization Efficiency Test")
    print("-" * 70)
    import time
    dim = 10000
    x = np.random.randn(dim)
    
    start = time.time()
    for _ in range(1000):
        result = sphere(x)
    elapsed = time.time() - start
    
    print(f"  1000 evaluations at D={dim}: {elapsed:.4f} seconds")
    print(f"  Average time per evaluation: {elapsed/1000*1e6:.2f} μs")
    status = "✓ PASS" if elapsed < 1.0 else "⚠ SLOW"
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
    info = get_sphere_info()
    
    print("\n" + "=" * 70)
    print(" " * 20 + "SPHERE FUNCTION INFORMATION")
    print("=" * 70)
    
    for key, value in info.items():
        print(f"{key.replace('_', ' ').title():25s}: {value}")
    
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
    from sphere import sphere
    
    # 30 boyutlu random çözüm
    x = np.random.uniform(-100, 100, 30)
    fitness = sphere(x)
    print(f"Fitness: {fitness}")
    
    # Sınır kontrolü ile
    fitness_bounded = sphere(x, bounds=(-100, 100))
    print(f"Fitness (bounded): {fitness_bounded}")
    """)
    
    # Gerçek örnek
    print("\nActual example:")
    print("-" * 70)
    x_random = np.random.uniform(-100, 100, 30)
    fitness_random = sphere(x_random)
    print(f"Random solution (D=30): f(x) = {fitness_random:.6f}")
    
    x_optimal = np.zeros(30)
    fitness_optimal = sphere(x_optimal)
    print(f"Optimal solution (D=30): f(x*) = {fitness_optimal:.10e}")
    
    print("\n" + "=" * 70)
    print("✅ sphere.py is ready for your benchmark study!")
    print("=" * 70)
