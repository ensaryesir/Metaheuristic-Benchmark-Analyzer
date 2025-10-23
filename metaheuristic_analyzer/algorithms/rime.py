import numpy as np
import math

def rime(objective_func, dim, bounds, pop_size, max_iter):
    """
    Rime Optimization Algorithm (RIME) - Orijinal Makaleye Birebir Uygun
    
    Orijinal Makale:
    Su, H., Zhao, D., Heidari, A. A., Liu, L., Zhang, X., Mafarja, M., & Chen, H. (2023). 
    "RIME: A physics-based optimization"
    Neurocomputing, 532, 183-214.
    
    DOI: https://doi.org/10.1016/j.neucom.2023.02.010
    
    Args:
        objective_func: Optimize edilecek fonksiyon
        dim: Problem boyutu
        bounds: Arama uzayı sınırları [(min, max)] veya (min, max)
        pop_size: Popülasyon boyutu
        max_iter: Maksimum iterasyon sayısı
        
    Returns:
        tuple: (best_solution, best_fitness, convergence_curve)
    """
    
    # Sınırları standartlaştır
    if isinstance(bounds, tuple):
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
    
    # Popülasyonu başlat
    population = np.random.uniform(lb, ub, (pop_size, dim))
    
    # Fitness değerlerini hesapla
    fitness = np.array([objective_func(ind) for ind in population])
    
    # En iyi çözüm (Rime-ice)
    best_idx = np.argmin(fitness)
    rime_ice = population[best_idx].copy()
    best_fitness = fitness[best_idx]
    
    convergence_curve = np.zeros(max_iter)
    
    for t in range(max_iter):
        # 1. CRITICAL: Rime Environment Factor (E) - Orijinal denklem
        # E = [tanh(4t/T - 2) + 1] × cos(t/T × π/2)
        E = (np.tanh(4 * t / max_iter - 2) + 1) * np.cos(t / max_iter * np.pi / 2)
        
        # 2. CRITICAL: Popülasyon ortalaması (positive greedy selection için)
        mean_fitness = np.mean(fitness)
        
        for i in range(pop_size):
            # Soft-rime search mechanism - Orijinal denklem
            if np.random.rand() < E:
                # SOFT-RIME: Piercing behavior around rime-ice
                # Rastgele açı (θ) - boyut sayısı kadar açı
                theta = np.random.uniform(0, 2 * np.pi, dim)
                r = np.random.rand()
                
                # Orijinal formül: X_new = X_best + r × cos(θ) × (X_best - X_i)
                new_pos = rime_ice + r * np.cos(theta) * (rime_ice - population[i])
            else:
                # HARD-RIME: Sticking behavior toward rime-ice
                # r_norm bir skaler olmalı (boyut bağımsız)
                r_norm = np.random.randn()
                
                # η hesaplaması - dikkat: r_norm skaler, bu yüzden norm hesaplanmaz
                # Orijinal makalede: η = E × |r_norm|
                eta = E * np.abs(r_norm)
                
                # Orijinal formül: X_new = X_best + η × (X_best - X_i)
                new_pos = rime_ice + eta * (rime_ice - population[i])
            
            # Sınırları kontrol et
            new_pos = np.clip(new_pos, lb, ub)
            
            # Fitness değerlendirme
            new_fitness = objective_func(new_pos)
            
            # 3. CRITICAL: POSITIVE GREEDY SELECTION - Orijinal mekanizma
            # Yeni çözüm mevcut çözümden daha iyi VEYA popülasyon ortalamasından daha iyiyse kabul et
            if new_fitness < fitness[i] or new_fitness < mean_fitness:
                population[i] = new_pos
                fitness[i] = new_fitness
                
                # Rime-ice'ı güncelle (standart greedy selection)
                if new_fitness < best_fitness:
                    best_fitness = new_fitness
                    rime_ice = new_pos.copy()
        
        convergence_curve[t] = best_fitness
        
        # İlerleme göstergesi
        if (t + 1) % 100 == 0:
            print(f"Iteration {t + 1}/{max_iter}, Best Fitness: {best_fitness:.8f}, E: {E:.3f}")
    
    return rime_ice, best_fitness, convergence_curve

# Test kodu
if __name__ == "__main__":
    # Test fonksiyonu olarak sphere kullan
    try:
        from benchmarks.sphere import sphere
    except ImportError:
        def sphere(x):
            return np.sum(x**2)
    
    # Algoritma parametreleri
    dim = 10
    bounds = (-5.12, 5.12)
    pop_size = 30
    max_iter = 500
    
    print("RIME Algorithm (ORIGINAL PAPER) Test - Sphere Function")
    print("=" * 60)
    
    best_solution, best_fitness, convergence = rime(
        objective_func=sphere,
        dim=dim,
        bounds=bounds,
        pop_size=pop_size,
        max_iter=max_iter
    )
    
    print(f"\nResults:")
    print(f"Best Solution: {best_solution[:5]}...")
    print(f"Best Fitness: {best_fitness:.10f}")
    print(f"Final Convergence: {convergence[-1]:.10f}")
    
    # Yakınsama analizi
    print(f"Initial Best: {convergence[0]:.4f}")
    print(f"Final Best: {convergence[-1]:.10f}")