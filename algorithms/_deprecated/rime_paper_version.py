import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import math

"""
    Neden Başarısız Oluyor?
        RIME algoritmasının her iki hareket stratejisi de (soft-rime ve hard-rime) şu temel formüle dayanır:

        new_pos = rime_ice + [faktör] * (rime_ice - population[i])

        Bu denklemin anlamı şudur: Her bir parçacık (population[i]), her zaman mevcut en iyi çözüm (rime_ice) ile kendi pozisyonu arasında uzanan doğru bir çizgi üzerinde hareket eder. 
        Algoritmanın, bu çizginin dışındaki bölgeleri keşfetmek için hiçbir mekanizması yoktur.

        Tek Çekim Noktası Sorunu: rime_ice bir kez bir yerel minimuma takıldığında, tüm popülasyon adeta bir kara delik gibi o noktaya doğru çekilir. 
        Popülasyon çeşitliliği hızla yok olur ve algoritma o noktadan kaçamaz.
        
        "Pozitif Açgözlü Seçim" Neden İşe Yaramıyor? 
        Bu mekanizma, kağıt üzerinde çeşitliliği artırmak için tasarlanmış olsa da, 
        eğer tüm yeni çözümler yine aynı "çekim merkezi" (rime_ice) etrafında üretiliyorsa, popülasyona anlamlı bir yenilik veya farklı bir arama yönü katamaz.
"""
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
    # Test edilecek benchmark fonksiyonlarını ve sınırlarını içe aktar
    from benchmarks.sphere import sphere
    from benchmarks.rastrigin import rastrigin
    from benchmarks.ackley import ackley

    # Genel algoritma parametreleri
    dim = 10
    pop_size = 30
    max_iter = 1000

    # Test edilecek fonksiyonları, isimlerini ve standart arama sınırlarını bir listede topla
    benchmarks_to_test = [
        {
            "name": "Sphere",
            "func": sphere,
            "bounds": (-5.12, 5.12)
        },
        {
            "name": "Rastrigin", 
            "func": rastrigin,
            "bounds": (-5.12, 5.12)
        },
        {
            "name": "Ackley",
            "func": ackley,
            "bounds": (-32.768, 32.768)
        }
    ]

    # Her bir benchmark fonksiyonu için RIME algoritmasını çalıştır
    for benchmark in benchmarks_to_test:
        print(f"\nRIME Algorithm Test - {benchmark['name']} Function")
        print("=" * 50)
        
        best_solution, best_fitness, convergence = rime(
            objective_func=benchmark['func'],
            dim=dim,
            bounds=benchmark['bounds'],
            pop_size=pop_size,
            max_iter=max_iter
        )
        
        print(f"\nResults for {benchmark['name']}:")
        print(f"Best Solution (first 5 dims): {best_solution[:5]}")
        print(f"Best Fitness: {best_fitness:.10f}")
        print(f"Improvement: {convergence[0]:.4f} → {convergence[-1]:.10f}")
        
        # Yakınsama analizi
        if convergence[-1] != 0:
            improvement_ratio = convergence[0] / convergence[-1]
            print(f"Improvement Ratio: {improvement_ratio:.2e}x")
        else:
            print(f"Improvement Ratio: Optimal solution found!")
        
        print("-" * 50)
