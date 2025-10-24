import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import math

"""
    Improved RIME Optimization Algorithm - Geliştirilmiş Versiyon (Parametreli)
    
    ⚡ Algoritma Özellikleri:
    - Temel Fikir: Fiziksel kırağı oluşumu ve davranışı
    - Güçlü Yönler: Sosyal bileşen, gürültü ekleme, periyodik sıfırlama
    - Zayıf Yönler: Orijinal RIME'a göre daha fazla parametre
    
    🔧 Parametre Tavsiyeleri:
    - social_rate: 0.3 (Sosyal etkileşim olasılığı)
    - noise_factor: 0.05 (Gürültü şiddeti) 
    - reset_interval: 200 (Sıfırlama aralığı)
    
    Orijinal Makale:
    Su, H., Zhao, D., Heidari, A. A., Liu, L., Zhang, X., Mafarja, M., & Chen, H. (2023). 
    "RIME: A physics-based optimization"
    Neurocomputing, 532, 183-214.
    
    Args:
        objective_func: Optimize edilecek fonksiyon
        dim: Problem boyutu
        bounds: Arama uzayı sınırları [(min, max)] veya (min, max)
        pop_size: Popülasyon boyutu
        max_iter: Maksimum iterasyon sayısı
        social_rate: Sosyal bileşen kullanma olasılığı (0-1)
        noise_factor: Gürültü ekleme faktörü
        reset_interval: Periyodik sıfırlama aralığı
        
    Returns:
        tuple: (best_solution, best_fitness, convergence_curve)
"""

def rime_improved(objective_func, dim, bounds, pop_size, max_iter, 
                  social_rate=0.3, noise_factor=0.05, reset_interval=200):

    
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
        # Geliştirilmiş E faktörü - daha dengeli keşif/sömürü
        E = (np.tanh(4 * t / max_iter - 2) + 1) * np.cos(t / max_iter * np.pi / 2)
        
        for i in range(pop_size):
            # 1. GELİŞTİRME: Rastgele birey etkileşimi (PARAMETRELİ)
            if np.random.rand() < social_rate:  # Parametreden al
                j = np.random.randint(pop_size)
                while j == i:
                    j = np.random.randint(pop_size)
                social_component = population[j] - population[i]
            else:
                social_component = np.zeros(dim)
            
            # 2. GELİŞTİRME: Soft-rime search - genişletilmiş keşif
            if np.random.rand() < E:
                theta = np.random.uniform(0, 2 * np.pi, dim)
                r = np.random.rand() * 2  # Daha büyük adımlar
                
                # Orijinal + sosyal bileşen + rastgele keşif
                new_pos = (rime_ice + r * np.cos(theta) * (ub - lb) * 0.1 + 
                          social_component * 0.5)
            else:
                # 3. GELİŞTİRME: Hard-rime search - geliştirilmiş sömürü
                r_norm = np.random.randn()
                eta = E * np.abs(r_norm)
                
                # Rastgele gürültü ekle (lokal minimumdan kaçış) - PARAMETRELİ
                noise = np.random.randn(dim) * noise_factor * (ub - lb) * (1 - t/max_iter)
                
                new_pos = (rime_ice + eta * (rime_ice - population[i]) + 
                          social_component * 0.3 + noise)
            
            # Sınırları kontrol et
            new_pos = np.clip(new_pos, lb, ub)
            
            # Fitness değerlendirme
            new_fitness = objective_func(new_pos)
            
            # 4. GELİŞTİRME: Standart greedy selection (positive greedy kaldırıldı)
            if new_fitness < fitness[i]:
                population[i] = new_pos
                fitness[i] = new_fitness
                
                # Rime-ice'ı güncelle
                if new_fitness < best_fitness:
                    best_fitness = new_fitness
                    rime_ice = new_pos.copy()
        
        # 5. GELİŞTİRME: Periyodik rastgele sıfırlama (PARAMETRELİ)
        if t % reset_interval == 0 and t > 0:
            # En kötü %10'u rastgele yeniden başlat
            worst_indices = np.argsort(fitness)[-pop_size//10:]
            for idx in worst_indices:
                population[idx] = np.random.uniform(lb, ub, dim)
                fitness[idx] = objective_func(population[idx])
        
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

    # RIME parametreleri
    rime_params = {
        'social_rate': 0.3,
        'noise_factor': 0.05,
        'reset_interval': 200
    }

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

    # Her bir benchmark fonksiyonu için GELİŞTİRİLMİŞ RIME algoritmasını çalıştır
    for benchmark in benchmarks_to_test:
        print(f"\nIMPROVED RIME Algorithm Test - {benchmark['name']} Function")
        print("=" * 60)
        
        best_solution, best_fitness, convergence = rime_improved(
            objective_func=benchmark['func'],
            dim=dim,
            bounds=benchmark['bounds'],
            pop_size=pop_size,
            max_iter=max_iter,
            **rime_params
        )
        
        print(f"\nResults for {benchmark['name']}:")
        print(f"Best Solution (first 5 dims): {best_solution[:5]}")
        print(f"Best Fitness: {best_fitness:.10f}")
        print(f"Improvement: {convergence[0]:.4f} → {convergence[-1]:.10f}")
        
        if convergence[-1] != 0:
            improvement_ratio = convergence[0] / convergence[-1]
            print(f"Improvement Ratio: {improvement_ratio:.2e}x")
        else:
            print(f"Improvement Ratio: Optimal solution found!")
        
        print("-" * 60)
