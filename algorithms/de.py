import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np

"""
    Differential Evolution (DE/rand/1/bin) algoritması
    
    ⚡ Algoritma Özellikleri:
    - Strateji: DE/rand/1/bin (en yaygın varyant)
    - Güçlü Yönler: Basit, etkili, az parametre, multimodal performans
    - Zayıf Yönler: Parametre hassasiyeti
    
    🔧 Parametre Tavsiyeleri:
    - F = 0.5 (Standart), Zor problemler için: 0.8
    - CR = 0.9 (Standart), Keşif için: 0.5-0.7
    
    Orijinal Makale:
    Storn, R., & Price, K. (1997). 
    "Differential Evolution – A Simple and Efficient Heuristic for global Optimization over Continuous Spaces"
    Journal of Global Optimization, 11(4), 341–359.
    
    Args:
        objective_func: Optimize edilecek fonksiyon
        dim: Problem boyutu
        bounds: Arama uzayı sınırları [(min, max)] veya (min, max)
        pop_size: Popülasyon boyutu
        max_iter: Maksimum iterasyon sayısı
        F: Mutasyon faktörü (varsayılan: 0.5)
        CR: Çaprazlama oranı (varsayılan: 0.9)
        
    Returns:
        tuple: (best_solution, best_fitness, convergence_curve)
"""

def de(objective_func, dim, bounds, pop_size, max_iter, F=0.5, CR=0.9):
    
    # Sınırları standartlaştır
    if isinstance(bounds, tuple):
        # Tek tuple: (-10, 10) → tüm boyutlar için aynı sınır
        lb = np.full(dim, bounds[0])
        ub = np.full(dim, bounds[1])
    else:
        # Liste formatı: [(-10, 10), (-5, 5), ...]
        bounds = np.array(bounds)
        if bounds.shape == (2,):
            # [min, max] formatı
            lb = np.full(dim, bounds[0])
            ub = np.full(dim, bounds[1])
        else:
            # [[min1, max1], [min2, max2], ...] formatı
            lb = bounds[:, 0]
            ub = bounds[:, 1]
    
    # Popülasyonu başlat
    population = np.random.uniform(lb, ub, (pop_size, dim))
    
    # Fitness değerlerini hesapla
    fitness = np.array([objective_func(ind) for ind in population])
    
    # En iyi bireyi bul
    best_idx = np.argmin(fitness)
    best_solution = population[best_idx].copy()
    best_fitness = fitness[best_idx]
    
    convergence_curve = np.zeros(max_iter)
    
    for iteration in range(max_iter):
        for i in range(pop_size):
            # Üç farklı birey seç (i'den farklı)
            candidates = [idx for idx in range(pop_size) if idx != i]
            a, b, c = np.random.choice(candidates, 3, replace=False)
            
            # Mutant vektör oluştur
            mutant = population[a] + F * (population[b] - population[c])
            
            # Sınırları kontrol et ve düzelt
            mutant = np.clip(mutant, lb, ub)
            
            # Çaprazlama (binomial crossover)
            trial = population[i].copy()
            cross_points = np.random.rand(dim) < CR
            j_rand = np.random.randint(dim)  # En az bir genin mutlaka değişmesi için
            cross_points[j_rand] = True
            trial[cross_points] = mutant[cross_points]
            
            # Seçim
            trial_fitness = objective_func(trial)
            if trial_fitness < fitness[i]:
                population[i] = trial
                fitness[i] = trial_fitness
                
                # En iyiyi güncelle
                if trial_fitness < best_fitness:
                    best_fitness = trial_fitness
                    best_solution = trial.copy()
        
        convergence_curve[iteration] = best_fitness
        
        # İlerleme göstergesi (opsiyonel)
        if (iteration + 1) % 100 == 0:
            print(f"Iteration {iteration + 1}/{max_iter}, Best Fitness: {best_fitness:.6f}")
    
    return best_solution, best_fitness, convergence_curve

# Test kodu
if __name__ == "__main__":
    # Test edilecek benchmark fonksiyonlarını ve sınırlarını içe aktar
    from benchmarks.sphere import sphere
    from benchmarks.rastrigin import rastrigin
    from benchmarks.ackley import ackley

    # Genel algoritma parametreleri
    dim = 10
    pop_size = 50
    # Zorlu fonksiyonlar için iterasyon sayısını artırmak iyi bir pratiktir
    max_iter = 1000
    F = 0.5
    CR = 0.9

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

    # Her bir benchmark fonksiyonu için DE algoritmasını çalıştır
    for benchmark in benchmarks_to_test:
        print(f"\nDE Algorithm Test - {benchmark['name']} Function")
        print("=" * 40)
        
        best_solution, best_fitness, convergence = de(
            objective_func=benchmark['func'],
            dim=dim,
            bounds=benchmark['bounds'],
            pop_size=pop_size,
            max_iter=max_iter,
            F=F,
            CR=CR
        )
        
        print(f"\nResults for {benchmark['name']}:")
        print(f"Best Solution (first 4 dims): {best_solution[:4]}")
        print(f"Best Fitness: {best_fitness:.8f}")
        print(f"Improvement: {convergence[0]:.4f} → {convergence[-1]:.8f}")
        print("-" * 40)
