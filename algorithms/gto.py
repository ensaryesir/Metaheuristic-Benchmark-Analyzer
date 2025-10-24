import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np

"""
    Artificial Gorilla Troops Optimizer (GTO) algoritması
    
    ⚡ Algoritma Özellikleri:
    - Temel Fikir: Goril sürülerinin sosyal davranışları
    - Güçlü Yönler: Dengeli keşif-sömürü, multimodal performans
    - Zayıf Yönler: Nispeten yeni, daha az test edilmiş
    
    🔧 Parametre Tavsiyeleri:
    - beta = 3.0 (Orijinal makale)
    - p = 0.03 (Orijinal makale)
    - w = 0.8 (Sabit parametre)
    
    Orijinal Makale:
    Abdollahzadeh, B., Soleimanian Gharehchopogh, F., & Mirjalili, S. (2021).
    "Artificial Gorilla Troops Optimizer: A New Nature‐Inspired Metaheuristic Algorithm for Global Optimization Problems"
    International Journal of Intelligent Systems, 36(10), 5887-5958.
    
    Args:
        objective_func: Optimize edilecek fonksiyon
        dim: Problem boyutu
        bounds: Arama uzayı sınırları [(min, max)] veya (min, max)
        pop_size: Popülasyon boyutu
        max_iter: Maksimum iterasyon sayısı
        beta: Sürü liderliği parametresi (varsayılan: 3.0)
        p: Rastgele hareket olasılığı (varsayılan: 0.03)
        
    Returns:
        tuple: (best_solution, best_fitness, convergence_curve)
"""

def gto(objective_func, dim, bounds, pop_size, max_iter, beta=3.0, p=0.03):
    
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
    
    # Sabit parametreler (orijinal makaleye göre)
    w = 0.8  # Sabit parametre
    
    # Popülasyonu başlat
    gorillas = np.random.uniform(lb, ub, (pop_size, dim))
    
    # Fitness değerlerini hesapla
    fitness = np.array([objective_func(ind) for ind in gorillas])
    
    # Silverback'i bul
    best_idx = np.argmin(fitness)
    silverback = gorillas[best_idx].copy()
    best_fitness = fitness[best_idx]
    
    convergence_curve = np.zeros(max_iter)
    
    for t in range(max_iter):
        # Parametreyi güncelle (C - exploration control parameter)
        C = w * (1 - t / max_iter)
        
        for i in range(pop_size):
            # Exploration Phase (Keşif Fazı)
            if np.random.rand() < C:
                if np.random.rand() < p:
                    # Migration to unknown places
                    X_new = lb + np.random.rand(dim) * (ub - lb)
                else:
                    # Migration to other gorillas
                    if np.random.rand() >= 0.5:
                        # Follow other gorillas
                        random_gorilla = gorillas[np.random.randint(pop_size)]
                        X_new = (np.random.rand() - 0.5) * 2 * random_gorilla + \
                               C * gorillas[i]
                    else:
                        # Move to known place
                        L = 2 * np.random.rand() - 1
                        X_new = gorillas[i] - L * (L * (ub - lb) + lb)
            else:
                # Exploitation Phase (Sömürü Fazı)
                if np.random.rand() >= 0.5:
                    # Follow the silverback
                    g = 2 ** np.random.rand()
                    L = 2 * np.random.rand() - 1
                    X_new = C * (silverback - g * gorillas[i]) + silverback
                else:
                    # Competition for adult females
                    Q = 2 * np.random.rand() - 1
                    A = beta * Q
                    H = np.random.rand(dim) < 0.5
                    X_new = silverback - (A * (2 * np.random.rand() - 1) * 
                            (silverback - gorillas[i])) * H
            
            # Sınırları kontrol et
            X_new = np.clip(X_new, lb, ub)
            
            # Fitness değerlendirme
            new_fitness = objective_func(X_new)
            
            # Greedy selection
            if new_fitness < fitness[i]:
                gorillas[i] = X_new
                fitness[i] = new_fitness
                
                # Silverback'i güncelle
                if new_fitness < best_fitness:
                    best_fitness = new_fitness
                    silverback = X_new.copy()
                    best_idx = i
        
        convergence_curve[t] = best_fitness
        
        # İlerleme göstergesi
        if (t + 1) % 100 == 0:
            print(f"Iteration {t + 1}/{max_iter}, Best Fitness: {best_fitness:.8f}")
    
    return silverback, best_fitness, convergence_curve

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
    
    # GTO parametreleri (orijinal makaleye göre)
    beta = 3.0    # Sürü liderliği parametresi
    p = 0.03      # Rastgele hareket olasılığı

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

    # Her bir benchmark fonksiyonu için GTO algoritmasını çalıştır
    for benchmark in benchmarks_to_test:
        print(f"\nGTO Algorithm Test - {benchmark['name']} Function")
        print("=" * 50)
        
        best_solution, best_fitness, convergence = gto(
            objective_func=benchmark['func'],
            dim=dim,
            bounds=benchmark['bounds'],
            pop_size=pop_size,
            max_iter=max_iter,
            beta=beta,
            p=p
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
