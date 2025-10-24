import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import math

"""
    Marine Predators Algorithm (MPA) algoritması
    
    ⚡ Algoritma Özellikleri:
    - Temel Fikir: Deniz predatörlerinin avlanma stratejileri
    - Güçlü Yönler: 3 fazlı yapı, Levy flight, FADs etkisi
    - Zayıf Yönler: Karmaşık implementasyon
    
    🔧 Parametre Tavsiyeleri:
    - P = 0.5 (Orijinal makale)
    - FADs = 0.2 (Orijinal makale)
    
    Orijinal Makale:
    Faramarzi, A., Heidarinejad, M., Mirjalili, S., & Gandomi, A. H. (2020).
    "Marine Predators Algorithm: A nature-inspired metaheuristic"
    Expert Systems with Applications, 152, 113377.
    
    Args:
        objective_func: Optimize edilecek fonksiyon
        dim: Problem boyutu
        bounds: Arama uzayı sınırları [(min, max)] veya (min, max)
        pop_size: Popülasyon boyutu
        max_iter: Maksimum iterasyon sayısı
        P: Sabit parametre (varsayılan: 0.5)
        FADs: Fish Aggregating Devices etkisi (varsayılan: 0.2)
        
    Returns:
        tuple: (best_solution, best_fitness, convergence_curve)
"""

def mpa(objective_func, dim, bounds, pop_size, max_iter, P=0.5, FADs=0.2):
    
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
    
    # Step size için vektör
    stepsize = np.zeros((pop_size, dim))
    
    # Fitness tarihçesi
    fitness_history = np.zeros((pop_size, 1))
    
    # Popülasyonu başlat
    prey = np.random.uniform(lb, ub, (pop_size, dim))
    
    # Fitness değerlerini hesapla
    fitness = np.array([objective_func(ind) for ind in prey])
    fitness_history[:, 0] = fitness.copy()
    
    # Elite matrix (En iyi avcılar)
    elite = prey.copy()
    elite_fitness = fitness.copy()
    
    # En iyi çözüm
    best_idx = np.argmin(fitness)
    best_solution = prey[best_idx].copy()
    best_fitness = fitness[best_idx]
    
    convergence_curve = np.zeros(max_iter)
    
    for iteration in range(max_iter):
        # CF - Adaptive parameter
        CF = (1 - iteration / max_iter) ** (2 * iteration / max_iter)
        
        # RL - Levy flight vektörü
        RL = 0.05 * levy_flight(pop_size, dim)
        
        # RB - Brownian motion vektörü
        RB = np.random.randn(pop_size, dim)
        
        # FAZ 1: İlk 1/3 iterasyon (High velocity ratio)
        if iteration < max_iter / 3:
            for i in range(pop_size):
                # Step size hesapla
                stepsize[i] = 0.5 * RB[i] * (elite[i] - RB[i] * prey[i])
                # Prey güncelle
                prey[i] = prey[i] + P * np.random.rand() * stepsize[i]
        
        # FAZ 2: Orta 1/3 iterasyon (Unit velocity ratio)
        elif iteration < 2 * max_iter / 3:
            for i in range(pop_size):
                if i < pop_size / 2:
                    # First half: Prey - Levy
                    stepsize[i] = 0.5 * RB[i] * (elite[i] - RB[i] * prey[i])
                    prey[i] = prey[i] + P * CF * stepsize[i]
                else:
                    # Second half: Prey - Brownian
                    stepsize[i] = 0.5 * RL[i] * (elite[i] - RL[i] * prey[i])
                    prey[i] = prey[i] + P * CF * stepsize[i]
        
        # FAZ 3: Son 1/3 iterasyon (Low velocity ratio)
        else:
            for i in range(pop_size):
                # Step size hesapla
                stepsize[i] = 0.5 * RL[i] * (RL[i] * elite[i] - prey[i])
                # Prey güncelle
                prey[i] = elite[i] + P * CF * stepsize[i]
        
        # Sınırları kontrol et
        prey = np.clip(prey, lb, ub)
        
        # Fitness değerlerini güncelle
        for i in range(pop_size):
            new_fitness = objective_func(prey[i])
            
            # Eğer yeni fitness daha iyiyse güncelle
            if new_fitness < fitness[i]:
                fitness[i] = new_fitness
                elite[i] = prey[i].copy()
                elite_fitness[i] = new_fitness
                
                # Global en iyiyi güncelle
                if new_fitness < best_fitness:
                    best_fitness = new_fitness
                    best_solution = prey[i].copy()
        
        # Marine Memory saving
        for i in range(pop_size):
            if fitness[i] < fitness_history[i, 0]:
                fitness_history[i, 0] = fitness[i]
                elite[i] = prey[i].copy()
        
        # FADs effect (Fish Aggregating Devices)
        if np.random.rand() < FADs:
            U = np.random.rand(pop_size, dim) < FADs
            for i in range(pop_size):
                if np.random.rand() < 0.2:  # %20 şansla rastgele güncelle
                    prey[i] = prey[i] + CF * (lb + np.random.rand(dim) * (ub - lb)) * U[i]
                else:
                    r1, r2 = np.random.randint(0, pop_size, 2)
                    prey[i] = prey[i] + (prey[r1] - prey[r2]) * U[i]
        
        convergence_curve[iteration] = best_fitness
        
        # İlerleme göstergesi
        if (iteration + 1) % 100 == 0:
            print(f"Iteration {iteration + 1}/{max_iter}, Best Fitness: {best_fitness:.8f}")
    
    return best_solution, best_fitness, convergence_curve

def levy_flight(n, dim):
    """
    Levy flight dağılımı üretir
    """
    beta = 1.5
    sigma = (math.gamma(1 + beta) * np.sin(np.pi * beta / 2) / 
             (math.gamma((1 + beta) / 2) * beta * (2 ** ((beta - 1) / 2)))) ** (1 / beta)
    
    u = np.random.normal(0, sigma, (n, dim))
    v = np.random.normal(0, 1, (n, dim))
    
    step = u / (np.abs(v) ** (1 / beta))
    return step

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
    
    # MPA parametreleri (orijinal makaleye göre)
    P = 0.5      # Sabit parametre
    FADs = 0.2   # Fish Aggregating Devices etkisi

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

    # Her bir benchmark fonksiyonu için MPA algoritmasını çalıştır
    for benchmark in benchmarks_to_test:
        print(f"\nMPA Algorithm Test - {benchmark['name']} Function")
        print("=" * 50)
        
        best_solution, best_fitness, convergence = mpa(
            objective_func=benchmark['func'],
            dim=dim,
            bounds=benchmark['bounds'],
            pop_size=pop_size,
            max_iter=max_iter,
            P=P,
            FADs=FADs
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
