import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np

def gto(objective_func, dim, bounds, pop_size, max_iter, beta=3.0, p=0.03):
    """
    Artificial Gorilla Troops Optimizer (GTO) - Orijinal Makaleye Uygun
    
    Orijinal Makale: "GTO: A New Metaheuristic Algorithm for Global Optimization"
    
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
    # Test fonksiyonu olarak sphere kullan
    from benchmarks.sphere import sphere
    
    # Algoritma parametreleri
    dim = 10
    bounds = (-5.12, 5.12)
    pop_size = 30
    max_iter = 500
    
    print("GTO Algorithm (Original) Test - Sphere Function")
    print("=" * 50)
    
    best_solution, best_fitness, convergence = gto(
        objective_func=sphere,
        dim=dim,
        bounds=bounds,
        pop_size=pop_size,
        max_iter=max_iter,
        beta=3.0,
        p=0.03
    )
    
    print(f"\nResults:")
    print(f"Best Solution: {best_solution[:5]}...")
    print(f"Best Fitness: {best_fitness:.10f}")
    print(f"Final Convergence: {convergence[-1]:.10f}")
    
    # Yakınsama analizi
    print(f"Initial Best: {convergence[0]:.4f}")
    print(f"Final Best: {convergence[-1]:.10f}")
    print(f"Improvement: {convergence[0] / convergence[-1] if convergence[-1] != 0 else 'N/A':.2e}x")