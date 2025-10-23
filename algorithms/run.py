import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np

def run(objective_func, dim, bounds, pop_size, max_iter):
    """
    Runge Kutta Optimizer (RUN) - Enhanced Version
    
    Bu çalışmada, orijinal RUN algoritmasının temel bileşenleri korunarak
    aşağıdaki iyileştirmeler uygulanmıştır:
    
    1. Enhanced Solution Quality (ESQ): Popülasyon ortalaması ve adaptif 
       delta mekanizması ile akıllı arama stratejisi
    
    2. Smart Pool Update: Her 10 iterasyonda bir kötü çözümlerin yeniden
       başlatılması ile çeşitlilik korunması
    
    3. Adaptive Parameters: Zamanla değişen parametreler ile keşif/sömürü
       dengesinin optimize edilmesi
    
    Orijinal Makale:
    Ahmadianfar, I., Heidari, A. A., Gandomi, A. H., Chu, X., & Chen, H. (2021). 
    "RUN beyond the metaphor: An efficient optimization algorithm based on Runge Kutta method"
    Expert Systems with Applications, 181, 115079.
    
    DOI: https://doi.org/10.1016/j.eswa.2021.115079
    URL: https://www.sciencedirect.com/science/article/abs/pii/S0957417421005012
    
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
    X = np.random.uniform(lb, ub, (pop_size, dim))
    
    # Fitness değerlerini hesapla
    fitness = np.array([objective_func(ind) for ind in X])
    
    # En iyi çözüm
    best_idx = np.argmin(fitness)
    best_solution = X[best_idx].copy()
    best_fitness = fitness[best_idx]
    
    convergence_curve = np.zeros(max_iter)
    
    for t in range(max_iter):
        # Parametreleri güncelle
        alpha = 20 * np.exp(-3 * t / max_iter)  # Adaptive parameter
        beta = 0.2 * (1 - t / max_iter)         # Search step size
        
        for i in range(pop_size):
            # Enhanced solution quality (ESQ) - Orijinal RUN mekanizması
            if np.random.rand() < 0.5:
                # RUN core mechanism - Runge-Kutta tabanlı arama
                
                # Rastgele üç çözüm seç
                r1, r2, r3 = np.random.choice(pop_size, 3, replace=False)
                
                # Ortalama pozisyon
                Xavg = (X[r1] + X[r2] + X[r3]) / 3
                
                # Adaptive step size
                delta = 2 * np.random.rand() * np.abs(Xavg - X[i])
                
                # Runge-Kutta katsayıları
                # K1 = f(t, X)
                K1 = 0.5 * np.random.rand() * (best_solution - X[i] + delta)
                
                # K2 = f(t + h/2, X + h*K1/2)
                XK2 = X[i] + K1 * np.random.rand()
                K2 = 0.5 * np.random.rand() * (best_solution - XK2 + delta / 2)
                
                # K3 = f(t + h/2, X + h*K2/2)  
                XK3 = X[i] + K2 * np.random.rand()
                K3 = 0.5 * np.random.rand() * (best_solution - XK3 + delta / 2)
                
                # K4 = f(t + h, X + h*K3)
                XK4 = X[i] + K3 * np.random.rand()
                K4 = 0.5 * np.random.rand() * (best_solution - XK4 + delta)
                
                # Runge-Kutta güncellemesi
                RK_update = (K1 + 2*K2 + 2*K3 + K4) / 6
                
                # Yeni pozisyon
                X_new = X[i] + alpha * RK_update
                
            else:
                # Keşif mekanizması
                if np.random.rand() < 0.5:
                    # Global keşif
                    r = np.random.randint(pop_size)
                    X_new = best_solution + np.random.randn(dim) * beta * (X[r] - X[i])
                else:
                    # Yerel keşif
                    X_new = X[i] + np.random.randn(dim) * beta * (best_solution - X[i])
            
            # Sınırları kontrol et
            X_new = np.clip(X_new, lb, ub)
            
            # Fitness değerlendirme
            new_fitness = objective_func(X_new)
            
            # Greedy selection
            if new_fitness < fitness[i]:
                X[i] = X_new
                fitness[i] = new_fitness
                
                # En iyi çözümü güncelle
                if new_fitness < best_fitness:
                    best_fitness = new_fitness
                    best_solution = X_new.copy()
        
        # Smart pool update - Orijinal RUN özelliği
        if t % 10 == 0:
            # Popülasyonun en kötü %20'sini yeniden başlat
            worst_indices = np.argsort(fitness)[-pop_size//5:]
            for idx in worst_indices:
                if np.random.rand() < 0.5:
                    # Rastgele yeniden başlatma
                    X[idx] = np.random.uniform(lb, ub, dim)
                    fitness[idx] = objective_func(X[idx])
                else:
                    # En iyi çözüm etrafında yeniden başlatma
                    X[idx] = best_solution + np.random.randn(dim) * 0.1 * (ub - lb)
                    X[idx] = np.clip(X[idx], lb, ub)
                    fitness[idx] = objective_func(X[idx])
                
                # En iyi çözümü kontrol et
                if fitness[idx] < best_fitness:
                    best_fitness = fitness[idx]
                    best_solution = X[idx].copy()
        
        convergence_curve[t] = best_fitness
        
        # İlerleme göstergesi
        if (t + 1) % 100 == 0:
            print(f"Iteration {t + 1}/{max_iter}, Best Fitness: {best_fitness:.8f}")
    
    return best_solution, best_fitness, convergence_curve

# Test kodu
if __name__ == "__main__":
    # Test fonksiyonu olarak sphere kullan
    from benchmarks.sphere import sphere
    
    # Algoritma parametreleri
    dim = 10
    bounds = (-5.12, 5.12)
    pop_size = 30
    max_iter = 500
    
    print("RUN Algorithm (Enhanced Version) Test - Sphere Function")
    print("=" * 60)
    
    best_solution, best_fitness, convergence = run(
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
    print(f"Improvement: {convergence[0] / convergence[-1] if convergence[-1] != 0 else 'N/A':.2e}x")