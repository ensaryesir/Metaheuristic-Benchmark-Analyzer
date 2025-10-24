import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np

"""
    Particle Swarm Optimization (PSO) algoritması
    
    ⚡ Algoritma Özellikleri:
    - Temel Fikir: Kuş sürüsü ve balık sürüsü davranışları
    - Güçlü Yönler: Hızlı yakınsama, sezgisel anlaşılırlık
    - Zayıf Yönler: Erken yakınsama, parametre hassasiyeti
    
    🔧 Parametre Tavsiyeleri:
    - w = 0.9 → 0.4 (lineer azalma)
    - c1, c2 = 2.0 (Standart), Keşif için: c1=2.5, c2=1.5
    
    Orijinal Makale:
    Kennedy, J., & Eberhart, R. (1995). 
    "Particle Swarm Optimization"
    Proceedings of ICNN'95 - International Conference on Neural Networks.
    
    Args:
        objective_func: Optimize edilecek fonksiyon
        dim: Problem boyutu
        bounds: Arama uzayı sınırları [(min, max)] veya (min, max)
        pop_size: Popülasyon boyutu
        max_iter: Maksimum iterasyon sayısı
        w: Atalet katsayısı (varsayılan: 0.9)
        c1: Bilişsel katsayı (varsayılan: 2.0)
        c2: Sosyal katsayı (varsayılan: 2.0)
        
    Returns:
        tuple: (best_solution, best_fitness, convergence_curve)
"""

def pso(objective_func, dim, bounds, pop_size, max_iter, w=0.9, c1=2.0, c2=2.0):
    
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
    
    # Hız sınırlarını hesapla (genellikle pozisyon sınırlarının %10-20'si)
    v_max = 0.2 * (ub - lb)
    v_min = -v_max
    
    # Popülasyonu başlat
    positions = np.random.uniform(lb, ub, (pop_size, dim))
    velocities = np.random.uniform(v_min, v_max, (pop_size, dim))
    
    # Kişisel en iyi pozisyonları
    pbest_positions = positions.copy()
    pbest_scores = np.array([objective_func(pos) for pos in positions])
    
    # Global en iyi pozisyon
    gbest_idx = np.argmin(pbest_scores)
    gbest_position = pbest_positions[gbest_idx].copy()
    gbest_score = pbest_scores[gbest_idx]
    
    convergence_curve = np.zeros(max_iter)
    
    for iteration in range(max_iter):
        # Atalet katsayısını lineer azalt (0.9 → 0.4)
        current_w = w - (w - 0.4) * iteration / max_iter
        
        for i in range(pop_size):
            # Rastgele katsayılar
            r1, r2 = np.random.rand(2)
            
            # Hız güncelleme
            cognitive_component = c1 * r1 * (pbest_positions[i] - positions[i])
            social_component = c2 * r2 * (gbest_position - positions[i])
            
            velocities[i] = (current_w * velocities[i] + 
                           cognitive_component + social_component)
            
            # Hız sınırlarını uygula
            velocities[i] = np.clip(velocities[i], v_min, v_max)
            
            # Pozisyon güncelleme
            positions[i] += velocities[i]
            
            # Pozisyon sınırlarını uygula
            positions[i] = np.clip(positions[i], lb, ub)
            
            # Fitness değerlendirme
            current_score = objective_func(positions[i])
            
            # Kişisel en iyiyi güncelle
            if current_score < pbest_scores[i]:
                pbest_scores[i] = current_score
                pbest_positions[i] = positions[i].copy()
                
                # Global en iyiyi güncelle
                if current_score < gbest_score:
                    gbest_score = current_score
                    gbest_position = positions[i].copy()
        
        convergence_curve[iteration] = gbest_score
        
        # İlerleme göstergesi
        if (iteration + 1) % 100 == 0:
            print(f"Iteration {iteration + 1}/{max_iter}, Best Fitness: {gbest_score:.8f}")
    
    return gbest_position, gbest_score, convergence_curve

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
    
    # PSO parametreleri
    w = 0.9      # Atalet katsayısı
    c1 = 2.0     # Bilişsel katsayı
    c2 = 2.0     # Sosyal katsayı

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

    # Her bir benchmark fonksiyonu için PSO algoritmasını çalıştır
    for benchmark in benchmarks_to_test:
        print(f"\nPSO Algorithm Test - {benchmark['name']} Function")
        print("=" * 50)
        
        best_solution, best_fitness, convergence = pso(
            objective_func=benchmark['func'],
            dim=dim,
            bounds=benchmark['bounds'],
            pop_size=pop_size,
            max_iter=max_iter,
            w=w,
            c1=c1,
            c2=c2
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
