import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np

def pso(objective_func, dim, bounds, pop_size, max_iter, w=0.9, c1=2.0, c2=2.0):
    """
    Particle Swarm Optimization (PSO) algoritması
    
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
    # Test fonksiyonu olarak sphere kullan
    from benchmarks.sphere import sphere
    
    # Algoritma parametreleri
    dim = 10
    bounds = (-5.12, 5.12)
    pop_size = 30
    max_iter = 500
    
    print("PSO Algorithm Test - Sphere Function")
    print("=" * 40)
    
    best_solution, best_fitness, convergence = pso(
        objective_func=sphere,
        dim=dim,
        bounds=bounds,
        pop_size=pop_size,
        max_iter=max_iter
    )
    
    print(f"\nResults:")
    print(f"Best Solution: {best_solution[:5]}...")  # İlk 5 değeri göster
    print(f"Best Fitness: {best_fitness:.10f}")
    print(f"Final Convergence: {convergence[-1]:.10f}")
    
    # Yakınsama analizi
    print(f"Initial Best: {convergence[0]:.4f}")
    print(f"Final Best: {convergence[-1]:.10f}")
    print(f"Improvement: {convergence[0] / convergence[-1] if convergence[-1] != 0 else 'N/A':.2e}x")