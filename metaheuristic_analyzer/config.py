# config.py
# Merkezi konfigürasyon dosyası - tüm deney parametreleri

import numpy as np

# =============================================================================
# TEMEL DENEY AYARLARI
# =============================================================================

# Algoritma ve deney parametreleri
POP_SIZE = 30          # Popülasyon boyutu (tüm algoritmalar için)
MAX_ITER = 500         # Maksimum iterasyon sayısı
DIMENSION = 30         # Problem boyutu
NUM_RUNS = 30          # Her algoritma için bağımsız çalıştırma sayısı
RANDOM_SEED = 42       # Tekrarlanabilirlik için seed değeri

# Arama uzayı sınırları
BOUNDS = (-10, 10)     # Temel arama uzayı

# Benchmark fonksiyonları için özel sınırlar
BENCHMARK_BOUNDS = {
    'sphere': (-5.12, 5.12),
    'rastrigin': (-5.12, 5.12), 
    'ackley': (-32.768, 32.768)
}

# =============================================================================
# ALGORİTMA ÖZEL PARAMETRELERİ
# =============================================================================

# Particle Swarm Optimization (PSO) parametreleri
PSO_PARAMS = {
    'w': 0.9,    # Atalet ağırlığı
    'c1': 2.0,   # Bilişsel katsayı
    'c2': 2.0    # Sosyal katsayı
}

# Differential Evolution (DE) parametreleri  
DE_PARAMS = {
    'F': 0.5,    # Mutasyon faktörü
    'CR': 0.9    # Çaprazlama oranı
}

# Marine Predators Algorithm (MPA) parametreleri
MPA_PARAMS = {
    'P': 0.5,    # Sabit parametre
    'FADs': 0.2  # Fish Aggregating Devices etkisi
}

# Artificial Gorilla Troops Optimizer (GTO) parametreleri
GTO_PARAMS = {
    'beta': 3.0,  # Sürü liderliği parametresi
    'p': 0.03     # Rastgele hareket olasılığı
}

# Rime Optimization Algorithm (RIME) parametreleri  
RIME_PARAMS = {
    # Orijinal RIME algoritması parametre kabul ETMEZ
    # Bu yüzden boş bırakıyoruz
}

# Runge Kutta Optimizer (RUN) parametreleri
RUN_PARAMS = {
    # RUN algoritması adaptif parametreler kullanır
}

# =============================================================================
# ÇALIŞTIRILACAK ALGORİTMALAR
# =============================================================================

ALGORITHMS = {
    'PSO': {
        'function': 'pso',
        'params': PSO_PARAMS,
        'color': '#1f77b4',  # Mavi
        'marker': 'o'
    },
    'DE': {
        'function': 'de', 
        'params': DE_PARAMS,
        'color': '#ff7f0e',  # Turuncu
        'marker': 's'
    },
    'MPA': {
        'function': 'mpa',
        'params': MPA_PARAMS,
        'color': '#2ca02c',  # Yeşil
        'marker': '^'
    },
    'GTO': {
        'function': 'gto',
        'params': GTO_PARAMS, 
        'color': '#d62728',  # Kırmızı
        'marker': 'D'
    },
    'RIME': {
        'function': 'rime',
        'params': {},  # BOŞ parametre listesi
        'color': '#9467bd',  # Mor
        'marker': 'v'
    },
    'RUN': {
        'function': 'run',
        'params': RUN_PARAMS,
        'color': '#8c564b',  # Kahverengi
        'marker': 'p'
    }
}

# =============================================================================
# BENCHMARK FONKSİYONLARI
# =============================================================================

BENCHMARK_FUNCTIONS = {
    'sphere': {
        'function': 'sphere',
        'global_min': 0.0,
        'optimal_solution': np.zeros(DIMENSION),
        'bounds': (-5.12, 5.12)
    },
    'rastrigin': {
        'function': 'rastrigin',
        'global_min': 0.0, 
        'optimal_solution': np.zeros(DIMENSION),
        'bounds': (-5.12, 5.12)
    },
    'ackley': {
        'function': 'ackley',
        'global_min': 0.0,
        'optimal_solution': np.zeros(DIMENSION), 
        'bounds': (-32.768, 32.768)
    }
}

# =============================================================================
# DOSYA ve GRAFİK AYARLARI
# =============================================================================

# Sonuç dosyaları için yollar
RESULTS_DIR = "results"
SUMMARY_FILE_PREFIX = "summary_"
CONVERGENCE_FILE_PREFIX = "convergence_" 
STATS_FILE_PREFIX = "stats_"
BOXPLOT_FILE_PREFIX = "boxplot_"

# Grafik ayarları
FIGURE_SIZE = (12, 8)
DPI = 300
FONT_SIZE = 12
LEGEND_FONT_SIZE = 10

# =============================================================================
# İSTATİSTİKSEL ANALİZ AYARLARI
# =============================================================================

# Wilcoxon işaretli sıra testi için
ALPHA = 0.05  # Anlamlılık seviyesi

# Performans metrikleri
METRICS = ['best_fitness', 'mean_fitness', 'std_fitness', 'convergence_speed']

def get_algorithm_params(algorithm_name):
    """Algoritma parametrelerini getir"""
    return ALGORITHMS[algorithm_name]['params']

def get_benchmark_bounds(function_name):
    """Benchmark fonksiyonu için sınırları getir"""
    return BENCHMARK_FUNCTIONS[function_name].get('bounds', BOUNDS)

def validate_config():
    """Konfigürasyon ayarlarını doğrula"""
    assert POP_SIZE > 0, "POP_SIZE pozitif olmalı"
    assert MAX_ITER > 0, "MAX_ITER pozitif olmalı" 
    assert DIMENSION > 0, "DIMENSION pozitif olmalı"
    assert NUM_RUNS > 0, "NUM_RUNS pozitif olmalı"
    assert BOUNDS[0] < BOUNDS[1], "BOUNDS geçersiz"
    
    print("✓ Konfigürasyon ayarları doğrulandı")

# Konfigürasyon doğrulaması
if __name__ != "__main__":
    validate_config()