# Merkezi konfigürasyon dosyası - tüm deney parametreleri

import numpy as np
import os
from datetime import datetime

# =============================================================================
# TEMEL DENEY AYARLARI
# =============================================================================

# Algoritma ve deney parametreleri
POP_SIZE = 30          # Popülasyon boyutu (tüm algoritmalar için)
MAX_ITER = 1000        # Maksimum iterasyon sayısı (zorlu fonksiyonlar için artırıldı)
DIMENSION = 10         # Problem boyutu (testlerde kullandığımız değer)
NUM_RUNS = 10          # Her algoritma için bağımsız çalıştırma sayısı (hız için)
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
# ALGORİTMA ÖZEL PARAMETRELERİ (ORİJİNAL MAKALELERE GÖRE)
# =============================================================================

# Particle Swarm Optimization (PSO) parametreleri
# Orijinal Makale: Kennedy & Eberhart, 1995
PSO_PARAMS = {
    'w': 0.729,    # Atalet ağırlığı (orijinal makalede 0.729)
    'c1': 1.49445, # Bilişsel katsayı (orijinal makalede 1.49445)
    'c2': 1.49445  # Sosyal katsayı (orijinal makalede 1.49445)
}

# Differential Evolution (DE) parametreleri  
# Orijinal Makale: Storn & Price, 1997
DE_PARAMS = {
    'F': 0.5,    # Mutasyon faktörü (orijinal makalede 0.5)
    'CR': 0.9    # Çaprazlama oranı (orijinal makalede 0.9)
}

# Marine Predators Algorithm (MPA) parametreleri
# Orijinal Makale: Faramarzi et al., 2020
MPA_PARAMS = {
    'P': 0.5,    # Sabit parametre (orijinal makalede 0.5)
    'FADs': 0.2  # Fish Aggregating Devices etkisi (orijinal makalede 0.2)
}

# Artificial Gorilla Troops Optimizer (GTO) parametreleri
# Orijinal Makale: Abdollahzadeh et al., 2021
GTO_PARAMS = {
    'beta': 3.0,  # Sürü liderliği parametresi (orijinal makalede 3.0)
    'p': 0.03     # Rastgele hareket olasılığı (orijinal makalede 0.03)
}

# Geliştirilmiş Rime Optimization Algorithm (RIME) parametreleri  
# NOT: Orijinal RIME parametresizdir, geliştirilmiş versiyon için sosyal bileşen parametreleri
RIME_PARAMS = {
    'social_rate': 0.3,      # Sosyal bileşen kullanma olasılığı
    'noise_factor': 0.05,    # Gürültü faktörü
    'reset_interval': 200    # Periyodik sıfırlama aralığı
}

# Runge Kutta Optimizer (RUN) parametreleri
# Orijinal Makale: Ahmadianfar et al., 2021
RUN_PARAMS = {
    # RUN algoritması tamamen adaptif parametreler kullanır
    # Orijinal makalede ek parametre yok
}

# =============================================================================
# ÇALIŞTIRILACAK ALGORİTMALAR
# =============================================================================

ALGORITHMS = {
    'PSO': {
        'function': 'pso',
        'params': PSO_PARAMS,
        'color': '#1f77b4',  # Mavi
        'marker': 'o',
        'linestyle': '-'
    },
    'DE': {
        'function': 'de', 
        'params': DE_PARAMS,
        'color': '#ff7f0e',  # Turuncu
        'marker': 's',
        'linestyle': '--'
    },
    'MPA': {
        'function': 'mpa',
        'params': MPA_PARAMS,
        'color': '#2ca02c',  # Yeşil
        'marker': '^',
        'linestyle': '-.'
    },
    'GTO': {
        'function': 'gto',
        'params': GTO_PARAMS, 
        'color': '#d62728',  # Kırmızı
        'marker': 'D',
        'linestyle': ':'
    },
    'RIME': {
        'function': 'rime_improved',  # Geliştirilmiş versiyon
        'params': RIME_PARAMS,
        'color': '#9467bd',  # Mor
        'marker': 'v',
        'linestyle': '-'
    },
    'RUN': {
        'function': 'run',
        'params': RUN_PARAMS,
        'color': '#8c564b',  # Kahverengi
        'marker': 'p',
        'linestyle': '--'
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
        'bounds': (-5.12, 5.12),
        'difficulty': 'low'
    },
    'rastrigin': {
        'function': 'rastrigin',
        'global_min': 0.0, 
        'optimal_solution': np.zeros(DIMENSION),
        'bounds': (-5.12, 5.12),
        'difficulty': 'high'
    },
    'ackley': {
        'function': 'ackley',
        'global_min': 0.0,
        'optimal_solution': np.zeros(DIMENSION), 
        'bounds': (-32.768, 32.768),
        'difficulty': 'medium'
    }
}

# =============================================================================
# DOSYA ve GRAFİK AYARLARI
# =============================================================================

# Dinamik sonuç klasörü oluşturma
EXPERIMENT_TIMESTAMP = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
RESULTS_DIR = os.path.join("results", f"experiment_{EXPERIMENT_TIMESTAMP}")
os.makedirs(RESULTS_DIR, exist_ok=True)

# Sonuç dosyaları için yollar
SUMMARY_FILE = os.path.join(RESULTS_DIR, "performance_summary.csv")
CONVERGENCE_DIR = os.path.join(RESULTS_DIR, "convergence_plots")
STATS_FILE = os.path.join(RESULTS_DIR, "statistical_analysis.txt")
BOXPLOT_DIR = os.path.join(RESULTS_DIR, "boxplots")
TIMING_FILE = os.path.join(RESULTS_DIR, "computation_timing.csv")

# Grafik klasörlerini oluştur
os.makedirs(CONVERGENCE_DIR, exist_ok=True)
os.makedirs(BOXPLOT_DIR, exist_ok=True)

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
METRICS = [
    'best_fitness', 
    'mean_fitness', 
    'std_fitness', 
    'convergence_iteration',
    'computation_time',
    'success_rate'
]

# Yakınsama kriteri
CONVERGENCE_THRESHOLD = 1e-10  # Yakınsama kabul eşiği
MAX_TIME_SECONDS = 3600        # Maksimum çalışma süresi (1 saat)

def get_algorithm_params(algorithm_name):
    """Algoritma parametrelerini getir"""
    return ALGORITHMS[algorithm_name]['params']

def get_benchmark_bounds(function_name):
    """Benchmark fonksiyonu için sınırları getir"""
    return BENCHMARK_FUNCTIONS[function_name].get('bounds', BOUNDS)

def get_results_directory():
    """Sonuçlar dizinini getir"""
    return RESULTS_DIR

def validate_config():
    """Konfigürasyon ayarlarını doğrula"""
    assert POP_SIZE > 0, "POP_SIZE pozitif olmalı"
    assert MAX_ITER > 0, "MAX_ITER pozitif olmalı" 
    assert DIMENSION > 0, "DIMENSION pozitif olmalı"
    assert NUM_RUNS > 0, "NUM_RUNS pozitif olmalı"
    assert BOUNDS[0] < BOUNDS[1], "BOUNDS geçersiz"
    
    # Algoritma parametrelerini kontrol et
    for algo_name, algo_config in ALGORITHMS.items():
        assert 'function' in algo_config, f"{algo_name} için function tanımlı değil"
        assert 'params' in algo_config, f"{algo_name} için params tanımlı değil"
    
    print("✓ Konfigürasyon ayarları doğrulandı")
    print(f"✓ Sonuçlar kaydedilecek: {RESULTS_DIR}")

# Konfigürasyon doğrulaması
if __name__ != "__main__":
    validate_config()