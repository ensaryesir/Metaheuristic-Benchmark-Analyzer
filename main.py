import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import wilcoxon
import os
import warnings
import time
from datetime import datetime

warnings.filterwarnings('ignore')

# Config'den tüm parametreleri import et
from config import *

# Algoritmaları import et
from algorithms.pso import pso
from algorithms.de import de
from algorithms.mpa import mpa
from algorithms.gto import gto
from algorithms.run import run
from algorithms.rime import rime_improved

# Benchmark fonksiyonlarını import et
from benchmarks.sphere import sphere
from benchmarks.rastrigin import rastrigin
from benchmarks.ackley import ackley

def run_experiment():
    """Ana deney yürütme fonksiyonu"""
    
    # Results klasörünü oluştur
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)
    
    # Algoritma ve fonksiyon eşleştirmeleri
    algorithms = {
        "PSO": pso,
        "DE": de, 
        "MPA": mpa,
        "GTO": gto,
        "RUN": run,
        "RIME": rime_improved 
    }
    
    benchmark_functions = {
        "Sphere": sphere,
        "Rastrigin": rastrigin,
        "Ackley": ackley
    }
    
    # Sonuçları saklamak için veri yapısı
    results = {}
    
    print("=" * 70)
    print("METASEZGİSEL ALGORİTMA KARŞILAŞTIRMA DENEYİ")
    print("=" * 70)
    print(f"Başlangıç Zamanı: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Popülasyon Boyutu: {POP_SIZE}")
    print(f"Maksimum İterasyon: {MAX_ITER}")
    print(f"Problem Boyutu: {DIMENSION}")
    print(f"Çalıştırma Sayısı: {NUM_RUNS}")
    print(f"Temel Arama Uzayı: {BOUNDS}")
    print("=" * 70)
    
    # Deney yürütme döngüsü
    total_start_time = time.time()
    
    for func_name, func in benchmark_functions.items():
        print(f"\n🔍 {func_name} fonksiyonu için deneyler başlıyor...")
        results[func_name] = {}
        
        # Benchmark fonksiyonu için özel sınırları al
        func_bounds = get_benchmark_bounds(func_name.lower())
        
        for alg_name, alg_func in algorithms.items():
            print(f"\n  🚀 {alg_name} algoritması çalıştırılıyor...")
            results[func_name][alg_name] = {
                'best_scores': [],
                'best_solutions': [],
                'convergence_curves': [],
                'computation_times': []
            }
            
            # Algoritma parametrelerini al
            alg_params = get_algorithm_params(alg_name)
            
            for run_num in range(NUM_RUNS):
                start_time = time.time()
                
                print(f"    ⚡ Çalıştırma {run_num + 1}/{NUM_RUNS}...", end=" ")
                
                try:
                    # Algoritmayı çalıştır (parametreleri ile birlikte)
                    if alg_params:
                        best_solution, best_fitness, convergence_curve = alg_func(
                            objective_func=func,
                            dim=DIMENSION,
                            bounds=func_bounds,
                            pop_size=POP_SIZE,
                            max_iter=MAX_ITER,
                            **alg_params
                        )
                    else:
                        best_solution, best_fitness, convergence_curve = alg_func(
                            objective_func=func,
                            dim=DIMENSION,
                            bounds=func_bounds,
                            pop_size=POP_SIZE,
                            max_iter=MAX_ITER
                        )
                    
                    computation_time = time.time() - start_time
                    
                    # Sonuçları kaydet
                    results[func_name][alg_name]['best_scores'].append(best_fitness)
                    results[func_name][alg_name]['best_solutions'].append(best_solution)
                    results[func_name][alg_name]['convergence_curves'].append(convergence_curve)
                    results[func_name][alg_name]['computation_times'].append(computation_time)
                    
                    print(f"Skor: {best_fitness:.4e} | Zaman: {computation_time:.2f}s")
                    
                except Exception as e:
                    print(f"❌ Hata: {str(e)}")
                    # Hata durumunda varsayılan değerler
                    results[func_name][alg_name]['best_scores'].append(float('inf'))
                    results[func_name][alg_name]['best_solutions'].append(None)
                    results[func_name][alg_name]['convergence_curves'].append([])
                    results[func_name][alg_name]['computation_times'].append(0)
    
    total_time = time.time() - total_start_time
    print(f"\n🎉 TÜM DENEYLER TAMAMLANDI! Toplam süre: {total_time:.2f} saniye")
    
    return results, algorithms, benchmark_functions

def analyze_results(results, algorithms, benchmark_functions):
    """Sonuçları analiz et ve raporla"""
    
    print("\n" + "=" * 70)
    print("SONUÇ ANALİZİ VE RAPORLAMA")
    print("=" * 70)
    
    # Timing analizi için veri
    timing_data = []
    
    # Her fonksiyon için analiz
    for func_name in benchmark_functions.keys():
        print(f"\n📊 {func_name} Fonksiyonu İçin Performans Özeti:")
        print("-" * 60)
        
        # Özet verisi oluştur
        summary_data = []
        for alg_name in algorithms.keys():
            scores = results[func_name][alg_name]['best_scores']
            times = results[func_name][alg_name]['computation_times']
            
            # Geçerli skorları filtrele (inf değerlerini çıkar)
            valid_scores = [s for s in scores if s != float('inf')]
            valid_times = [t for i, t in enumerate(times) if scores[i] != float('inf')]
            
            if valid_scores:
                # Timing verisini topla
                timing_data.append({
                    'Function': func_name,
                    'Algorithm': alg_name,
                    'Avg_Time': np.mean(valid_times),
                    'Std_Time': np.std(valid_times),
                    'Min_Time': np.min(valid_times),
                    'Max_Time': np.max(valid_times)
                })
                
                summary_data.append({
                    'Algorithm': alg_name,
                    'Best': np.min(valid_scores),
                    'Worst': np.max(valid_scores),
                    'Mean': np.mean(valid_scores),
                    'Std': np.std(valid_scores),
                    'Median': np.median(valid_scores),
                    'Avg Time (s)': np.mean(valid_times),
                    'Success Rate': f"{(len(valid_scores) / len(scores)) * 100:.1f}%"
                })
            else:
                summary_data.append({
                    'Algorithm': alg_name,
                    'Best': 'N/A',
                    'Worst': 'N/A',
                    'Mean': 'N/A',
                    'Std': 'N/A',
                    'Median': 'N/A',
                    'Avg Time (s)': 'N/A',
                    'Success Rate': '0%'
                })
        
        # DataFrame oluştur ve göster
        df = pd.DataFrame(summary_data)
        print(df.to_string(index=False, float_format='%.4e'))
        
        # CSV olarak kaydet
        csv_filename = os.path.join(RESULTS_DIR, f"performance_summary_{func_name.lower()}.csv")
        df.to_csv(csv_filename, index=False)
        print(f"💾 Özet tablo kaydedildi: {csv_filename}")
    
    # Timing analizini kaydet
    if timing_data:
        timing_df = pd.DataFrame(timing_data)
        timing_file = os.path.join(RESULTS_DIR, "computation_timing.csv")
        timing_df.to_csv(timing_file, index=False)
        print(f"💾 Zamanlama analizi kaydedildi: {timing_file}")

def create_convergence_plots(results, algorithms, benchmark_functions):
    """Yakınsama grafikleri oluştur"""
    
    print("\n📈 GÖRSELLEŞTİRME - Yakınsama Grafikleri")
    
    # Yakınsama grafikleri klasörünü oluştur
    convergence_dir = os.path.join(RESULTS_DIR, "convergence_plots")
    os.makedirs(convergence_dir, exist_ok=True)
    
    for func_name in benchmark_functions.keys():
        plt.figure(figsize=FIGURE_SIZE)
        
        for alg_name in algorithms.keys():
            convergence_curves = results[func_name][alg_name]['convergence_curves']
            
            # Geçerli eğrileri filtrele
            valid_curves = [curve for curve in convergence_curves if len(curve) > 0]
            
            if valid_curves:
                # Ortalama yakınsama eğrisini hesapla
                min_length = min(len(curve) for curve in valid_curves)
                truncated_curves = [curve[:min_length] for curve in valid_curves]
                avg_convergence = np.mean(truncated_curves, axis=0)
                
                # Grafik ayarları
                alg_config = ALGORITHMS.get(alg_name, {})
                color = alg_config.get('color', None)
                marker = alg_config.get('marker', None)
                linestyle = alg_config.get('linestyle', '-')
                
                plt.plot(avg_convergence, label=alg_name, linewidth=2, 
                        color=color, marker=marker, markevery=50, linestyle=linestyle)
        
        plt.title(f'{func_name} Fonksiyonu - Yakınsama Eğrileri', 
                 fontsize=14, fontweight='bold')
        plt.xlabel('İterasyon', fontsize=12)
        plt.ylabel('En İyi Fitness', fontsize=12)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.yscale('log')
        
        # Grafiği kaydet
        plot_filename = os.path.join(convergence_dir, f"{func_name.lower()}_convergence.png")
        plt.savefig(plot_filename, dpi=DPI, bbox_inches='tight')
        plt.close()
        print(f"💾 Yakınsama grafiği kaydedildi: {plot_filename}")

def create_boxplots(results, algorithms, benchmark_functions):
    """Kutu grafikleri oluştur"""
    
    print("\n📊 GÖRSELLEŞTİRME - Kutu Grafikleri")
    
    # Kutu grafikleri klasörünü oluştur
    boxplot_dir = os.path.join(RESULTS_DIR, "boxplots")
    os.makedirs(boxplot_dir, exist_ok=True)
    
    for func_name in benchmark_functions.keys():
        plt.figure(figsize=FIGURE_SIZE)
        
        # Veriyi hazırla
        data = []
        labels = []
        
        for alg_name in algorithms.keys():
            scores = results[func_name][alg_name]['best_scores']
            valid_scores = [s for s in scores if s != float('inf')]
            
            if valid_scores:
                data.append(valid_scores)
                labels.append(alg_name)
        
        # Kutu grafiği oluştur
        plt.boxplot(data, labels=labels, patch_artist=True)
        plt.title(f'{func_name} Fonksiyonu - Performans Dağılımları', 
                 fontsize=14, fontweight='bold')
        plt.xlabel('Algoritmalar', fontsize=12)
        plt.ylabel('Fitness Değerleri', fontsize=12)
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        plt.yscale('log')
        
        # Grafiği kaydet
        plot_filename = os.path.join(boxplot_dir, f"{func_name.lower()}_boxplot.png")
        plt.savefig(plot_filename, dpi=DPI, bbox_inches='tight')
        plt.close()
        print(f"💾 Kutu grafiği kaydedildi: {plot_filename}")

def statistical_analysis(results, algorithms, benchmark_functions):
    """İstatistiksel analiz yap"""
    
    print("\n📊 İSTATİSTİKSEL TESTLER - Wilcoxon Signed-Rank Test")
    
    for func_name in benchmark_functions.keys():
        print(f"\n🔬 {func_name} Fonksiyonu İçin İstatistiksel Analiz:")
        print("-" * 60)
        
        # En iyi ortalama skoru elde eden algoritmayı bul
        mean_scores = {}
        for alg_name in algorithms.keys():
            scores = results[func_name][alg_name]['best_scores']
            valid_scores = [s for s in scores if s != float('inf')]
            if valid_scores:
                mean_scores[alg_name] = np.mean(valid_scores)
            else:
                mean_scores[alg_name] = float('inf')
        
        champion = min(mean_scores, key=mean_scores.get)
        print(f"🏆 Şampiyon Algoritma: {champion} (Ortalama: {mean_scores[champion]:.6e})")
        
        # Wilcoxon testleri
        champion_scores = [s for s in results[func_name][champion]['best_scores'] if s != float('inf')]
        test_results = []
        
        print("\nWilcoxon Signed-Rank Test Sonuçları:")
        print("Algoritma\t\tp-değeri\t\tAnlamlı Fark")
        print("-" * 50)
        
        for alg_name in algorithms.keys():
            if alg_name != champion:
                other_scores = [s for s in results[func_name][alg_name]['best_scores'] if s != float('inf')]
                
                if champion_scores and other_scores and len(champion_scores) == len(other_scores):
                    try:
                        statistic, p_value = wilcoxon(champion_scores, other_scores)
                        significant = "✅ Evet" if p_value < ALPHA else "❌ Hayır"
                        print(f"{alg_name}\t\t{p_value:.6f}\t\t{significant}")
                        test_results.append(f"{alg_name}: p={p_value:.6f}, Anlamlı={significant}")
                    except Exception as e:
                        print(f"{alg_name}\t\tTest yapılamadı\t\t-")
                        test_results.append(f"{alg_name}: Test yapılamadı - {str(e)}")
                else:
                    print(f"{alg_name}\t\tVeri uyumsuz\t\t-")
                    test_results.append(f"{alg_name}: Veri uyumsuz")
        
        # İstatistiksel sonuçları dosyaya yaz
        stats_filename = os.path.join(RESULTS_DIR, f"statistical_analysis_{func_name.lower()}.txt")
        with open(stats_filename, 'w', encoding='utf-8') as f:
            f.write(f"{func_name} Fonksiyonu İstatistiksel Analiz Sonuçları\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Şampiyon Algoritma: {champion}\n")
            f.write(f"Şampiyon Ortalama Skor: {mean_scores[champion]:.6e}\n\n")
            f.write("Wilcoxon Signed-Rank Test Sonuçları:\n")
            f.write("-" * 40 + "\n")
            for result in test_results:
                f.write(result + "\n")
            f.write(f"\nNot: p < {ALPHA} ise fark istatistiksel olarak anlamlıdır.\n")
        
        print(f"💾 İstatistiksel sonuçlar kaydedildi: {stats_filename}")

def main():
    """Ana yürütme fonksiyonu"""
    
    try:
        # Deneyleri çalıştır
        results, algorithms, benchmark_functions = run_experiment()
        
        # Sonuçları analiz et
        analyze_results(results, algorithms, benchmark_functions)
        
        # Grafikleri oluştur
        create_convergence_plots(results, algorithms, benchmark_functions)
        create_boxplots(results, algorithms, benchmark_functions)
        
        # İstatistiksel analiz yap
        statistical_analysis(results, algorithms, benchmark_functions)
        
        print("\n" + "=" * 70)
        print("🎊 TÜM ANALİZLER TAMAMLANDI!")
        print(f"📁 Sonuçlar '{RESULTS_DIR}' klasöründe saklandı.")
        print("=" * 70)
        
    except Exception as e:
        print(f"\n❌ CRITICAL ERROR: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
