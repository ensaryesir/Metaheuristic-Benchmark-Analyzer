import numpy as np

def rastrigin(x):
    """
    Rastrigin fonksiyonu - multimodal bir test fonksiyonu
    
    Args:
        x: numpy array, çözüm vektörü
        
    Returns:
        float: fonksiyon değeri
    """
    A = 10
    n = len(x)
    return A * n + np.sum(x**2 - A * np.cos(2 * np.pi * x))

# Test kodu
if __name__ == "__main__":
    # Global minimum testi
    test_point = np.zeros(3)
    print(f"Rastrigin(0,0,0) = {rastrigin(test_point):.6f}")  # 0.0 çıkmalı
    
    # Başka bir test noktası
    test_point2 = np.array([1.0, 1.0, 1.0])
    print(f"Rastrigin(1,1,1) = {rastrigin(test_point2):.6f}")  # 3.0 çıkmalı
    
    # Sınır testi
    test_point3 = np.array([2.0, 2.0, 2.0])
    print(f"Rastrigin(2,2,2) = {rastrigin(test_point3):.6f}")  # 12.0 çıkmalı
