import numpy as np

def sphere(x):
    """
    Sphere fonksiyonu - basit ve unimodal bir test fonksiyonu
    
    Args:
        x: numpy array, çözüm vektörü
        
    Returns:
        float: fonksiyon değeri
    """
    return np.sum(x**2)

# Test kodu
if __name__ == "__main__":
    # Global minimum testi
    test_point = np.zeros(3)
    print(f"Sphere(0,0,0) = {sphere(test_point):.6f}")  # 0.0 çıkmalı
    
    # Başka bir test noktası
    test_point2 = np.array([1.0, 2.0, 3.0])
    print(f"Sphere(1,2,3) = {sphere(test_point2):.6f}")  # 1+4+9=14 çıkmalı
