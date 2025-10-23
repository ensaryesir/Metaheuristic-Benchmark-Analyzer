import numpy as np

def ackley(x):
    """
    Ackley fonksiyonu - multimodal bir test fonksiyonu
    
    Args:
        x: numpy array, çözüm vektörü
        
    Returns:
        float: fonksiyon değeri
    """
    a = 20
    b = 0.2
    c = 2 * np.pi
    n = len(x)
    
    sum1 = np.sum(x**2)
    sum2 = np.sum(np.cos(c * x))
    
    term1 = -a * np.exp(-b * np.sqrt(sum1 / n))
    term2 = -np.exp(sum2 / n)
    
    return term1 + term2 + a + np.exp(1)

# Test kodu
if __name__ == "__main__":
    # Global minimum testi
    test_point = np.zeros(5)
    print(f"Ackley(0,0,0,0,0) = {ackley(test_point):.6f}")  # 0.0 çıkmalı
    
    # Başka bir test noktası
    test_point2 = np.array([1.0, 2.0, 1.0, 2.0, 1.0])
    print(f"Ackley(1,2,1,2,1) = {ackley(test_point2):.6f}")
    
    # Tek boyutlu test
    test_point3 = np.array([1.0])
    print(f"Ackley(1) = {ackley(test_point3):.6f}")