# CUDA Tabanlı Görüntü İşleme Çekirdeklerinin Geliştirilmesi

## 📌 Projenin Kapsamı
- CUDA programlayarak görüntü işleme algoritmalarını sıfırdan geliştirmek.
- CPU'da çalışan OpenCV gibi kütüphaneler yerine GPU'da paralel programlama yaparak C/C++ düzeyinde kernel tasarımı yapmak.
- GPU'daki bellek türlerine (global memory, unified memory) göre kernel'ların çalıştırılması.
- Farklı görüntü işleme kernel'larının GPU'da asenkron çalıştırılması. 

 ## 📌 Geliştirilen CUDA Kernel'ları
 - BGR2GRAY
 - Binary Threshold
 - Resize
 - Blurlama
 - Median Blur
 - Gaussian Blur
 - Contrast Enhancement
 - Dilation
 - Erosion
 - Horizontal Flip
 - Laplacian Sharpen
 - Background Subtraction
 - Motion Detection
 - Sobel Filter
 - Drawing Rectangle
 - Contour Detection

 ## 📌 GPU'da Kullanılan Bellek Türleri
- CUDA ile yazılan 16 farklı görüntü işleme kernel'ı da hem global memory hem de unified memory için çalışır haldedir.
- Global memory de her bir iş parçacığının (thread) GPU'da DRAM bölümündeki adresleri ile indexleme yapılmıştır.
- Bu indexleme de thread numaraları, blok numaraları ve blok boyutları kullanılmıştır.
- Unified memory de ise CPU ve GPU aynı bellek alanını driver vasıtasıyla kullanmaktadır bir diğer deyişle aynı pointer hem CPU hem de GPU için kullanılmaktadır.
- Proje kapsamında şuanlık pinned memory ve shared memory kullanılmamıştır.

## 📌 Sonuçlar
RTX 3060 ve WSL ile yapılmış testlere göre CUDA'da kernel yazılarak hareketli nesne tespiti pipeline'ının sonuçları:
  - 1920×1080 görüntüde 5 milisaniye
  - 2K görüntüde 6 milisaniye
  - 4K görüntüde 15 milisaniye

 ## 📌 CUDA için Kaynaklar
Aşağıdaki link, CUDA programlama konusunda yazılmış çok iyi bir kaynaktır. CUDA'yı hem donanım seviyesinde (CPU ile GPU'nun PCIe ve NVlink ile bağlantısını, SM blokları ile grid/block/thread hiyerarşisi ve thread indekslemenin nasıl yapılacağı, unified memory ile global memory arasındaki farkların neler olduğu, 32'lik threadlerden oluşan warp mantığını , nvcc ile derlemenin nasıl yapılacağını, CUDA toolkit ve L1/L2 Cache) anlatmakta olup hem de konuların anlaşılması için C++ ile örnek kodlar göstermektedir. Herkese tavsiye ederim. 
- https://docs.nvidia.com/cuda/cuda-programming-guide/index.html
