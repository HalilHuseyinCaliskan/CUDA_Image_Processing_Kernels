# CUDA Tabanlı Görüntü İşleme Çekirdeklerinin Geliştirilmesi

## 📌 Projenin Kapsamı
- CUDA programlayarak görüntü işleme algoritmalarını sıfırdan geliştirmek.
- CPU'da çalışan OpenCV gibi kütüphaneler yerine GPU'da paralel programlama yaparal C/C++ düzeyinde kernel tasarımı yapmak.
- GPU'daki bellek türlerine (global memory, unified memory) göre kernel'ların çalıştırılması.

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

   ## 📌 Sonuçlar
RTX 3060'daki testlere göre CUDA'da kernel yazılarak hareket algılama pipeline'ının sonuçları:
  - 1920×1080 görüntüde 5 milisaniye
  - 2K görüntüde 6 milisaniye
  - 4K görüntüde 15 milisaniye
  
     
