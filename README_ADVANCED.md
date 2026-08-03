# Advanced Tear Film Analysis System

## 🎯 Genel Bakış

Bu sistem, PTLib JavaScript kütüphanesinden esinlenerek geliştirilmiş, gelişmiş gözyaşı filmi analiz yazılımıdır. 6 temel kalite artırıcı özellik sunar:

### ✨ Temel Özellikler

1. **Blink Detection & Epoch Segmentation** 
   - Video başlamadan önce tüm parlaklık sinyalini analiz eder
   - Z-score tabanlı güçlü istatistiksel göz kırpma tespiti
   - Videoyu temiz "açık göz aralıklarına" (epochs) böler
   
2. **Glare Exclusion**
   - Referans ışık pozisyonları etrafında güvenlik bölgeleri oluşturur
   - False-positive parçacık tespitlerini önler
   
3. **Adaptive Bandpass Filtering**
   - Difference of Gaussians (DoG) ile bandpass filtreleme
   - Lokal mean/std tabanlı adaptif eşikleme
   - Zayıf kontrastlı parçacıkları bile tespit eder
   
4. **FWHM Shape Analysis** ⭐ YENİ! (v2.1.0)
   - Full Width at Half Maximum (FWHM) tabanlı şekil ölçümü
   - Major/minor radius, orientation, elongation hesaplama
   - Motion streak (uzama) tespiti
   - Fiziksel parçacık boyutu ölçümü
   
5. **Automatic Parameter Optimization** ⭐ YENİ! (v2.1.0)
   - ValidationOptimizer sınıfı ile grid search
   - Ground truth ile otomatik F1 score optimizasyonu
   - Precision, Recall, F1 metrikleri
   - En iyi thresh_k ve floor_threshold önerileri
   
6. **Modular Architecture**
   - OOP tasarım prensipleri
   - 8 bağımsız sınıf (SOLID principles)
   - Kolay parametre ayarlama
   - Streamlit UI desteği (titration + optimization)

---

## 📦 Gereksinimler

```bash
pip install opencv-python numpy scipy pandas
```

---

## 🚀 Hızlı Başlangıç

### Temel Kullanım

```python
from tear_film_advanced import TearFilmConfig, TearFilmAnalyzer

# Konfigürasyon oluştur
config = TearFilmConfig(
    video_path="path/to/video.mkv",
    output_csv="results.csv",
    show_visualization=True
)

# Analizi çalıştır
analyzer = TearFilmAnalyzer(config)
analyzer.analyze_video()
```

### Parametre Özelleştirme

```python
config = TearFilmConfig(
    video_path="video.mkv",
    
    # Blink Detection Parameters
    blink_z_threshold=4.0,      # Daha düşük = daha hassas blink tespiti
    blink_pad_frames=1,         # Blink çevresindeki padding (frames)
    min_epoch_length=5,         # Minimum epoch uzunluğu (frames)
    
    # Glare Exclusion Parameters
    glare_buffer_radius=30,     # Referans ışık etrafındaki buffer (piksel)
    ref_light_threshold=200,    # Referans ışık tespiti için eşik
    
    # Particle Detection Parameters (TITRATION)
    thresh_k=3.0,               # Adaptif eşik çarpanı (daha yüksek = daha seçici)
    sigma_small=1.0,            # Bandpass small sigma
    sigma_large=6.0,            # Bandpass large sigma
    local_window_size=41,       # Lokal istatistik pencere boyutu
    floor_threshold=0.5,        # Minimum eşik değeri
    min_particle_area=1,        # Minimum parçacık alanı (piksel)
    max_particle_area=50,       # Maximum parçacık alanı (piksel)
    
    # Tracking Parameters
    max_tracking_distance=0.2,  # Maximum takip mesafesi (normalize birim)
)
```

---

## 🎛️ Parametre Açıklamaları

### Blink Detection (Göz Kırpma Tespiti)

| Parametre | Varsayılan | Açıklama |
|-----------|------------|----------|
| `blink_z_threshold` | 4.0 | Z-score eşiği. **Daha düşük** = daha hassas (daha fazla blink tespit edilir) |
| `blink_pad_frames` | 1 | Blink çevresine eklenen frame sayısı |
| `min_epoch_length` | 5 | Epoch olarak kabul edilecek minimum frame sayısı |

**Öneriler:**
- Video çok hareketliyse `blink_z_threshold` artırın (örn: 5.0)
- Blink'ler kaçırılıyorsa azaltın (örn: 3.0)

---

### Glare Exclusion (Yansıma Maskeleme)

| Parametre | Varsayılan | Açıklama |
|-----------|------------|----------|
| `glare_buffer_radius` | 30 | Referans ışıklar etrafındaki dışlama yarıçapı (piksel) |
| `ref_light_threshold` | 200 | Referans ışık tespiti için parlaklık eşiği |

**Öneriler:**
- Referans ışıklar büyükse `glare_buffer_radius` artırın (örn: 40-50)
- False-positive'ler ışık yakınındaysa buffer'ı artırın

---

### Particle Detection (Parçacık Tespiti) - TİTRATİON

| Parametre | Varsayılan | Açıklama |
|-----------|------------|----------|
| `thresh_k` | 3.0 | **EN ÖNEMLİ PARAMETRE**. Adaptif eşik çarpanı. Yüksek = daha seçici |
| `sigma_small` | 1.0 | Bandpass filtre small sigma (küçük detaylar) |
| `sigma_large` | 6.0 | Bandpass filtre large sigma (büyük yapılar) |
| `local_window_size` | 41 | Lokal istatistik hesaplama penceresi (tek sayı olmalı) |
| `floor_threshold` | 0.5 | Minimum eşik değeri (gürültü filtreleme) |
| `min_particle_area` | 1 | Minimum parçacık alanı (piksel²) |
| `max_particle_area` | 50 | Maximum parçacık alanı (piksel²) |

**Titration Stratejisi:**

1. **Çok fazla false-positive (gürültü) varsa:**
   - `thresh_k` artırın: 3.5 → 4.0 → 5.0
   - `min_particle_area` artırın: 2 → 3
   - `floor_threshold` artırın: 1.0 → 2.0

2. **Gerçek parçacıklar kaçırılıyorsa:**
   - `thresh_k` azaltın: 3.0 → 2.5 → 2.0
   - `min_particle_area` azaltın: 1
   - `floor_threshold` azaltın: 0.2 → 0.1

3. **Büyük parçacıklar ignore ediliyorsa:**
   - `max_particle_area` artırın: 70 → 100

---

### Tracking (Takip)

| Parametre | Varsayılan | Açıklama |
|-----------|------------|----------|
| `max_tracking_distance` | 0.2 | Frame'ler arası maximum hareket mesafesi (normalize) |

**Öneriler:**
- Parçacıklar hızlı hareket ediyorsa artırın (0.3-0.4)
- Yanlış eşleştirmeler varsa azaltın (0.15-0.1)

---

## 📊 Çıktı Formatı

CSV dosyası şu kolonları içerir:

```csv
frame,time_sec,time_since_blink_s,epoch,particle_id,x_norm,y_norm,mms_velocity
1,0.143,0.000,0,1,0.6296,3.8851,0.0585
2,0.286,0.143,0,1,0.6310,3.8890,0.0612
...
```

- `frame`: Frame numarası
- `time_sec`: Mutlak zaman (saniye) - video başlangıcından itibaren
- `time_since_blink_s`: **Göz kırpmadan itibaren geçen süre** - epoch-relative time (power-law analiz için)
- `epoch`: Epoch indeksi (hangi açık göz aralığı)
- `particle_id`: Parçacık ID'si
- `x_norm`, `y_norm`: Normalize koordinatlar (referans ışık mesafesi birimi)
- `mms_velocity`: Momentary Moving Speed (MMS) - 0.1 saniyedeki hareket
- `major_radius`, `minor_radius`, `orientation`, `elongation`: FWHM şekil analizi (opsiyonel)

---

## 🎨 Streamlit UI Kullanımı

Parametre ayarları için interaktif UI:

```bash
streamlit run tear_film_ui.py
```

UI özellikleri:
- Gerçek zamanlı parametre ayarlama
- Tek frame üzerinde titration önizlemesi
- Parçacık tespit kalitesi görselleştirme
- Batch analiz

---

## 🔬 Algoritma Detayları

### 1. Blink Detection Pipeline

```
Video Frame → Grayscale → Mean & Bright Pixel Count
    ↓
Robust Z-Score Hesaplama (Median-based)
    ↓
Z > threshold → Blink Frame olarak işaretle
    ↓
Ardışık blink frame'leri birleştir + padding
    ↓
Blink Range'leri oluştur
```

### 2. Epoch Segmentation

```
Video: [0 ────── BLINK ────── BLINK ────── END]
        ↓
Epochs: [EPOCH1] [SKIP] [EPOCH2] [SKIP] [EPOCH3]
```

### 3. Adaptive Particle Detection

```
Frame → Bandpass Filter (DoG)
    ↓
Local Mean & Std hesapla (41x41 window)
    ↓
Adaptive Threshold: pixel > mean + k*std + floor
    ↓
Glare Mask uygula (referans ışık buffer'ları dışla)
    ↓
Connected Components (area filter)
    ↓
Particle detections
```

### 4. Coordinate Normalization

```
Pixel (x, y) → Normalized (x_n, y_n)

x_n = (x - superior_light_x) / d_cn
y_n = (y - superior_light_y) / d_cn

d_cn = distance(superior_light, inferior_light)
```

Bu normalizasyon sayesinde farklı zoom seviyelerinde çekilmiş videolar karşılaştırılabilir.

---

## 📈 Veri Analizi Örneği

```python
import pandas as pd
import matplotlib.pyplot as plt

# Sonuçları yükle
df = pd.read_csv("tear_film_analysis_advanced.csv")

# Epoch bazlı istatistikler
print(df.groupby('epoch')['mms_velocity'].describe())

# Power-Law Decay Curve Analysis
from tear_film_advanced import compute_power_law_decay

power_law_result = compute_power_law_decay(df, bin_size=0.15)
if power_law_result:
    print(f"α (alpha): {power_law_result['alpha']:.3f}")
    print(f"β (beta): {power_law_result['beta']:.3f}")
    print(f"R²: {power_law_result['r_squared']:.4f}")
    print(f"Equation: {power_law_result['equation']}")
    
    # Plot power-law decay
    import matplotlib.pyplot as plt
    plt.scatter(power_law_result['binned_time'], power_law_result['binned_velocity'], label='Data')
    plt.plot(power_law_result['binned_time'], power_law_result['fitted_curve'], 'r-', label='Fit')
    plt.xlabel('Time Since Blink (s)')
    plt.ylabel('Velocity (mm/s)')
    plt.legend()
    plt.show()
```

---

## 🐛 Sorun Giderme

### Problem: Hiç epoch bulunamıyor
**Çözüm:** `blink_z_threshold` artırın (5.0-6.0) veya `min_epoch_length` azaltın

### Problem: Çok fazla false-positive parçacık
**Çözüm:** `thresh_k` artırın (4.0-5.0) veya `glare_buffer_radius` büyütün

### Problem: Gerçek parçacıklar kaçırılıyor
**Çözüm:** `thresh_k` azaltın (2.0-2.5) veya `sigma_large` artırın (8.0-10.0)

### Problem: Parçacık ID'leri sürekli değişiyor
**Çözüm:** `max_tracking_distance` artırın (0.3-0.4)

---

## 📚 Referanslar

Bu sistem şu kaynaklardan esinlenmiştir:
- PTLib JavaScript Library (tear film analysis)
- OpenCV Documentation
- Scipy Gaussian Filtering

---

## 👨‍💻 Geliştirici Notları

### Sınıf Mimarisi

```
TearFilmConfig (dataclass)
    └─ Tüm parametreleri tutar

BlinkDetector
    └─ compute_robust_z_score()
    └─ detect_blinks()

EpochSegmenter
    └─ segment_epochs()

GlareExcluder
    └─ detect_fixation_lights()
    └─ create_glare_mask()

ParticleDetector
    └─ bandpass_filter()
    └─ local_mean_std()
    └─ detect_particles()

ParticleTracker
    └─ update()
    └─ normalize_coordinates()

TearFilmAnalyzer (Orchestrator)
    └─ analyze_video()
```

### Genişletme Noktaları

1. **Custom Particle Filters:** `ParticleDetector.detect_particles()` metodunu override edin
2. **Alternative Tracking:** `ParticleTracker` sınıfını Kalman Filter ile değiştirin
3. **Multi-threading:** `_process_epochs()` metodunu parallelleştirin

---

## 📄 Lisans

MIT License - Araştırma amaçlı kullanım için açık kaynak

---

**Son Güncelleme:** 2026-07-27
**Versiyon:** 2.0.0
