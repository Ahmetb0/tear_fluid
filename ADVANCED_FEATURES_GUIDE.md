# Advanced Features Guide (v2.1.0)

## 🌟 Yeni Özellikler

### 1️⃣ FWHM-Based Particle Shape Analysis

#### Nedir?
Full Width at Half Maximum (FWHM) yöntemi ile parçacık şekil analizi. Her parçacık için:
- **Major Radius**: Büyük eksen yarıçapı (piksel)
- **Minor Radius**: Küçük eksen yarıçapı (piksel)
- **Orientation**: Yönelim açısı (radyan)
- **Elongation**: Uzama oranı (major/minor)
- **FWHM Area**: FWHM tabanlı alan
- **Peak Value**: Peak yoğunluk değeri

#### Ne İşe Yarar?
- **Motion streak tespiti**: Elongation > 2.0 olan parçacıklar hareket bulanıklığı gösterir
- **Fiziksel boyut ölçümü**: Binary area yerine yoğunluk profili tabanlı ölçüm
- **Şekil karakterizasyonu**: Dairesel vs. uzamış parçacık ayrımı

#### Nasıl Kullanılır?

```python
from tear_film_advanced import TearFilmConfig, TearFilmAnalyzer

# FWHM aktif (varsayılan)
config = TearFilmConfig(
    video_path="video.mkv",
    fwhm_enabled=True,  # FWHM shape analysis
    fwhm_search_radius=4,  # Peak arama yarıçapı
    fwhm_rel_threshold=0.5,  # Half-maximum threshold
    fwhm_max_radius=12  # Maksimum parçacık yarıçapı
)

analyzer = TearFilmAnalyzer(config)
analyzer.analyze_video()
```

#### CSV Çıktısı

FWHM aktifken CSV'de ek kolonlar:
```csv
frame,time_sec,epoch,particle_id,x_norm,y_norm,mms_velocity,major_radius,minor_radius,orientation,elongation,fwhm_area,peak_value
1,0.143,0,1,0.6296,3.8851,0.0585,8.8459,2.9039,-0.0366,3.0462,77.0000,97.9795
```

#### Veri Analizi Örneği

```python
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("tear_film_analysis.csv")

# Motion streak analizi
streaks = df[df['elongation'] > 2.0]
print(f"Motion streaks: {len(streaks)} / {len(df)} ({len(streaks)/len(df)*100:.1f}%)")

# Elongation dağılımı
plt.hist(df['elongation'], bins=50)
plt.xlabel('Elongation Ratio')
plt.ylabel('Count')
plt.title('Particle Elongation Distribution')
plt.show()

# Boyut vs Hız korelasyonu
plt.scatter(df['major_radius'], df['mms_velocity'], alpha=0.3)
plt.xlabel('Major Radius (pixels)')
plt.ylabel('MMS Velocity')
plt.title('Particle Size vs Velocity')
plt.show()
```

---

### 2️⃣ Automatic Parameter Optimization

#### Nedir?
Ground truth ile karşılaştırma yaparak otomatik parametre optimizasyonu. Grid search ile en iyi `thresh_k` ve `floor_threshold` değerlerini bulur.

#### Ne İşe Yarar?
- **Subjektif ayarlamayı ortadan kaldırır**: Manuel deneme-yanılma yerine istatistiksel optimizasyon
- **F1 Score maksimizasyonu**: Precision ve Recall dengesini optimize eder
- **Yeni veri setlerinde hızlı kalibrasyon**: Her yeni video tipi için en iyi parametreler

#### Nasıl Kullanılır?

##### Programatik Kullanım

```python
import cv2
from tear_film_advanced import (
    TearFilmConfig, ValidationOptimizer, 
    ParticleDetector, GlareExcluder
)

# 1. Frame yükle
cap = cv2.VideoCapture("video.mkv")
cap.set(cv2.CAP_PROP_POS_FRAMES, 30)
_, frame = cap.read()
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
cap.release()

# 2. Config oluştur
config = TearFilmConfig()

# 3. Glare mask hazırla
glare_excluder = GlareExcluder(config)
glare_excluder.detect_fixation_lights(gray)
glare_mask = glare_excluder.create_glare_mask(gray.shape)

# 4. Ground truth tanımla (manuel işaretleme)
ground_truth = [
    (120, 150),  # x, y pixel coordinates
    (180, 200),
    (250, 180),
    # ... daha fazla nokta
]

# 5. Optimizer oluştur ve çalıştır
optimizer = ValidationOptimizer(config)
optimizer.set_ground_truth(ground_truth)

# Grid search
best = optimizer.suggest_settings(
    gray,
    glare_mask,
    thresh_k_range=(1.5, 8.0, 0.5),
    floor_range=(0.0, 2.0, 0.25)
)

print(f"Best F1: {best['f1']:.4f}")
print(f"Best thresh_k: {best['thresh_k']:.2f}")
print(f"Best floor: {best['floor_threshold']:.2f}")

# 6. Optimize edilmiş config kullan
optimized_config = optimizer.apply_best_settings()
```

##### Streamlit UI ile

```bash
streamlit run tear_film_ui.py
```

1. **Video yükle** (sidebar)
2. **Optimization** sekmesine git
3. **Ground truth işaretle**:
   - Manuel olarak (x, y) koordinatları gir
   - VEYA CSV dosyası yükle (x, y kolonları)
4. **Grid search parametrelerini ayarla**
5. **Start Optimization** tıkla
6. **Sonuçları incele ve uygula**

#### Ground Truth Nasıl Oluşturulur?

**Yöntem 1: ImageJ/Fiji ile**
```
1. Video'dan bir frame'i dışa aktar
2. ImageJ'de aç
3. Point Tool ile gerçek parçacıkları işaretle
4. Analyze > Measure > X, Y koordinatlarını kaydet
5. CSV olarak export et
```

**Yöntem 2: Custom Python Script**
```python
import cv2
import csv

points = []

def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        print(f"Point added: ({x}, {y})")

# Frame görüntüle ve işaretle
cv2.namedWindow('Annotate')
cv2.setMouseCallback('Annotate', mouse_callback)
cv2.imshow('Annotate', frame)
cv2.waitKey(0)

# CSV'ye kaydet
with open('ground_truth.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['x', 'y'])
    writer.writerows(points)
```

#### Metrikler

**Precision** = TP / (TP + FP)
- Tespit edilenlerin ne kadarı gerçek?
- Yüksek precision = az false-positive

**Recall** = TP / (TP + FN)
- Gerçeklerin ne kadarı tespit edildi?
- Yüksek recall = az false-negative

**F1 Score** = 2 × (Precision × Recall) / (Precision + Recall)
- Precision ve Recall dengesini optimize eder
- En iyi genel performans metriği

#### Örnek Sonuçlar

```
============================================================
OPTIMIZATION COMPLETE
============================================================
Best F1 Score: 0.8750
Best Parameters:
  thresh_k: 3.50
  floor_threshold: 0.75

Metrics:
  Precision: 0.9333
  Recall: 0.8235
  F1 Score: 0.8750
  True Positives: 14
  False Positives: 1
  False Negatives: 3
  Detected: 15
============================================================
```

**Yorum:**
- Precision 0.93 → %93 tespit doğru
- Recall 0.82 → %82 gerçek parçacık yakalandı
- F1 0.87 → Mükemmel denge

---

## 🎯 Kullanım Senaryoları

### Senaryo 1: Motion Streak Analizi

**Amaç:** Hareket bulanıklığı gösteren parçacıkları tespit et

```python
df = pd.read_csv("results.csv")

# Elongated particles
streaks = df[df['elongation'] > 2.5]

# Ortalama yönelim
mean_orientation = streaks['orientation'].mean()

print(f"Streak particles: {len(streaks)}")
print(f"Mean orientation: {np.degrees(mean_orientation):.1f}°")
```

**Ne zaman kullanılır?**
- Hızlı göz hareketleri
- Düşük frame rate videolar
- Hareket yönü analizi

---

### Senaryo 2: İki Video Karşılaştırması

**Amaç:** Farklı koşullar altında parçacık şekil farklılıklarını karşılaştır

```python
# Video 1 (normal)
df1 = pd.read_csv("normal.csv")

# Video 2 (tedavi sonrası)
df2 = pd.read_csv("post_treatment.csv")

# Karşılaştır
print(f"Mean major radius:")
print(f"  Normal: {df1['major_radius'].mean():.2f}")
print(f"  Post-treatment: {df2['major_radius'].mean():.2f}")

print(f"Mean elongation:")
print(f"  Normal: {df1['elongation'].mean():.2f}")
print(f"  Post-treatment: {df2['elongation'].mean():.2f}")

# Statistical test
from scipy.stats import ttest_ind
t_stat, p_value = ttest_ind(df1['major_radius'], df2['major_radius'])
print(f"T-test p-value: {p_value:.4f}")
```

---

### Senaryo 3: Yeni Video Tipi için Kalibrasyon

**Amaç:** Farklı mikroskop/kamera için optimize parametre bul

```python
# 1. Yeni video tipinden sample frame al
# 2. 10-20 parçacık manuel işaretle
# 3. Grid search çalıştır

optimizer = ValidationOptimizer(config)
optimizer.set_ground_truth(manual_annotations)

best = optimizer.suggest_settings(
    frame, mask,
    thresh_k_range=(1.0, 10.0, 0.5),  # Geniş arama
    floor_range=(0.0, 3.0, 0.25)
)

# 4. En iyi parametreleri kaydet ve kullan
optimized_config = optimizer.apply_best_settings()
```

---

## 📊 CSV Çıktısı Format

### Temel Kolonlar (v2.0.0)
```
frame, time_sec, epoch, particle_id, x_norm, y_norm, mms_velocity
```

### FWHM Kolonları (v2.1.0 +)
```
major_radius, minor_radius, orientation, elongation, fwhm_area, peak_value
```

### Tam Örnek
```csv
frame,time_sec,epoch,particle_id,x_norm,y_norm,mms_velocity,major_radius,minor_radius,orientation,elongation,fwhm_area,peak_value
1,0.143,0,1,0.6296,3.8851,0.0585,8.8459,2.9039,-0.0366,3.0462,77.00,97.98
1,0.143,0,2,-0.0710,3.8628,0.0640,4.5488,2.9333,-1.5708,1.5500,52.00,85.34
```

---

## 🔧 Performans İpuçları

### FWHM Hesaplaması
- **Hızlandırma:** `fwhm_enabled=False` yaparak FWHM'i devre dışı bırak
- **Trade-off:** %10-15 daha hızlı ama şekil bilgisi yok

### Grid Search Optimizasyonu
- **Hızlı test:** Küçük aralıklar kullan (2-5 vs 10, 0.5 vs 0.2 step)
- **Production:** Geniş aralıklar ve küçük step (daha uzun ama daha doğru)
- **Tipik süre:** 20-100 kombinasyon = 30-120 saniye

---

## 🎓 Best Practices

### FWHM Kullanımı
1. **Her zaman aktif tut** - ek bilgi zarar vermez
2. **Elongation > 2.5** → motion streak olarak filtrele
3. **Major radius** → fiziksel parçacık boyutu için daha doğru

### Optimization Kullanımı
1. **Representative frame seç** - tipik parçacık yoğunluğu
2. **10-20 ground truth yeterli** - daha fazla ekleme çok fayda sağlamaz
3. **Match tolerance = 5 pixel** - iyi başlangıç noktası
4. **F1 > 0.80 hedefle** - klinik kalite için yeterli

---

**Version:** 2.1.0  
**Date:** 2026-07-27  
**Status:** ✅ Production Ready
