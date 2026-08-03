# Kurulum Rehberi - Advanced Tear Film Analysis

## 🔧 Sistem Gereksinimleri

- **Python:** 3.8 veya üzeri
- **İşletim Sistemi:** Windows, macOS, Linux
- **RAM:** Minimum 4GB (8GB önerilir)
- **Depolama:** Video dosyası boyutuna bağlı (~100MB boş alan)

---

## 📦 Adım 1: Python Paketlerini Kur

### Yöntem 1: requirements.txt ile (Önerilen)

```bash
cd C:\Users\Asus\Desktop\tear_fluid
pip install -r requirements.txt
```

### Yöntem 2: Manuel Kurulum

```bash
# Core paketler (zorunlu)
pip install opencv-python numpy scipy pandas matplotlib

# Streamlit UI için (opsiyonel)
pip install streamlit

# Gelişmiş görselleştirme için (opsiyonel)
pip install seaborn plotly
```

---

## ✅ Adım 2: Kurulum Testi

Kurulumun başarılı olduğunu test edin:

```bash
python -c "import cv2, numpy, scipy, pandas, matplotlib; print('✅ All core packages installed!')"
```

Streamlit için:
```bash
python -c "import streamlit; print('✅ Streamlit installed!')"
```

---

## 🚀 Adım 3: İlk Analizi Çalıştır

### 3.1. Basit Kullanım (Varsayılan Parametreler)

```bash
python tear_film_advanced.py
```

**Beklenen Çıktı:**
```
============================================================
ADVANCED TEAR FILM ANALYSIS
============================================================

[1/4] Detecting blinks...
Computing brightness signal for blink detection...
Total frames: 79
Detected 3 blink events

[2/4] Segmenting epochs...
Segmented into 4 valid epochs
  Total analyzable frames: 62/79 (78.5%)

[3/4] Processing 4 epochs...
  Epoch 1/4: frames 0-8
  ...

[4/4] Saving results to tear_film_analysis_advanced.csv...

============================================================
ANALYSIS COMPLETE
Total particles tracked: 1513
Output file: tear_film_analysis_advanced.csv
============================================================
```

### 3.2. Özelleştirilmiş Parametrelerle

`tear_film_advanced.py` dosyasındaki `main()` fonksiyonunu düzenleyin:

```python
def main():
    config = TearFilmConfig(
        video_path="KENDI_VIDEONUZ.mkv",  # Video yolunu değiştirin
        
        # Parametreleri ayarlayın
        thresh_k=3.5,  # Daha seçici
        glare_buffer_radius=40,  # Daha büyük buffer
        
        output_csv="sonuclar.csv",
        show_visualization=True  # Görselleştirme açık
    )
    
    analyzer = TearFilmAnalyzer(config)
    analyzer.analyze_video()
```

---

## 🎛️ Adım 4: Streamlit UI'ı Başlat (Opsiyonel)

Parametre ayarları için interaktif arayüz:

```bash
streamlit run tear_film_ui.py
```

Tarayıcınızda otomatik olarak açılacak (genelde `http://localhost:8501`)

**UI Özellikleri:**
- 🎯 Real-time parameter titration
- 🔍 Blink detection preview
- 🚀 One-click full analysis
- 📊 Results visualization

---

## 🛠️ Yaygın Sorunlar ve Çözümleri

### Sorun 1: "ModuleNotFoundError: No module named 'cv2'"

**Çözüm:**
```bash
pip install opencv-python
```

### Sorun 2: "ImportError: DLL load failed"

**Windows'ta:**
```bash
pip install opencv-python --upgrade
```

**Linux'ta:**
```bash
sudo apt-get install python3-opencv
```

### Sorun 3: Streamlit başlatılamıyor

**Çözüm:**
```bash
pip install streamlit --upgrade
streamlit hello  # Test komutu
```

### Sorun 4: Video açılamıyor - "Could not open video"

**Codec sorunu olabilir. Şunu deneyin:**
```bash
pip install opencv-contrib-python
```

veya videoyu H.264 codec ile yeniden encode edin:
```bash
ffmpeg -i input.mkv -c:v libx264 output.mp4
```

### Sorun 5: Hafıza hatası (MemoryError)

Video çok büyük olabilir. Çözümler:
1. Video resolution'ı azaltın
2. Video'yu kısaltın (test için ilk 30 saniye)
3. `show_visualization=False` yapın (RAM tasarrufu)

---

## 📂 Dosya Yapısı

Kurulum sonrası dizin yapınız şöyle olmalı:

```
tear_fluid/
│
├── tear_film_advanced.py      # Ana analiz sistemi
├── tear_film_ui.py            # Streamlit UI
├── veri_analizi.py            # Post-processing
│
├── requirements.txt           # Paket gereksinimleri
├── README.md                  # Proje özeti (English)
├── README_ADVANCED.md         # Kullanım kılavuzu
├── INSTALL.md                 # Bu dosya
├── POWER_LAW_ANALYSIS.md      # Power-law metodolojisi
├── SAFE_FRAME_GUIDE.md        # Safe frame rehberi
│
├── assests/                   # Video dosyaları (gitignore)
│   └── *.mkv
│
└── *.csv                      # Analiz sonuçları (gitignore)
```

---

## 🔍 Hızlı Başlangıç Test Senaryosu

Sistemi test etmek için:

```bash
# 1. Kütüphaneleri kontrol et
python -c "import cv2, numpy, scipy, pandas; print('OK')"

# 2. Test analizi çalıştır (varsayılan parametreler)
python tear_film_advanced.py

# 3. Sonuç CSV'sini kontrol et
python -c "import pandas as pd; df=pd.read_csv('tear_film_analysis_advanced.csv'); print(f'Records: {len(df)}')"

# 4. (Opsiyonel) UI'ı başlat
streamlit run tear_film_ui.py
```

Hepsi çalışıyorsa: **✅ Kurulum başarılı!**

---

## 📚 Sonraki Adımlar

1. **Parametreleri Öğren:** `README_ADVANCED.md` dosyasını okuyun
2. **Kurulum:** `INSTALL.md` ve `README.md` dosyalarına bakın
3. **Titration:** Streamlit UI ile parametreleri optimize edin
4. **Batch Processing:** Birden fazla video için script yazın

---

## 🔥 İleri Seviye Kullanım

### 1. Power-Law Decay Analizi (YENİ!)

Gözyaşı filmi hızının zamanla yavaşlamasını modelleyin (v = α × t^(-β)):

```bash
# Otomatik analiz ve görselleştirme
python veri_analizi.py tear_film_analysis_advanced.csv
```

**Programatik Kullanım:**
```python
import pandas as pd
from tear_film_advanced import compute_power_law_decay

df = pd.read_csv("tear_film_analysis_advanced.csv")
result = compute_power_law_decay(df, bin_size=0.15)

if result:
    print(f"Alpha (α): {result['alpha']:.3f}")
    print(f"Beta (β): {result['beta']:.3f}")
    print(f"R² Score: {result['r_squared']:.4f}")
```

📖 Detaylı rehber: `POWER_LAW_ANALYSIS.md`

### 2. Ground Truth Validation ve Optimizasyon

Streamlit UI'da **Optimization** sekmesi:
- Görsele tıklayarak parçacıkları işaretleyin (streamlit-image-coordinates ile)
- Grid search ile en iyi `thresh_k` ve `floor_threshold` bulun

```python
from tear_film_advanced import ValidationOptimizer

optimizer = ValidationOptimizer(config)
optimizer.set_ground_truth([(100, 200), (150, 250)])
best = optimizer.suggest_settings(frame, glare_mask, 
                                  thresh_k_range=(2.0, 6.0),
                                  floor_range=(0.0, 1.5))
```

### 3. FWHM Şekil Analizi

Parçacık şekillerini (major/minor radius, elongation) ölçün:

```python
config = TearFilmConfig(
    fwhm_enabled=True,
    fwhm_search_radius=4,
    fwhm_rel_threshold=0.5
)
```

CSV çıktısında `major_radius`, `minor_radius`, `orientation`, `elongation` kolonları.

### 4. Batch Processing

Birden fazla video için loop:

```python
import glob
from tear_film_advanced import TearFilmConfig, TearFilmAnalyzer

for video in glob.glob("videos/*.mkv"):
    config = TearFilmConfig(video_path=video, 
                           output_csv=f"{video.stem}_results.csv")
    TearFilmAnalyzer(config).analyze_video()
```

---

## 🆘 Yardım Kaynakları

### Documentation
- `README.md` - Project overview (English)
- `README_ADVANCED.md` - Detaylı kullanım kılavuzu
- `POWER_LAW_ANALYSIS.md` - Power-law decay curve metodolojisi
- `SAFE_FRAME_GUIDE.md` - Safe frame selection rehberi
- Kod içi docstring'ler

### Hata Ayıklama
```python
# Debug mode ile çalıştır
import logging
logging.basicConfig(level=logging.DEBUG)

# Tek bir epoch'u test et
config.min_epoch_length = 999  # Sadece en uzun epoch
```

### Topluluk
- GitHub Issues
- Python OpenCV Documentation: https://docs.opencv.org/
- Scipy Documentation: https://docs.scipy.org/

---

## 🔄 Güncelleme

Sistemi güncellemek için:

```bash
git pull origin main  # Eğer git kullanıyorsanız
pip install -r requirements.txt --upgrade
```

---

**Kurulum sorunları için:** Lütfen terminal çıktısını ve Python versiyonunuzu (`python --version`) paylaşın.

**İyi analizler!** 💧
