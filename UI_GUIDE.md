# Streamlit UI Kullanım Kılavuzu

Bu kılavuz, **Tear Film Analysis** Streamlit arayüzünün nasıl kullanılacağını adım adım açıklar.

## Başlatma

```powershell
cd C:\Users\Asus\Desktop\tear_fluid
python -m streamlit run tear_film_ui.py
```

---

## Genel akış

Tüm sekmeler **sol panelde yüklenen aynı videoyu** kullanır.

```
1. Sol panel → Video yükle → Load Video
2. Blink Detection → (isteğe bağlı) parametreleri ayarla
3. Titration / Optimization → klasik parçacık tespiti
4. Run Analysis → tam analiz + CSV
5. U-Net Gözyaşı Takibi → derin öğrenme segmentasyonu + takip
6. Results → klasik analiz CSV grafikleri
```

---

## 1. Sol panel — Video yükleme (ortak)

| Adım | Ne yapılır |
|------|------------|
| 1 | **Video dosyası** seçin (mp4, avi, mov, mkv) |
| 2 | **Load Video** butonuna basın |
| 3 | Otomatik blink analizi çalışır |

Yükleme sonrası görecekleriniz:

- Toplam kare sayısı
- Blink (göz kırpma) sayısı
- **Epoch** — göz açık aralıklar
- **Safe frame** — analize uygun kareler (blink dışı)

Video `.streamlit_uploads/` klasörüne kaydedilir; **Titration**, **Run Analysis** ve **U-Net** aynı dosyayı okur.

> **Not:** Eski yöntem (metin kutusuna dosya yolu yazma) kaldırıldı. Artık yalnızca dosya yükleme kullanılır.

---

## 2. Blink Detection sekmesi

**Amaç:** Göz kırpmalarını filtrelemek; safe frame listesini oluşturmak.

- Video yüklendiğinde blink analizi **otomatik** yapılır.
- Parlaklık sinyali grafiğinde kırmızı bölgeler = blink.
- Parametreleri değiştirip **Re-analyze with New Parameters** ile yeniden çalıştırabilirsiniz:
  - **Z-Score Threshold** — düşük = daha hassas blink tespiti
  - **Padding Frames** — blink sonrası güvenlik payı
  - **Minimum Epoch Length** — çok kısa aralıkları at

### U-Net ile ilişkisi

**Evet, blink detector U-Net için de çalışır.**

- **U-Net Gözyaşı Takibi** sekmesinde varsayılan olarak **“Sadece safe frame'leri işle”** işaretlidir.
- Bu modda yalnızca blink dışı kareler segmente edilir ve takip edilir — klasik pipeline ile aynı mantık.
- Kutuyu kapatırsanız videonun **tüm kareleri** işlenir (blink dönemleri dahil; genelde önerilmez).

Blink parametrelerini değiştirdikten sonra U-Net sekmesine geçmeden önce **Re-analyze** yapın; safe frame listesi güncellenir.

---

## 3. Titration sekmesi

**Amaç:** Klasik parçacık tespiti parametrelerini ayarlamak.

- Yalnızca **safe frame**'lerden birini seçerek önizleme yapılır.
- `thresh_k`, glare buffer, min/max area vb. ayarlanır.
- **Apply & Visualize** ile anlık sonuç görülür.

---

## 4. Run Analysis sekmesi

**Amaç:** Klasik `tear_film_advanced` pipeline ile tam video analizi.

- Blink / epoch filtrelemesi dahili olarak uygulanır.
- Çıktı: CSV (`epoch`, `mms_velocity`, `time_since_blink_s`, …).

---

## 5. U-Net Gözyaşı Takibi sekmesi

**Amaç:** Eğitilmiş U-Net (`unet_tear_film.pth`) ile segmentasyon + takip.

### Ön koşullar

1. Sol panelden video yüklü olmalı.
2. `unet_tear_film.pth` proje kökünde olmalı (`python train_unet.py`).

### Adımlar

1. Sekmeye gidin — aktif video adı üstte görünür.
2. **Sadece safe frame'leri işle** — blink filtresi (önerilen: açık).
3. FPS, maske eşiği (0.2), takip mesafesi ayarlayın.
4. **U-Net ile İşle ve Takip Et** butonuna basın.

### Çıktılar

| Çıktı | Açıklama |
|-------|----------|
| Takip videosu | Maskeler, ID'ler, yörünge çizgileri |
| Hız grafiği | px/s (isteğe bağlı mm/s) |
| CSV indir | `unet_particle_tracks.csv` |

> U-Net CSV'sini **Results** sekmesine yüklemeyin; orası klasik analiz içindir. U-Net sonuçları bu sekmede kalır.

---

## 6. Results sekmesi

**Amaç:** Klasik **Run Analysis** CSV grafikleri (power-law, epoch karşılaştırma).

- Beklenen sütunlar: `epoch`, `mms_velocity`, `time_since_blink_s`
- U-Net CSV yüklerseniz uygulama formatı algılar ve kısa özet gösterir (power-law çalışmaz).

---

## 7. Optimization sekmesi

**Amaç:** Grid search ile `thresh_k` / `floor_threshold` optimizasyonu.

- Safe frame üzerinde ground truth işaretleyip en iyi parametreleri bulur.
- Sonuçlar klasik analiz config'ine uygulanır.

---

## İki pipeline karşılaştırması

| | Klasik (Run Analysis) | U-Net Takibi |
|--|----------------------|--------------|
| Tespit | Adaptive bandpass + glare | Derin öğrenme maskesi |
| Blink filtresi | Evet (epoch) | Evet (safe frame checkbox) |
| Hız birimi | MMS (normalize) | px/s, opsiyonel mm/s |
| CSV | `epoch`, `mms_velocity` | `velocity_px_per_sec`, `centroid_x/y` |
| Results sekmesi | ✅ | ❌ (kendi sekmesi) |

---

## Sık karşılaşılan sorunlar

| Sorun | Çözüm |
|-------|--------|
| U-Net sekmesi “video yükleyin” diyor | Sol panel → dosya seç → **Load Video** |
| Safe frame %0 | Blink Detection → parametreleri gevşetin → Re-analyze |
| CUDA yok | U-Net CPU'da yavaş çalışır; eğitim için CUDA gerekli |
| Results'ta `epoch` hatası | Yanlış CSV — U-Net sonuçlarını U-Net sekmesinde görün |
| Model bulunamadı | `python train_unet.py` ile `unet_tear_film.pth` oluşturun |

---

## Önerilen iş akışı (U-Net)

1. Sol panelden göz videosunu yükle → **Load Video**
2. **Blink Detection** → safe frame oranını kontrol et (>%50 ideal)
3. Gerekirse blink parametrelerini ayarla → **Re-analyze**
4. **U-Net Gözyaşı Takibi** → safe frame ✅ → **İşle ve Takip Et**
5. Videoyu ve hız grafiğini incele → CSV indir

## Önerilen iş akışı (klasik analiz)

1. Video yükle → Blink kontrol
2. **Titration** → parametre ayarı
3. **Run Analysis** → CSV al
4. **Results** → power-law grafikleri
