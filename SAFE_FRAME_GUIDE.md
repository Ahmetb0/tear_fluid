# Safe Frame Selection Guide (v2.1.0)

## 🎯 Özellik Özeti

**Safe Frame Selection**, Streamlit UI'da kullanıcının sadece geçerli (blink-free) frame'lerden seçim yapabilmesini sağlar. Bu özellik analiz kalitesini ve kullanıcı deneyimini önemli ölçüde artırır.

---

## 🚀 Nasıl Çalışır?

### 1️⃣ Video Yükleme (Otomatik Ön İşlem)

Video yüklendiğinde sistem otomatik olarak:

```
Load Video Button Clicked
    ↓
Load Sample Frame
    ↓
Run BlinkDetector (Z-score analysis)
    ↓
Run EpochSegmenter (identify open-eye intervals)
    ↓
Create Safe Frames List (flatten all epoch frames)
    ↓
Cache Results in st.session_state
    ↓
Show Summary to User
```

**Cached Data:**

- `st.session_state.epochs` - List of Epoch objects
- `st.session_state.safe_frames` - Flattened list of valid frame numbers
- `st.session_state.blink_ranges` - List of (start, end) blink ranges
- `st.session_state.num_frames` - Total frames in video

---

### 2️⃣ Safe Frame Selector (UI Component)

**Before (v2.0.0):**

```python
# Regular slider - could select ANY frame including blinks
frame_idx = st.slider("Frame", 0, 100, 30)
```

**After (v2.1.0):**

```python
# Select slider - ONLY safe frames
selected_frame = st.select_slider(
    "Select Frame",
    options=st.session_state.safe_frames,  # Only valid frames!
    value=default_safe_frame
)
```

---

### 3️⃣ Frame Information Display

Her seçilen frame için kullanıcıya gösterilen bilgi:

```
✅ Safe Frame | Epoch 2/4 | 45% through epoch
```

**Detaylar:**

- `is_safe`: Boolean - frame güvenli mi?
- `epoch_idx`: Hangi epoch'ta (0-indexed)
- `epoch_progress`: Epoch içinde yüzde kaç ilerlenmiş
- `message`: Kullanıcıya gösterilen durum mesajı

---

## 📊 UI Değişiklikleri

### Titration Tab (Tab 1)

**Eski Durum:**

- Frame seçimi yok veya serbest slider

**Yeni Durum:**

```
🎯 Frame Selection (Safe Frames Only)
[======o============] Select Frame for Analysis
✅ Safe Frame | Epoch 2/4 | 45% through epoch
---
Particle Detection Parameters...
```

**Özellikler:**

- ✅ Sadece safe frame'ler seçilebilir
- ✅ Epoch bilgisi gösterilir
- ✅ Seçilen frame otomatik yüklenir
- ❌ Epoch bulunamazsa slider gizlenir + uyarı

---

### Optimization Tab (Tab 5)

**Eski Durum:**

- Frame 30'dan sabit frame kullanılıyordu

**Yeni Durum:**

```
🎯 Frame Selection (Safe Frames Only)
💡 Important: Select a frame with typical particle density
   from a stable open-eye period for best optimization results.

[======o============] Select Frame for Optimization
✅ Safe Frame | Epoch 3/4 | 67% through epoch | Frame # 42
---
1️⃣ Ground Truth Annotation...
```

**Özellikler:**

- ✅ Kullanıcı tipik frame seçebilir
- ✅ Optimization her zaman clean frame'de yapılır
- ✅ Frame numarası metric olarak gösterilir

---

### Blink Detection Tab (Tab 2)

**Eski Durum:**

- Sadece analiz butonu

**Yeni Durum:**

```
✅ Blink detection already performed (cached from video load)

[Total Frames: 79] [Blink Events: 3] [Valid Epochs: 4] [Safe Frames: 78.5%]

💡 These results are used for safe frame selection in Titration and Optimization tabs.

🔄 Re-analyze with New Parameters [Button]
```

**Özellikler:**

- ✅ Cache'teki sonuçlar gösterilir
- ✅ Re-analyze butonu ile yeniden analiz
- ✅ Sonuçlar otomatik güncellenir

---

### Sidebar (Global)

**Yeni Bölüm:**

```
📊 Current Video Status
  Epoch Information [Expandable]
    ├─ Total Frames: 79
    ├─ Blink Events: 3
    ├─ Valid Epochs: 4
    ├─ Safe Frames: 62 (78.5%)
    └─ Epoch Details:
       • Epoch 1: frames 0-8 (8 frames)
       • Epoch 2: frames 14-28 (14 frames)
       • Epoch 3: frames 33-53 (20 frames)
       • Epoch 4: frames 59-79 (20 frames)
```

---

## 🔧 Teknik Detaylar

### Yeni Fonksiyonlar

#### 1. `preprocess_video_epochs()`

```python
def preprocess_video_epochs(video_path: str, config: TearFilmConfig):
    """
    Preprocess video to detect blinks and segment epochs.
    Runs once when video is loaded and caches results.

    Returns:
        tuple: (epochs, safe_frames, blink_ranges, num_frames, elapsed)
    """
```

**Ne Yapar:**

- Video FPS'ini okur
- BlinkDetector çalıştırır
- EpochSegmenter çalıştırır
- Safe frame listesi oluşturur
- Elapsed time hesaplar

**Ne Zaman Çağrılır:**

- Video ilk yüklendiğinde (Load Video button)
- Blink detection parametreleri değiştiğinde (Re-analyze)

---

#### 2. `get_frame_epoch_info()`

```python
def get_frame_epoch_info(frame_idx: int, epochs, safe_frames):
    """
    Get epoch information for a given frame.

    Returns:
        dict: Information about frame safety and epoch
    """
```

**Ne Yapar:**

- Frame'in safe olup olmadığını kontrol eder
- Hangi epoch'ta olduğunu bulur
- Epoch içinde yüzde kaç ilerlediğini hesaplar
- Kullanıcı mesajı oluşturur

**Dönen Dict:**

```python
{
    'is_safe': True/False,
    'epoch_idx': 2,  # 0-indexed
    'epoch_progress': 45.0,  # percentage
    'epoch_length': 20,  # frames
    'message': '✅ Safe Frame | Epoch 3/4 | 45% through epoch'
}
```

---

### Session State Değişkenleri

```python
# Video loading state
st.session_state.video_loaded = True/False
st.session_state.sample_frame = (color_frame, gray_frame)
st.session_state.config = TearFilmConfig(...)

# NEW: Epoch preprocessing cache
st.session_state.epochs = [Epoch(...), Epoch(...), ...]
st.session_state.safe_frames = [0, 1, 2, 14, 15, ..., 78]
st.session_state.blink_ranges = [(8, 13), (28, 32), (53, 58)]
st.session_state.num_frames = 79
```

**Avantajlar:**

- ✅ Epoch detection sadece bir kez çalışır
- ✅ Tab değiştirirken state korunur
- ✅ Gereksiz recomputation önlenir
- ✅ UI responsive kalır

---

## 🎯 Kullanım Senaryoları

### Senaryo 1: Normal Kullanım

```
1. Video yükle → Epochs otomatik tespit edilir
2. Titration tab'a git → Safe frame seç
3. Parametreleri ayarla → Apply & Visualize
4. Optimization tab'a git → Farklı safe frame seç
5. Ground truth ekle → Grid search çalıştır
```

**Sonuç:** ✅ Her adımda clean frame garantisi

---

### Senaryo 2: Epoch Bulunamadı

```
1. Video yükle → 0 epoch tespit edildi
2. Herhangi bir tab'a git → Uyarı mesajı görünür:
   ❌ No valid epochs found!
   [Slider gizli]
   [Öneriler gösteriliyor]
3. Blink Detection tab'a git → Parametreleri ayarla
4. Re-analyze → Yeni epoch'lar bulunur
5. Diğer tab'lara dön → Slider şimdi çalışıyor
```

**Sonuç:** ✅ Kullanıcı ne yapması gerektiğini biliyor

---

### Senaryo 3: Parametre Optimizasyonu

```
1. Video yükle (default: z_threshold=4.0)
   → 4 epoch bulundu
2. Blink Detection tab
   → z_threshold=6.0 yap (daha az hassas)
3. Re-analyze
   → 2 epoch bulundu (daha agresif birleştirme)
4. Cache güncellendi
   → Safe frames artık daha fazla frame içeriyor
```

**Sonuç:** ✅ Gerçek zamanlı epoch ayarı

---

## 🚨 Error Handling

### 1. No Epochs Found

```python
if len(st.session_state.safe_frames) == 0:
    st.error("❌ No valid epochs found!")
    st.warning("""
        No analyzable open-eye intervals detected.

        This could mean:
        - Video has too many blinks
        - Blink detection too sensitive
        - Video too short

        Try adjusting blink_z_threshold or use different video.
    """)
    # Hide slider
```

### 2. Frame Outside Epochs (Shouldn't Happen)

```python
frame_info = get_frame_epoch_info(frame_idx, epochs, safe_frames)
if not frame_info['is_safe']:
    st.error("⚠️ UNSAFE: This frame is in a blink period!")
```

### 3. Video Load Failed

```python
if color_frame is None:
    st.sidebar.error("❌ Failed to load frame")
    # Epochs not computed, safe_frames = []
```

---

## 📈 Performans

### Preprocessing Time

```
Video: 79 frames @ 7 FPS
Blink Detection: ~8-10 seconds
Epoch Segmentation: <1 second
Total: ~10 seconds (acceptable, runs once)
```

### UI Responsiveness

```
Before: Every slider move → potential blink frame → UI crash
After: Slider only on safe frames → never crashes
```

---

## 🎓 Best Practices

### For Users

1. **Always load video first** - epochs computed automatically
2. **Check Blink Detection tab** - verify epochs are reasonable
3. **Adjust z_threshold if needed** - re-analyze updates cache
4. **Use middle of epochs** - most stable frames
5. **Avoid epoch boundaries** - even if technically "safe"

### For Developers

1. **Cache everything** - don't recompute epochs
2. **Validate before use** - always check `len(safe_frames) > 0`
3. **Show epoch info** - users need context
4. **Handle edge cases** - no epochs, single frame epochs, etc.
5. **Keep state consistent** - update all cache vars together

---

## 🔄 Update Cycle

```
Video Load
    ↓
Preprocess (10s)
    ↓
Cache Epochs ✅
    ↓
User Adjusts Params (Blink Detection Tab)
    ↓
Re-analyze (10s)
    ↓
Update Cache ✅
    ↓
UI Automatically Uses New Safe Frames ✅
```

---

## 📊 Comparison

| Feature         | Before (v2.0.0)     | After (v2.1.0)         |
| --------------- | ------------------- | ---------------------- |
| Frame Selection | Any frame (0-N)     | Only safe frames       |
| Blink Detection | On demand           | Automatic on load      |
| Epoch Info      | Hidden              | Visible in UI          |
| Cache           | None                | Full epoch cache       |
| Error Rate      | High (blink frames) | Zero (guaranteed safe) |
| User Experience | Confusing           | Clear and safe         |

---

## 🎯 Future Enhancements

### Potential Additions

1. **Frame Thumbnail Preview** - Show mini preview of selected frame
2. **Epoch Quality Score** - Rate each epoch (stable vs. transitional)
3. **Smart Frame Suggestion** - Auto-suggest best frame for optimization
4. **Batch Processing** - Analyze all safe frames in one go
5. **Export Safe Frame List** - CSV with frame numbers and epoch info

---

**Version:** 2.1.0  
**Feature:** Safe Frame Selection  
**Status:** ✅ Production Ready  
**Date:** 2026-07-27
