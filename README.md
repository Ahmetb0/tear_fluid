# Advanced Tear Film Analysis

Python-based computer vision pipeline for **tear film particle tracking**, **blink-aware epoch segmentation**, and **clinical velocity decay analysis**. Inspired by the PTLib JavaScript reference library, refactored into a modular OOP architecture with an optional Streamlit UI.

---

## What This Project Does

Tear film dynamics are often studied by tracking bright particles in high-speed eye videos after a blink. Raw video analysis is noisy: blinks, glare from fixation lights, and weak contrast make simple thresholding unreliable.

This system addresses that by:

1. **Detecting blinks** (robust Z-score on brightness) and splitting the video into clean **open-eye epochs**
2. **Excluding glare** around superior/inferior fixation lights
3. **Detecting particles** with adaptive bandpass filtering (Difference of Gaussians + local statistics)
4. **Tracking particles** across frames and computing **MMS velocity** (momentary moving speed)
5. **Analyzing decay** with a **power-law model** \( v = \alpha \cdot t^{-\beta} \) using time since blink
6. **Validating parameters** via ground-truth annotation and grid search (F1 optimization)

---

## Key Features

| Feature | Description |
|--------|-------------|
| **Blink & epoch segmentation** | Median/MAD Z-score blink detection; padded blink exclusion |
| **Glare exclusion** | Circular buffer masks around fixation lights |
| **Adaptive detection** | DoG bandpass + `thresh_k` / `floor_threshold` local thresholding |
| **FWHM shape analysis** | Major/minor radius, orientation, elongation per particle |
| **Power-law decay** | Pooled post-blink velocity curve with R², α, β |
| **Validation optimizer** | Grid search over detection params vs. manual ground truth |
| **Streamlit UI** | Titration, blink preview, full analysis, results, optimization |
| **Safe frame selection** | Frame picker limited to valid open-eye intervals |

---

## Project Structure

```
tear_fluid/
├── tear_film_advanced.py   # Core OOP pipeline (main entry point)
├── tear_film_ui.py         # Streamlit interactive UI
├── veri_analizi.py         # CLI power-law post-processing
├── requirements.txt
├── run_analysis.bat        # Windows: run analysis
├── start_ui.bat            # Windows: launch Streamlit UI
├── README.md               # Project overview (English)
├── README_ADVANCED.md      # Detailed user guide
├── INSTALL.md              # Installation guide
├── POWER_LAW_ANALYSIS.md   # Power-law decay methodology
└── SAFE_FRAME_GUIDE.md     # Safe frame selection docs
```

---

## Requirements

- Python 3.8+
- Windows / macOS / Linux
- OpenCV-compatible video input (e.g. `.mkv`, `.mp4`)

```bash
pip install -r requirements.txt
```

---

## Quick Start

### 1. Command-line analysis

Edit the video path in `tear_film_advanced.py` (`main()`), or use programmatically:

```python
from tear_film_advanced import TearFilmConfig, TearFilmAnalyzer

config = TearFilmConfig(
    video_path="path/to/your_video.mkv",
    output_csv="results.csv",
    show_visualization=False,
)

analyzer = TearFilmAnalyzer(config)
analyzer.analyze_video()
```

```bash
python tear_film_advanced.py
```

### 2. Streamlit UI (recommended)

```bash
python -m streamlit run tear_film_ui.py
```

Or on Windows:

```bat
start_ui.bat
```

Workflow: **Load video** → tune parameters in **Titration** → **Run Analysis** → view **Results** and power-law curve → optional **Optimization** with clickable ground truth.

### 3. Power-law post-processing

```bash
python veri_analizi.py tear_film_analysis_advanced.csv
```

---

## Output CSV Columns

| Column | Description |
|--------|-------------|
| `frame` | Frame index |
| `time_sec` | Absolute time (seconds) |
| `time_since_blink_s` | Time since epoch start (post-blink relative) |
| `include_in_power_law_fit` | `1` if epoch is valid for decay fitting |
| `epoch` | Open-eye interval index |
| `particle_id` | Track ID |
| `x_norm`, `y_norm` | Normalized coordinates |
| `mms_velocity` | Momentary moving speed |
| `major_radius`, … | FWHM shape metrics (when enabled) |

---

## Architecture (Core Classes)

- `TearFilmConfig` — centralized parameters
- `BlinkDetector` / `EpochSegmenter` — blink signal & open-eye intervals
- `GlareExcluder` — fixation light detection & mask
- `ParticleDetector` — adaptive bandpass particle detection + FWHM
- `ParticleTracker` — greedy nearest-neighbor tracking
- `ValidationOptimizer` — ground-truth grid search
- `TearFilmAnalyzer` — full video orchestration
- `compute_power_law_decay()` — pooled decay curve fitting

---

## Documentation

| Document | Purpose |
|----------|---------|
| [README_ADVANCED.md](README_ADVANCED.md) | Full feature & parameter reference |
| [INSTALL.md](INSTALL.md) | Step-by-step setup |
| [POWER_LAW_ANALYSIS.md](POWER_LAW_ANALYSIS.md) | Decay curve methodology |
| [SAFE_FRAME_GUIDE.md](SAFE_FRAME_GUIDE.md) | Safe frame UI behavior |

---

## Example Configuration

```python
config = TearFilmConfig(
    video_path="video.mkv",
    blink_z_threshold=4.0,
    blink_pad_frames=3,
    min_epoch_length=5,
    glare_buffer_radius=30,
    thresh_k=3.0,
    floor_threshold=0.5,
    min_particle_area=1,
    max_particle_area=50,
    fwhm_enabled=True,
    output_csv="tear_film_analysis.csv",
)
```

---

## License & Research Use

Research and clinical tooling project. Adapt parameters and validate against your acquisition setup before publication or diagnostic use.

---

## Acknowledgments

Algorithm design influenced by the **PTLib** tear film analysis reference (JavaScript). Implemented in Python with OpenCV, NumPy, SciPy, and Streamlit.

**Version:** 2.2.x  
**Author:** Tear Film Research Lab
