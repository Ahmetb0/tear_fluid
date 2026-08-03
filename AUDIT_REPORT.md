# Deep Audit Report — Tear Film Analysis Pipeline

**Date:** 2026-07-28  
**Scope:** `tear_film_advanced.py`, `tear_film_ui.py`  
**Mode:** Read-only analysis (no code changes)  
**Context:** Intermittent Power-Law quality (high R² on some videos, negative R² / flat-chaotic curves on others); suspected UI↔backend state and calculation disconnects.

---

## Executive Summary

Audit confirms **real architectural and mathematical weaknesses**, not only “bad videos.” The strongest findings:

1. **UI parameter flow is fragile** — Titration sliders reset to hardcoded defaults on every Streamlit rerun; optimized params can be silently overwritten; slider values are *not* written to config until “Apply.”
2. **`detect_fixation_lights()` has a critical side-effect bug** — geometry validation failure still *assigns* light coordinates and returns `False`, after which `_process_epochs` treats lights as “already detected” and proceeds with invalid references.
3. **Negative R² is mathematically expected** when pooled binned velocities lack a power-law shape; the R² formula itself is correct, but the **fitting pipeline + data provenance** often produces non-decay signals.
4. **Tracking is greedy nearest-neighbor** — ID swaps and re-births inject velocity spikes / gaps that contaminate early `time_since_blink_s` bins.
5. **`time_since_blink_s` is epoch-relative, not always blink-relative** — especially for epoch 0 starting at frame 0, and when early frames are skipped until fixation succeeds.

---

## 1. Streamlit State and Parameter Flow (UI vs Backend)

### How it is supposed to work

| Stage | Expected behavior |
|-------|-------------------|
| Titration | Sliders → update `st.session_state.config` → preview via `ParticleDetector` |
| Optimization | Grid search → Apply → update `config.thresh_k` / `floor_threshold` |
| Run Analysis | `TearFilmAnalyzer(st.session_state.config)` uses those values |

### What the code actually does

**Run Analysis does use `session_state.config` (when started):**

```577:586:tear_film_ui.py
            if st.button("▶️ Start Analysis", type="primary"):
                st.session_state.config.output_csv = output_csv
                st.session_state.config.show_visualization = show_viz
                ...
                    analyzer = TearFilmAnalyzer(st.session_state.config)
```

`TearFilmAnalyzer.__init__` wires the **same config object** into `ParticleDetector`, `GlareExcluder`, `ParticleTracker`, etc. Detection in `_process_epochs` reads `self.particle_detector` → `self.config.thresh_k` / `floor_threshold`. So once values are correctly stored on the config object, the backend **does** use them.

### Tespit edilen bug’lar / kopukluklar

#### BUG-S1 — Titration sliders ignore `session_state.config` (hardcoded defaults)

```305:342:tear_film_ui.py
            thresh_k = st.slider(..., value=3.0, ...)
            glare_buffer = st.slider(..., value=30, ...)
            floor_thresh = st.slider(..., value=0.5, ...)
```

Every Streamlit rerun re-instantiates sliders with **literal defaults**, not `st.session_state.config.thresh_k`. Effects:

- After Optimization “Apply” sets `config.thresh_k = 5.2`, Titration still **displays** 3.0.
- If user clicks “Apply & Visualize” without noticing, **optimized values are overwritten** back to slider defaults.
- User belief (“I optimized / I set params”) ≠ values used in analysis unless they carefully re-apply.

#### BUG-S2 — Slider motion does not update config until “Apply & Visualize”

Moving sliders alone does **not** mutate `session_state.config`. Run Analysis uses last **applied** config (or dataclass defaults if never applied). This is a classic Streamlit UX/state trap and matches “arka planda bir şeyler kopuk” reports.

#### BUG-S3 — Optimization Apply is partial

```1101:1103:tear_film_ui.py
                        st.session_state.config.thresh_k = best_result['thresh_k']
                        st.session_state.config.floor_threshold = best_result['floor_threshold']
```

Only `thresh_k` and `floor_threshold` are applied. `glare_buffer`, `sigma_*`, `min/max area`, blink params are untouched. `ValidationOptimizer.apply_best_settings()` exists in backend but UI does **not** call it.

#### BUG-S4 — Optimization grid uses a **copy** of config (good), but glare fallback diverges from full analysis

Optimizer builds `TearFilmConfig(**st.session_state.config.__dict__)` and temporarily overrides thresh/floor — correct for grid search.

UI fallback when fixation fails:

```python
glare_mask = np.ones(gray_frame.shape, dtype=bool)  # full frame
```

Full `TearFilmAnalyzer` path **never** uses this fallback; it either skips frames or (see BUG-F1) proceeds with poisoned lights. So optimized F1 on “full frame” can disagree with production analysis that uses glare exclusion (or broken lights).

#### BUG-S5 — `visualize_detection` can draw / crash when lights are invalid

On failed detection with `superior_light is None`, code still calls `cv2.circle(..., glare_excluder.superior_light, ...)`. On geometry-fail path, lights are set incorrectly (BUG-F1) while UI shows “using full frame” — preview mask and drawn circles disagree.

### Mimari zafiyet

Config is a mutable shared dataclass in session state, but **widget values are not bound to it**. There is no single “source of truth” sync layer (e.g. sliders keyed to config, or “dirty” flag).

### Çözüm önerileri

1. Bind all sliders: `value=st.session_state.config.thresh_k` (and unique keys).
2. On every slider change (or before Run Analysis), sync widgets → config automatically.
3. After Optimization Apply, either call `apply_best_settings()` or refresh Titration widgets from config.
4. Show a persistent “Active config” panel that mirrors **exactly** what Run Analysis will use.
5. Align glare-failure policy between Optimization UI and `TearFilmAnalyzer`.

---

## 2. Curve Fitting and R² Mathematics

### Implementation under review

`compute_power_law_decay()` in `tear_film_advanced.py` (~L1249–1362):

- Model: \(v = \alpha \cdot t^{-\beta}\)
- Data: bins on `time_since_blink_s`, **median** velocity per bin, `count ≥ 3`, ≥ 4 bins
- Filter: `time > 0` and `velocity > 0`
- `curve_fit` with `p0=[median_velocities[0], 0.5]`, `bounds=([0.01,0.01],[100,3.0])`
- \(R^2 = 1 - SS_{res}/SS_{tot}\) on **binned medians** (not raw points)

### R² formülü doğru mu?

**Evet.** Numpy implementasyonu klasik determination coefficient. Negatif R², modelin yatay ortalamadan **daha kötü** olduğunu gösterir; bu bir kod hatası değil, **model-data uyumsuzluğu** sinyalidir.

### Tespit edilen matematiksel / pipeline zafiyetleri

#### BUG-P1 — Fit quality is dominated by upstream velocity quality

If tracking produces flat noise or spikes, medians become roughly constant or non-monotonic → power-law cannot beat the mean → **R² < 0**. Video-to-video variance is expected under current pipeline.

#### BUG-P2 — Unstable / suboptimal initialization and parameterization

- `p0` uses **first bin’s median** as α. If that bin is an outlier (blink bleed, ID swap), optimizer starts far from a sensible basin.
- Fitting in **linear space** for a multiplicative power-law is less stable than log-space:
  - Prefer fit \(\ln v = \ln\alpha - \beta \ln t\) with OLS, then refine with `curve_fit`, or use `p0` from log regression.
- `alpha` upper bound `100` may clip legitimate normalized velocities on some geometries (norm distance small → large normalized speeds).

#### BUG-P3 — Pooling all epochs into one curve

All epochs share one `time_since_blink_s` axis. If:

- Epoch 0 is “open eye from video start” (not post-blink), or
- Some epochs are short/noisy,

then medians mix incompatible dynamics → flat/chaotic binned series → negative R².

#### BUG-P4 — Early post-blink dynamics often missing

`time_col > 0` drops t=0. First frame after blink often has **no MMS** (tracker needs 2 frames). Combined with `blink_pad` and fixation skip, the steep part of the decay can be under-sampled; remaining curve looks flat → R² collapses.

#### BUG-P5 — Fit evaluated only at discrete bin centers

`fitted_curve` is `power_law(bin_centers, ...)`. UI plots that as a “smooth” red line through few points. Visually can look wrong even when parameters are OK; not the main cause of negative R², but weakens interpretation.

#### BUG-P6 — No goodness-of-fit gates before reporting

Negative R² / huge β / tiny α are still shown as “the” clinical curve. No automatic reject/fallback (e.g. “insufficient decay structure”).

### Çözüm önerileri

1. Log-space pre-fit for `p0`; widen or data-driven α bounds.
2. Fit **per epoch**, then report median β / pooled only if epoch R² > threshold.
3. Exclude first N ms after epoch start; winsorize velocity outliers (e.g. 99th percentile).
4. Report R² on raw points **and** bins; flag R² < 0 as “model rejected.”
5. Optional: require minimum time span (e.g. ≥ 1.5 s) and minimum bins before fitting.

---

## 3. Epoch and `time_since_blink_s` Calculation

### Epoch construction

```248:259:tear_film_advanced.py
        cuts = [0]
        for start, end in blink_ranges:
            cuts.extend([start, end + 1])
        cuts.append(num_frames)
        ...
            if end - start >= self.config.min_epoch_length:
                epochs.append(Epoch(start, end))
```

Processing uses `range(start_frame, end_frame)` → blink start frame excluded; post-blink restart at `end+1`. **Off-by-one logic is internally consistent.**

Padding:

```194:195:tear_film_advanced.py
                start = max(0, i - self.config.blink_pad_frames)
                end = min(n - 1, j - 1 + self.config.blink_pad_frames)
```

Default `blink_pad_frames=1` is **thin**. Partial lid motion / tear surge frames adjacent to blink often **do not** exceed Z-threshold and remain inside epochs.

### `time_since_blink_s` calculation

```1188:1202:tear_film_advanced.py
        epoch_start_times = {}
        for result in self.results:
            ...
                epoch_start_times[epoch_id] = min(..., time_sec)
        ...
            result['time_since_blink_s'] = result['time_sec'] - epoch_start_times[epoch_id]
```

This is **time since first recorded result in that epoch**, not:

- `epoch.start_frame / fps`, nor
- true physiological “time since blink end.”

### Tespit edilen bug’lar

#### BUG-E1 — Epoch 0 from frame 0 is mislabeled as post-blink

If video opens with an open eye and no prior blink, epoch `[0, first_blink)` still gets `time_since_blink_s` from its first sample. Pooling this with true post-blink epochs **corrupts** the decay curve.

#### BUG-E2 — Fixation skip shifts “t=0”

```1055:1057:tear_film_advanced.py
                if self.glare_excluder.superior_light is None:
                    if not self.glare_excluder.detect_fixation_lights(gray):
                        continue
```

Skipped frames never enter `results`. `time_since_blink_s≈0` attaches to first **successful** analysis frame, which may be hundreds of ms after true blink end — early high-velocity phase lost or misaligned across epochs.

#### BUG-E3 — Insufficient blink padding → velocity explosions at epoch edges

Contaminated near-blink frames produce huge displacements / false detections → extreme `mms_velocity` at small `time_since_blink_s` → early bins explode → fit becomes wild or R² tanks depending on video.

#### BUG-E4 — `min_epoch_length=5` frames is very short

At 30 fps ≈ 0.17 s. Power-law / binning needs longer open intervals; short epochs add noise bins when pooled.

### Çözüm önerileri

1. Define `time_since_blink_s = (frame_idx - epoch.start_frame) / fps` (frame-based), not min result time.
2. Drop or separately flag epoch 0 if it does not follow a blink (`start_frame == 0`).
3. Increase default `blink_pad_frames` (e.g. 3–5) and/or trim first 100–200 ms of each epoch for velocity export.
4. Require `min_epoch_length` in **seconds** (e.g. ≥ 1.0 s) for inclusion in power-law pooling.

---

## 4. Coordinate Matching (Validation Tolerance)

### Matching algorithm

```750:793:tear_film_advanced.py
        for i, (gt_x, gt_y) in enumerate(self.ground_truth_points):
            ...
                dx = particle['x'] - gt_x
                dy = particle['y'] - gt_y
                dist = np.sqrt(dx*dx + dy*dy)
                if dist < best_dist and dist <= tolerance:
```

- Uses detection **contour centroids** (`float`), GT `(x,y)` from clicks.
- Greedy one-to-one (GT → nearest free detection).
- Tolerance from `config.validation_match_tolerance` (default 5 px); UI slider overrides on optimizer config copy.
- **FWHM radii / orientation are not used** in matching.

### Tespit edilen sorunlar

#### BUG-V1 — No type bug, but scale/display risk remains upstream

Numpy float vs click int is fine. Historical UI issues (`streamlit-image-coordinates` + scaling/rerun) can store GT points that are **systematically shifted** relative to algorithm centroids even when types match. That is a **data integrity** risk, not an OpenCV dtype mismatch in `match_detections`.

#### BUG-V2 — Greedy matching ≠ optimal assignment

Classic conflict: two GT points prefer the same detection; greedy order can inflate FP/FN vs Hungarian matching. Affects which `(thresh_k, floor)` wins grid search.

#### BUG-V3 — Tolerance is absolute pixels, independent of particle size

5 px may be tight for large FWHM particles or loose for small ones. Optimizing detection thresholds against a fixed px tolerance couples F1 to resolution/optics differently across videos → “works on video A, fails on B.”

#### BUG-V4 — Optimization vs production glare path mismatch

(Already BUG-S4.) Grid search F1 may be computed on full-frame mask while analysis uses glare circles — parameters that “win” validation may not transfer.

#### BUG-V5 — Heatmap best-point indexing fragility

```1143:1144:tear_film_ui.py
                        best_idx_y = np.where(pivot.index == best_result['thresh_k'])[0][0]
```

Float identity after `np.arange` / pivot can fail (`IndexError`) or mark wrong cell — secondary UI bug, not core matching.

### Çözüm önerileri

1. Use Hungarian (`scipy.optimize.linear_sum_assignment`) for GT↔detection.
2. Optional match radius = `max(tolerance, k * major_radius)` when FWHM available.
3. Persist GT with frame index + image shape; reject clicks if display scale metadata mismatches.
4. Run grid search with the **same** glare mask policy as `TearFilmAnalyzer`.

---

## 5. Noise, Tracking Edge Cases, and MMS Velocity

### MMS definition

```699:702:tear_film_advanced.py
                delta_t = 1.0 / fps
                mms_velocity = (best_distance / delta_t) * 0.1
```

- `best_distance` is in **normalized** units (superior–inferior distance = 1).
- Result is “normalized units per 0.1 s,” **not physical mm/s**, despite UI labels.
- Only matched tracks get velocity; **new IDs have no velocity** on birth frame.
- Only `velocity > 0` rows are saved → stationary / unmatched births omitted.

### Tracker reset

```719:722:tear_film_advanced.py
    def reset(self):
        self.tracked_objects = {}
        self.trajectories = {}
        # next_id NOT reset
```

IDs keep increasing across epochs (OK for uniqueness). Tracks do not carry across blinks (intended).

### Tespit edilen bug’lar / zafiyetler

#### BUG-T1 — Greedy NN identity swaps → fake velocity spikes

Matching loops **detections**, each taking the nearest remaining track within `max_tracking_distance` (0.2 norm). No mutual best / auction / Kalman. Crowded tear film → swaps → one-frame jumps near max distance → large MMS injected into early or mid bins → power-law destruction.

#### BUG-T2 — Track death + rebirth looks like “new particle”

If detection flickers (threshold noise), track is lost; next frame creates **new ID** with no velocity, then a jump velocity on second frame. CSV gains fragmented trajectories and bursty speeds.

#### BUG-T3 — Unmatched old tracks silently discarded

Any track not matched in the current frame is dropped from `tracked_objects` (not kept with coasting). High disappearance rate → many one-frame “matches” that are actually different physical particles.

#### BUG-T4 — Fixation light poison (critical cross-cutting bug)

```311:329:tear_film_advanced.py
        self.superior_light = centers[0]
        self.inferior_light = centers[1]
        self.normalization_distance = ...
        if ... geometry OK:
            return True
        return False  # lights ALREADY assigned
```

```1055:1060:tear_film_advanced.py
                if self.glare_excluder.superior_light is None:
                    if not self.glare_excluder.detect_fixation_lights(gray):
                        continue
                glare_mask = self.glare_excluder.create_glare_mask(gray.shape)
```

**Sequence on geometry failure:**

1. Frame N: detect sets lights, returns `False` → `continue`.
2. Frame N+1: `superior_light is not None` → **skip redetect**.
3. Analysis proceeds with **failed-geometry** lights → wrong normalization scale and glare holes.

This alone can make some videos “excellent” (lights validate) and others “chaotic” (lights fail geometry once, then poison all subsequent frames) — matching the reported symptom pattern.

Also: lights are detected **once** and never updated (drift / head movement unhandled).

#### BUG-T5 — `visualize_detection` null-dereference risk

Drawing circles when `superior_light` is still `None` after hard failure.

### Çözüm önerileri

1. **Fix detect_fixation_lights:** on failure, clear `superior_light` / `inferior_light` / `normalization_distance`; only assign on success.
2. In `_process_epochs`, treat “valid lights” as an explicit flag, not `is not None`.
3. Replace greedy NN with bipartite matching; add max age / coasting for missed detections.
4. Cap MMS (e.g. reject `distance > 0.5 * max_tracking_distance` as match) or winsorize before export.
5. Relabel units honestly; optionally convert via measured inter-light mm if available.
6. Export track length / age; exclude first match after birth from power-law bins.

---

## Consolidated Bug List (Priority)

| ID | Severity | Area | One-line finding |
|----|----------|------|------------------|
| BUG-F1 | **Critical** | Fixation / analysis | Failed geometry still sets lights; later frames skip redetect → poisoned normalization/glare |
| BUG-S1/S2 | **High** | UI state | Sliders hardcoded; config only updates on Apply → silent wrong params |
| BUG-T1/T2 | **High** | Tracking | Greedy NN + rebirth → velocity spikes / gaps |
| BUG-E1/E2 | **High** | Time base | `time_since_blink_s` ≠ reliable post-blink time; epoch0 & skip skew |
| BUG-E3 | **Medium** | Blink pad | `blink_pad_frames=1` allows near-blink contamination |
| BUG-P2/P3 | **Medium** | Power-law | Linear fit + all-epoch pooling → negative R² on noisy videos |
| BUG-S4 | **Medium** | Validation | Opt glare fallback ≠ analyzer path |
| BUG-V2/V3 | **Medium** | Matching | Greedy + fixed 5 px tolerance, no FWHM |
| BUG-T5 | **Low–Med** | UI viz | Circle draw when lights None / inconsistent mask |
| BUG-V5 | **Low** | UI heatmap | Float equality for best cell |

---

## Mimari / Matematiksel Zafiyetler (Özet)

1. **No single config sync layer** between widgets, optimizer, and analyzer.  
2. **Normalization and glare** hinge on a brittle two-blob + geometry heuristic with a state machine bug (BUG-F1).  
3. **Velocity is link-length based**, not a filtered kinematic estimate — sensitive to ID error.  
4. **Power-law assumes** every epoch is a clean post-blink decay in the same units; pipeline does not enforce that.  
5. **R² is honest but unguarded** — negative values are reported as clinical curves instead of rejected fits.  
6. **Validation optimizes detection thresholds**, not tracking/timebase — cannot fix chaotic MMS by thresh_k alone.

---

## Çözüm Önerileri (Önerilen Uygulama Sırası)

### Phase A — Correctness (do first)

1. Fix `detect_fixation_lights` assignment-on-failure; add `lights_valid` flag.  
2. Bind Titration sliders to `session_state.config`; sync before Run Analysis.  
3. Frame-based `time_since_blink_s`; exclude non-post-blink epoch 0 from pooled fit.

### Phase B — Signal quality

4. Stronger blink padding / epoch edge trim.  
5. Bipartite tracking + velocity winsorization.  
6. Align Optimization glare policy with analyzer.

### Phase C — Modeling

7. Log-space power-law init; per-epoch fit + quality gate (reject R² < 0).  
8. Hungarian GT matching; optional FWHM-aware tolerance.  
9. Honest unit labeling and clinical QC flags in UI.

---

## Mapping to Reported Symptom (“good R² vs negative R²”)

| When video “works” | When video “fails” |
|--------------------|--------------------|
| Fixation geometry validates → stable norm distance | Geometry fails once → BUG-F1 poisons scale/mask |
| Clear blinks, long epochs, real decay | Short/noisy epochs, epoch0 mixed in, thin padding |
| Tracking mostly consistent | Crowding → ID swaps → flat/spiky MMS |
| User applied params and didn’t overwrite via Titration | Sliders reset / never Applied → suboptimal detection |
| Power-law shape present in medians | Medians flat → **correctly** negative R² |

**Conclusion:** Negative R² is often a **faithful diagnostic** that the exported velocity-vs-time series is not a power-law decay — frequently due to BUG-F1, timebase pooling, and tracking noise — compounded by UI config sync gaps that make parameters appear “set” when they are not.

---

## Files Reviewed

- `tear_film_advanced.py` — `TearFilmConfig`, `BlinkDetector`, `EpochSegmenter`, `GlareExcluder`, `ParticleDetector`, `ParticleTracker`, `ValidationOptimizer`, `TearFilmAnalyzer`, `compute_power_law_decay`
- `tear_film_ui.py` — Titration / Blink / Run Analysis / Results / Optimization parameter and state paths

---

**Audit status:** Complete (analysis only; no code modifications per request).  
**Recommended next step:** Implement Phase A fixes, then re-run the same “good” vs “bad” videos with identical config dumps logged at analysis start.
