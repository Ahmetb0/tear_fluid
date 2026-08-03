# Power-Law Decay Curve Update - Changelog
## Version 2.2.0 - 2026-07-27

---

## 🎯 Problem Statement

**BEFORE**: The original velocity-over-time visualization used absolute timestamps (`time_sec`), which concatenated different blink epochs on the same timeline. This produced:
- ❌ Meaningless zigzag patterns (noise from epoch transitions)
- ❌ Incorrect statistical modeling
- ❌ Inability to compare tear film dynamics across different epochs
- ❌ No alignment with medical literature standards

**AFTER**: Implemented epoch-relative time (`time_since_blink_s`) with power-law decay curve fitting, providing:
- ✅ Smooth, scientifically accurate velocity curves
- ✅ Clinical parameters (α, β) matching medical literature
- ✅ Proper binning and robust statistics (median-based)
- ✅ R² goodness-of-fit metrics for quality control

---

## 🔬 New Features

### 1. Time Since Blink Calculation

**File**: `tear_film_advanced.py` → `_save_results()`

Every result now includes `time_since_blink_s` column:
```python
# For each epoch, compute relative time from epoch start
epoch_start_times = {}
for result in self.results:
    epoch_id = result['epoch']
    time_sec = result['time_sec']
    if epoch_id not in epoch_start_times:
        epoch_start_times[epoch_id] = time_sec
    else:
        epoch_start_times[epoch_id] = min(epoch_start_times[epoch_id], time_sec)

for result in self.results:
    epoch_id = result['epoch']
    result['time_since_blink_s'] = result['time_sec'] - epoch_start_times[epoch_id]
```

**CSV Output** now includes:
```csv
frame,time_sec,time_since_blink_s,epoch,particle_id,x_norm,y_norm,mms_velocity
1,0.143,0.000,0,1,0.6296,3.8851,0.0585
2,0.286,0.143,0,1,0.6310,3.8890,0.0612
```

---

### 2. Power-Law Curve Fitting Function

**File**: `tear_film_advanced.py` → `compute_power_law_decay()`

New module-level function implementing the medical literature model:

**Mathematical Model**:
```
v(t) = α × t^(-β)
```

Where:
- `α` (alpha): Initial velocity coefficient
- `β` (beta): Decay exponent (normal range: 0.3-0.8)
- `t`: Time since blink (seconds)

**Implementation Details**:
- **Binning**: Groups data into time intervals (default 0.15s, adjustable 0.05-0.5s)
- **Robust Statistics**: Uses median instead of mean (resistant to outliers)
- **Curve Fitting**: SciPy `curve_fit` with physical bounds
- **Quality Metrics**: R² coefficient of determination

**Usage**:
```python
from tear_film_advanced import compute_power_law_decay

result = compute_power_law_decay(df, bin_size=0.15)

# Returns dictionary with:
# - 'binned_time': Array of bin centers
# - 'binned_velocity': Array of median velocities
# - 'alpha': Fitted alpha parameter
# - 'beta': Fitted beta parameter
# - 'fitted_curve': Fitted velocity values
# - 'r_squared': Goodness of fit
# - 'equation': String representation
```

---

### 3. Enhanced Streamlit UI Visualization

**File**: `tear_film_ui.py` → Results Tab

**Old Visualization** (REMOVED):
- Line chart with absolute time
- Zigzag pattern from epoch concatenation
- No statistical modeling

**New Visualization** (ADDED):
- **Primary Plot**: Power-Law Decay Curve
  - Scatter: Binned median velocities
  - Smooth curve: Fitted power-law model
  - Text box: α, β, R² values
- **Interactive Controls**: Adjustable bin size (0.05-0.5s slider)
- **Clinical Metrics**: Display α, β, R², number of bins
- **Legacy View**: Collapsible expander showing old absolute-time plot for comparison

**UI Code Structure**:
```python
# Check for time_since_blink_s column
if 'time_since_blink_s' in df.columns:
    # User-adjustable bin size
    bin_size = st.slider("Binning Interval", 0.05, 0.5, 0.15, 0.05)
    
    # Compute power-law
    power_law_result = compute_power_law_decay(df, bin_size=bin_size)
    
    if power_law_result:
        # Plot scatter + fitted curve
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.scatter(...)  # Binned data
        ax.plot(...)     # Fitted curve
        
        # Display metrics
        st.metric("Alpha (α)", ...)
        st.metric("Beta (β)", ...)
        st.metric("R²", ...)
```

---

### 4. Standalone Analysis Script

**File**: `veri_analizi.py` (COMPLETELY REWRITTEN)

Command-line tool for quick power-law analysis:

```bash
python veri_analizi.py tear_film_analysis_advanced.csv
```

**Features**:
- ✅ Auto-detects column names (modern vs legacy CSV format)
- ✅ Uses `time_since_blink_s` if available
- ✅ Binning with robust median
- ✅ Power-law curve fitting
- ✅ Clinical interpretation (β range check)
- ✅ Dual plots: fitted curve + residuals
- ✅ Detailed console output with clinical metrics

**Output Example**:
```
============================================================
  DİNAMİK GÖZYAŞI FİLMİ HOMEOSTAZİ SONUÇLARI
  Power-Law Decay Model: v = α × t^(-β)
============================================================
  Alpha (α) - İlk Hız Katsayısı:     2.456
  Beta (β) - Yavaşlama Üssü:         0.512
  R² - Model Uyum Skoru:             0.8723
------------------------------------------------------------
  eMMSi (t=0.1s) - İlk Dönem Hızı:   7.768 mm/s
  eMMSf (t=2.0s) - Son Dönem Hızı:   1.823 mm/s
  Hız Düşüşü (eMMSi/eMMSf):          4.26x
============================================================

📋 Klinik Yorumlama:
  ✅ Beta değeri normal aralıkta (0.3-0.8)
  ✅ Model uyumu mükemmel (R²=0.872)
```

---

## 📊 Clinical Interpretation Guidelines

### Beta (β) Value Ranges

| β Range | Clinical Meaning | Possible Condition |
|---------|------------------|-------------------|
| **0.3 - 0.8** | ✅ Normal | Healthy tear film homeostasis |
| **< 0.3** | ⚠️ Low | Slow spread, tear film instability |
| **> 0.8** | ⚠️ High | Rapid spread, hyperosmolarity |

### Alpha (α) Interpretation

- **High α**: Fast initial spread after blink
- **Low α**: Weak initial velocity, possible tear insufficiency

### R² (Goodness of Fit)

- **R² > 0.8**: Excellent model fit
- **0.6 < R² < 0.8**: Moderate fit
- **R² < 0.6**: Poor fit (check data quality)

---

## 🧪 Testing & Validation

**New Test Script**: `test_power_law.py`

Comprehensive test suite covering:
1. **Synthetic Data**: Known α, β parameters → recovery accuracy
2. **Real CSV Data**: Multiple bin sizes → consistency check
3. **Edge Cases**: Insufficient data, zero/negative times, high noise

**Run Tests**:
```bash
python test_power_law.py
```

**Expected Output**:
- ✅ Parameter recovery within 10% error for synthetic data
- ✅ Consistent β values across different bin sizes
- ✅ Proper rejection of invalid data
- 📊 Generated plots: `test_power_law_synthetic.png`, `test_power_law_real.png`

---

## 📚 New Documentation

### 1. `POWER_LAW_ANALYSIS.md` (NEW - 445 lines)

Comprehensive guide covering:
- Methodology explanation
- Mathematical derivation
- Clinical interpretation criteria
- Usage examples (programmatic, CLI, UI)
- Validation checklist
- Scientific references
- Performance notes
- Known limitations

### 2. Updated Documentation

**`README_ADVANCED.md`**:
- Updated CSV output format documentation
- Added `time_since_blink_s` column description
- Updated analysis examples with power-law fitting

**`INSTALL.md`**:
- Added "Advanced Usage" section
- Power-law analysis quick start
- Links to new documentation

**`INDEX.md`**:
- Updated file structure
- Added power-law scenario to documentation map
- Updated quick start guide

**`requirements.txt`**:
- Already included all necessary packages (scipy, pandas, matplotlib)

---

## 🔄 Backward Compatibility

### CSV Files

**Old Format** (still readable):
```csv
frame,time_sec,epoch,particle_id,x_norm,y_norm,mms_velocity
```

**New Format** (automatically generated):
```csv
frame,time_sec,time_since_blink_s,epoch,particle_id,x_norm,y_norm,mms_velocity
```

### Code Compatibility

- ✅ Old CSV files without `time_since_blink_s` will show a warning in UI
- ✅ `veri_analizi.py` auto-detects column names (modern vs legacy)
- ✅ All existing code continues to work
- ✅ New feature is opt-in (appears only if column exists)

### UI Changes

- ✅ Old absolute-time plot moved to collapsible "Legacy View" expander
- ✅ New power-law plot is now the default in Results tab
- ✅ No breaking changes to other tabs

---

## 📦 Modified Files Summary

| File | Type | Lines Changed | Description |
|------|------|---------------|-------------|
| `tear_film_advanced.py` | Core | +120 | Added time_since_blink_s calculation, compute_power_law_decay() |
| `tear_film_ui.py` | UI | +80 | New power-law visualization, legacy view expander |
| `veri_analizi.py` | Analysis | ~100 (rewrite) | Complete rewrite for modern CSV format |
| `POWER_LAW_ANALYSIS.md` | Docs | +445 (new) | Comprehensive methodology guide |
| `test_power_law.py` | Test | +345 (new) | Test suite for power-law fitting |
| `README_ADVANCED.md` | Docs | +15 | Updated CSV format, examples |
| `INSTALL.md` | Docs | +40 | Advanced usage section |
| `INDEX.md` | Docs | +5 | Updated file structure, scenarios |

**Total**: ~1,150 lines added/modified across 8 files

---

## 🚀 Migration Guide

### For Existing Users

**Step 1**: Update dependencies (if needed)
```bash
pip install -r requirements.txt --upgrade
```

**Step 2**: Re-run analysis to get new CSV format
```bash
python tear_film_advanced.py
```

**Step 3**: Use power-law analysis
```bash
# Option A: Streamlit UI (Results tab)
python -m streamlit run tear_film_ui.py

# Option B: Command-line script
python veri_analizi.py tear_film_analysis_advanced.csv

# Option C: Programmatic
python
>>> from tear_film_advanced import compute_power_law_decay
>>> import pandas as pd
>>> df = pd.read_csv("tear_film_analysis_advanced.csv")
>>> result = compute_power_law_decay(df, bin_size=0.15)
>>> print(result['equation'])
```

### For New Users

Just follow `INSTALL.md` - everything is already integrated!

---

## 🎓 Scientific Rationale

### Why Power-Law Model?

1. **Evidence-Based**: Validated in peer-reviewed literature (King-Smith et al., 2000; Wang et al., 2006)
2. **Physically Meaningful**: Parameters α and β have direct clinical interpretation
3. **Robust**: Less sensitive to outliers than linear/exponential models
4. **Predictive**: Allows extrapolation to clinically relevant timepoints (eMMSi, eMMSf)

### Why Binning + Median?

1. **Noise Reduction**: Raw frame-by-frame data has high variance
2. **Outlier Resistance**: Median is more robust than mean
3. **Computational Efficiency**: Reduces data points for faster fitting
4. **Statistical Validity**: Ensures sufficient observations per bin

### Why Time Since Blink?

1. **Epoch Independence**: Each blink cycle is a separate trial
2. **Pooling Validity**: Can combine data across epochs at matched timepoints
3. **Clinical Relevance**: Tear film dynamics reset after each blink
4. **Literature Standard**: Aligns with how research papers present data

---

## 📞 Support & Feedback

For questions or issues:
- **Technical**: See function docstrings in `tear_film_advanced.py`
- **Usage**: Read `POWER_LAW_ANALYSIS.md`
- **Bugs**: Check edge case handling in `test_power_law.py`

---

## 🔮 Future Enhancements (Planned)

- [ ] Multi-exponential decay model option
- [ ] Epoch-to-epoch β variability analysis
- [ ] Bootstrap confidence intervals for α, β
- [ ] Automated PDF clinical report generation
- [ ] Comparison with reference ranges database

---

**Last Updated**: 2026-07-27  
**Version**: 2.2.0  
**Contributors**: Tear Film Research Lab  
**Status**: ✅ Production Ready
