import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import sys

# 1. Veriyi Oku
csv_file = "tear_film_analysis_advanced.csv" if len(sys.argv) < 2 else sys.argv[1]

try:
    df = pd.read_csv(csv_file)
    print(f"✅ Loaded: {csv_file}")
    print(f"📊 Columns: {list(df.columns)}")
except FileNotFoundError:
    print(f"❌ File not found: {csv_file}")
    print("Usage: python veri_analizi.py [csv_file]")
    sys.exit(1)

# 2. Use time_since_blink_s if available (modern approach)
if 'time_since_blink_s' in df.columns:
    time_col = 'time_since_blink_s'
    print("✅ Using 'time_since_blink_s' (epoch-relative time)")
elif 'Zaman(sn)' in df.columns:
    time_col = 'Zaman(sn)'
    print("⚠️ Using legacy 'Zaman(sn)' column")
else:
    print("❌ No time column found")
    sys.exit(1)

# Determine velocity column
if 'mms_velocity' in df.columns:
    velocity_col = 'mms_velocity'
elif 'MMS_Hizi' in df.columns:
    velocity_col = 'MMS_Hizi'
else:
    print("❌ No velocity column found")
    sys.exit(1)

# 3. Filter valid data (positive time and velocity)
temiz_df = df[(df[time_col] > 0) & (df[velocity_col] > 0)].copy()
print(f"📈 Valid data points: {len(temiz_df)} / {len(df)}")

# 4. Binning: Group by time intervals
bin_size = 0.15  # seconds
max_time = temiz_df[time_col].max()
bins = np.arange(0, max_time + bin_size, bin_size)
temiz_df['time_bin'] = pd.cut(temiz_df[time_col], bins=bins)

# Calculate median velocity per bin (more robust than mean)
binned_stats = temiz_df.groupby('time_bin', observed=True)[velocity_col].agg(['median', 'mean', 'count'])
binned_stats = binned_stats[binned_stats['count'] >= 3]  # Require at least 3 points per bin

print(f"📊 Number of bins: {len(binned_stats)}")

# Extract bin centers and median velocities
zaman_verisi = np.array([interval.mid for interval in binned_stats.index])
mms_verisi = binned_stats['median'].values

# Remove any NaN values
valid_mask = (~np.isnan(zaman_verisi)) & (~np.isnan(mms_verisi))
zaman_verisi = zaman_verisi[valid_mask]
mms_verisi = mms_verisi[valid_mask]

# 5. Power-Law Eğri Uydurma: v = α * t^(-β)
def power_law(t, alpha, beta):
    return alpha * np.power(t, -beta)

try:
    if len(zaman_verisi) < 4:
        raise ValueError("Insufficient data points for curve fitting (need at least 4 bins)")
    
    # Initial guess: alpha ~ first velocity, beta ~ 0.5
    initial_guess = [mms_verisi[0], 0.5]
    
    # Fit the curve with physical bounds
    popt, pcov = curve_fit(
        power_law, 
        zaman_verisi, 
        mms_verisi,
        p0=initial_guess,
        bounds=([0.01, 0.01], [100, 3.0]),
        maxfev=5000
    )
    alpha, beta = popt
    
    # Calculate fitted values
    fitted_values = power_law(zaman_verisi, alpha, beta)
    
    # Calculate R² (goodness of fit)
    ss_res = np.sum((mms_verisi - fitted_values) ** 2)
    ss_tot = np.sum((mms_verisi - np.mean(mms_verisi)) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

    # Clinical metrics (eMMSi and eMMSf)
    eMMSi = power_law(0.1, alpha, beta)  # Initial velocity at t=0.1s
    eMMSf = power_law(2.0, alpha, beta)  # Final velocity at t=2.0s

    # 6. Sonuçları Konsola Yazdır
    print("\n" + "=" * 60)
    print("  DİNAMİK GÖZYAŞI FİLMİ HOMEOSTAZİ SONUÇLARI")
    print("  Power-Law Decay Model: v = α × t^(-β)")
    print("=" * 60)
    print(f"  Alpha (α) - İlk Hız Katsayısı:     {alpha:.4f}")
    print(f"  Beta (β) - Yavaşlama Üssü:         {beta:.4f}")
    print(f"  R² - Model Uyum Skoru:             {r_squared:.4f}")
    print("-" * 60)
    print(f"  eMMSi (t=0.1s) - İlk Dönem Hızı:   {eMMSi:.4f} mm/s")
    print(f"  eMMSf (t=2.0s) - Son Dönem Hızı:   {eMMSf:.4f} mm/s")
    print(f"  Hız Düşüşü (eMMSi/eMMSf):          {eMMSi/eMMSf:.2f}x")
    print("=" * 60)
    
    # Clinical interpretation
    print("\n📋 Klinik Yorumlama:")
    if 0.3 <= beta <= 0.8:
        print("  ✅ Beta değeri normal aralıkta (0.3-0.8)")
    elif beta < 0.3:
        print("  ⚠️ Beta düşük: Yavaş yayılma, gözyaşı instabilitesi olabilir")
    else:
        print("  ⚠️ Beta yüksek: Hızlı yayılma, hiperozmolarite olabilir")
    
    if r_squared > 0.8:
        print(f"  ✅ Model uyumu mükemmel (R²={r_squared:.3f})")
    elif r_squared > 0.6:
        print(f"  ⚠️ Model uyumu orta (R²={r_squared:.3f})")
    else:
        print(f"  ❌ Model uyumu zayıf (R²={r_squared:.3f}) - veri kalitesini kontrol edin")

    # 7. Görselleştirme
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Left plot: Power-law curve
    ax1.scatter(zaman_verisi, mms_verisi, s=80, alpha=0.6, color='steelblue', 
               label='Binned Median Velocity', zorder=3)
    
    t_smooth = np.linspace(zaman_verisi.min(), zaman_verisi.max(), 200)
    ax1.plot(t_smooth, power_law(t_smooth, alpha, beta), 
            linewidth=3, color='crimson', 
            label=f'Fitted: v = {alpha:.3f} × t^(-{beta:.3f})', zorder=4)
    
    ax1.text(0.02, 0.98, 
            f"R² = {r_squared:.4f}\nα = {alpha:.3f}\nβ = {beta:.3f}",
            transform=ax1.transAxes, fontsize=11,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    ax1.set_xlabel('Time Since Blink (seconds)', fontsize=12)
    ax1.set_ylabel('Velocity (mm/s)', fontsize=12)
    ax1.set_title('Power-Law Decay Curve', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.legend(loc='upper right', fontsize=10)
    
    # Right plot: Residuals
    residuals = mms_verisi - fitted_values
    ax2.scatter(zaman_verisi, residuals, s=60, alpha=0.6, color='orange')
    ax2.axhline(y=0, color='red', linestyle='--', linewidth=2)
    ax2.set_xlabel('Time Since Blink (seconds)', fontsize=12)
    ax2.set_ylabel('Residuals (mm/s)', fontsize=12)
    ax2.set_title('Fit Residuals', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.show()
    
    print("\n✅ Analiz tamamlandı!")

except Exception as e:
    print(f"\n❌ Power-law fitting başarısız: {e}")
    print("Veri kalitesini veya bin parametrelerini kontrol edin.")