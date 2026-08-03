"""
Test script for Power-Law Decay Curve Analysis
================================================

Tests the compute_power_law_decay function with synthetic and real data.

Usage:
    python test_power_law.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tear_film_advanced import compute_power_law_decay


def test_synthetic_data():
    """Test power-law fitting with synthetic data."""
    print("\n" + "="*60)
    print("TEST 1: Synthetic Data (Known Parameters)")
    print("="*60)
    
    # Generate synthetic data with known parameters
    true_alpha = 2.5
    true_beta = 0.5
    
    np.random.seed(42)
    time_points = np.linspace(0.1, 3.0, 100)
    
    # True power-law with noise
    true_velocity = true_alpha * np.power(time_points, -true_beta)
    noise = np.random.normal(0, 0.1, len(time_points))
    noisy_velocity = true_velocity + noise
    
    # Create DataFrame
    df = pd.DataFrame({
        'time_since_blink_s': time_points,
        'mms_velocity': noisy_velocity,
        'epoch': 0
    })
    
    # Compute power-law
    result = compute_power_law_decay(df, bin_size=0.15)
    
    if result:
        print(f"✅ Fitting successful!")
        print(f"   True α: {true_alpha:.3f} | Fitted α: {result['alpha']:.3f}")
        print(f"   True β: {true_beta:.3f} | Fitted β: {result['beta']:.3f}")
        print(f"   R²: {result['r_squared']:.4f}")
        print(f"   Equation: {result['equation']}")
        
        # Check accuracy
        alpha_error = abs(result['alpha'] - true_alpha) / true_alpha
        beta_error = abs(result['beta'] - true_beta) / true_beta
        
        if alpha_error < 0.1 and beta_error < 0.1:
            print("   ✅ Parameters recovered within 10% error")
        else:
            print(f"   ⚠️ Parameter error: α={alpha_error*100:.1f}%, β={beta_error*100:.1f}%")
        
        # Plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Data and fit
        ax1.scatter(time_points, noisy_velocity, alpha=0.3, s=20, label='Noisy Data')
        ax1.plot(time_points, true_velocity, 'g--', linewidth=2, label='True Model')
        ax1.plot(result['binned_time'], result['fitted_curve'], 'r-', linewidth=2, label='Fitted Model')
        ax1.set_xlabel('Time Since Blink (s)')
        ax1.set_ylabel('Velocity (mm/s)')
        ax1.set_title('Synthetic Data: Power-Law Recovery')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Binned data
        ax2.scatter(result['binned_time'], result['binned_velocity'], s=80, alpha=0.6, color='steelblue')
        ax2.plot(result['binned_time'], result['fitted_curve'], 'r-', linewidth=2)
        ax2.set_xlabel('Time Since Blink (s)')
        ax2.set_ylabel('Binned Median Velocity (mm/s)')
        ax2.set_title('Binned Data and Fit')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('test_power_law_synthetic.png', dpi=150)
        print("   📊 Plot saved: test_power_law_synthetic.png")
        plt.show()
        
    else:
        print("❌ Fitting failed!")


def test_real_csv():
    """Test with real CSV file if available."""
    print("\n" + "="*60)
    print("TEST 2: Real CSV Data")
    print("="*60)
    
    import os
    csv_file = "tear_film_analysis_advanced.csv"
    
    if not os.path.exists(csv_file):
        print(f"⚠️ CSV file not found: {csv_file}")
        print("   Run analysis first to generate data.")
        return
    
    df = pd.read_csv(csv_file)
    print(f"✅ Loaded: {csv_file}")
    print(f"   Rows: {len(df)}")
    print(f"   Columns: {list(df.columns)}")
    
    if 'time_since_blink_s' not in df.columns:
        print("❌ time_since_blink_s column not found!")
        print("   Re-run analysis with updated code.")
        return
    
    # Test different bin sizes
    bin_sizes = [0.10, 0.15, 0.20, 0.25]
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    for idx, bin_size in enumerate(bin_sizes):
        result = compute_power_law_decay(df, bin_size=bin_size)
        
        if result:
            ax = axes[idx]
            ax.scatter(result['binned_time'], result['binned_velocity'], 
                      s=60, alpha=0.6, color='steelblue')
            ax.plot(result['binned_time'], result['fitted_curve'], 
                   'r-', linewidth=2)
            ax.set_title(f"Bin={bin_size}s | α={result['alpha']:.2f}, β={result['beta']:.2f}, R²={result['r_squared']:.3f}")
            ax.set_xlabel('Time Since Blink (s)')
            ax.set_ylabel('Velocity (mm/s)')
            ax.grid(True, alpha=0.3)
            
            print(f"   Bin {bin_size}s: α={result['alpha']:.3f}, β={result['beta']:.3f}, R²={result['r_squared']:.4f}")
    
    plt.tight_layout()
    plt.savefig('test_power_law_real.png', dpi=150)
    print("   📊 Plot saved: test_power_law_real.png")
    plt.show()


def test_edge_cases():
    """Test edge cases and error handling."""
    print("\n" + "="*60)
    print("TEST 3: Edge Cases")
    print("="*60)
    
    # Case 1: Insufficient data
    print("\n1. Insufficient data (only 2 points):")
    df_tiny = pd.DataFrame({
        'time_since_blink_s': [0.1, 0.2],
        'mms_velocity': [1.0, 0.9],
        'epoch': 0
    })
    result = compute_power_law_decay(df_tiny)
    if result is None:
        print("   ✅ Correctly rejected insufficient data")
    else:
        print("   ❌ Should have rejected insufficient data")
    
    # Case 2: Zero/negative times
    print("\n2. Zero/negative times:")
    df_bad_time = pd.DataFrame({
        'time_since_blink_s': [-0.1, 0.0, 0.1, 0.2, 0.3] * 20,
        'mms_velocity': [1.0, 1.1, 1.2, 1.0, 0.9] * 20,
        'epoch': 0
    })
    result = compute_power_law_decay(df_bad_time)
    if result:
        print(f"   ✅ Handled bad times, used {len(result['binned_time'])} valid bins")
    else:
        print("   ⚠️ Rejected due to bad times")
    
    # Case 3: Very noisy data
    print("\n3. Very noisy data (high variance):")
    np.random.seed(123)
    time_points = np.linspace(0.1, 2.0, 200)
    base_velocity = 2.0 * np.power(time_points, -0.5)
    noisy_velocity = base_velocity + np.random.normal(0, base_velocity * 0.5, len(time_points))
    
    df_noisy = pd.DataFrame({
        'time_since_blink_s': time_points,
        'mms_velocity': noisy_velocity.clip(0.01),  # Ensure positive
        'epoch': 0
    })
    
    result = compute_power_law_decay(df_noisy, bin_size=0.15)
    if result:
        print(f"   ✅ Fitted noisy data: R²={result['r_squared']:.4f}")
        if result['r_squared'] < 0.8:
            print("   ⚠️ Low R² indicates poor fit (expected for noisy data)")
    else:
        print("   ❌ Failed to fit noisy data")


def main():
    """Run all tests."""
    print("\n" + "="*70)
    print("  POWER-LAW DECAY CURVE ANALYSIS - TEST SUITE")
    print("="*70)
    
    test_synthetic_data()
    test_real_csv()
    test_edge_cases()
    
    print("\n" + "="*70)
    print("  ALL TESTS COMPLETED")
    print("="*70)
    print("\n✅ If all tests passed, the power-law analysis is working correctly!")


if __name__ == "__main__":
    main()
