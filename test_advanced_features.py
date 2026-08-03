"""
Test Script for Advanced Features (FWHM & ValidationOptimizer)
===============================================================
Tests the new FWHM shape analysis and parameter optimization features.
"""

import cv2
import numpy as np
from tear_film_advanced import (
    TearFilmConfig, TearFilmAnalyzer,
    ParticleDetector, GlareExcluder, ValidationOptimizer
)


def test_fwhm_analysis():
    """Test FWHM-based particle shape analysis."""
    print("\n" + "="*60)
    print("TEST 1: FWHM Shape Analysis")
    print("="*60)
    
    # Configuration with FWHM enabled
    config = TearFilmConfig(
        video_path="C:/Users/Asus/Desktop/tear_fluid/assests/AYDIN_MEHMET TUNAHAN_Right_2026_07_26-14_01_16.mkv",
        output_csv="test_fwhm_output.csv",
        fwhm_enabled=True,
        show_visualization=False
    )
    
    # Run analysis
    print("\nRunning analysis with FWHM enabled...")
    analyzer = TearFilmAnalyzer(config)
    analyzer.analyze_video()
    
    # Check results
    import pandas as pd
    df = pd.read_csv("test_fwhm_output.csv")
    
    print(f"\n✅ Analysis complete!")
    print(f"Total records: {len(df)}")
    print(f"\nCSV Columns: {list(df.columns)}")
    
    # Check if FWHM columns exist
    fwhm_columns = ['major_radius', 'minor_radius', 'orientation', 'elongation']
    has_fwhm = all(col in df.columns for col in fwhm_columns)
    
    if has_fwhm:
        print("\n✅ FWHM columns present in CSV!")
        print("\nFWHM Statistics:")
        print(f"  Major Radius - Mean: {df['major_radius'].mean():.4f}, "
              f"Std: {df['major_radius'].std():.4f}")
        print(f"  Minor Radius - Mean: {df['minor_radius'].mean():.4f}, "
              f"Std: {df['minor_radius'].std():.4f}")
        print(f"  Elongation - Mean: {df['elongation'].mean():.4f}, "
              f"Max: {df['elongation'].max():.4f}")
        
        # Find most elongated particles (potential motion streaks)
        elongated = df[df['elongation'] > 2.0]
        print(f"\n  Highly elongated particles (elongation > 2.0): {len(elongated)} "
              f"({len(elongated)/len(df)*100:.1f}%)")
    else:
        print("\n❌ FWHM columns missing!")
        return False
    
    return True


def test_validation_optimizer():
    """Test ValidationOptimizer with synthetic ground truth."""
    print("\n" + "="*60)
    print("TEST 2: Validation Optimizer")
    print("="*60)
    
    # Load a sample frame
    video_path = "C:/Users/Asus/Desktop/tear_fluid/assests/AYDIN_MEHMET TUNAHAN_Right_2026_07_26-14_01_16.mkv"
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 30)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        print("❌ Could not load video frame")
        return False
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Setup config and detectors
    config = TearFilmConfig(
        thresh_k=3.0,
        floor_threshold=0.5,
        validation_match_tolerance=5.0
    )
    
    glare_excluder = GlareExcluder(config)
    if not glare_excluder.detect_fixation_lights(gray):
        print("❌ Could not detect fixation lights")
        return False
    
    print("\n✅ Fixation lights detected")
    
    # Create glare mask
    glare_mask = glare_excluder.create_glare_mask(gray.shape)
    
    # Detect particles with default parameters
    detector = ParticleDetector(config)
    particles = detector.detect_particles(gray, glare_mask)
    
    print(f"✅ Detected {len(particles)} particles with default settings")
    
    # Create synthetic ground truth (use detected particles as approximate truth)
    # In real use, this would be manually annotated
    if len(particles) < 5:
        print("❌ Too few particles for meaningful test")
        return False
    
    # Use first 10 particles as "ground truth"
    ground_truth = [(p['x'], p['y']) for p in particles[:10]]
    print(f"✅ Created synthetic ground truth with {len(ground_truth)} particles")
    
    # Initialize optimizer
    optimizer = ValidationOptimizer(config)
    optimizer.set_ground_truth(ground_truth)
    
    # Run grid search (small range for testing)
    print("\n⏳ Running grid search (this will take ~30 seconds)...")
    best_result = optimizer.suggest_settings(
        gray,
        glare_mask,
        thresh_k_range=(2.0, 5.0, 1.0),  # Small range for quick test
        floor_range=(0.0, 1.0, 0.5),
        verbose=True
    )
    
    print(f"\n✅ Optimization complete!")
    print(f"\nBest Parameters:")
    print(f"  thresh_k: {best_result['thresh_k']:.2f}")
    print(f"  floor_threshold: {best_result['floor_threshold']:.2f}")
    print(f"  F1 Score: {best_result['f1']:.4f}")
    print(f"  Precision: {best_result['precision']:.4f}")
    print(f"  Recall: {best_result['recall']:.4f}")
    
    # Export results
    optimizer.export_results("test_optimization_results.csv")
    print(f"\n✅ Results exported to test_optimization_results.csv")
    
    return True


def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("ADVANCED FEATURES TEST SUITE")
    print("Version 2.1.0")
    print("="*60)
    
    tests = [
        ("FWHM Shape Analysis", test_fwhm_analysis),
        ("Validation Optimizer", test_validation_optimizer),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            success = test_func()
            results[test_name] = "✅ PASS" if success else "❌ FAIL"
        except Exception as e:
            print(f"\n❌ Test failed with error: {e}")
            import traceback
            traceback.print_exc()
            results[test_name] = "❌ ERROR"
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    for test_name, result in results.items():
        print(f"{test_name:30s} {result}")
    
    all_passed = all("PASS" in r for r in results.values())
    
    if all_passed:
        print("\n🎉 All tests passed!")
    else:
        print("\n⚠️ Some tests failed. Check output above.")
    
    print("="*60)
    
    return all_passed


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
