"""
Example Usage Scripts for Advanced Tear Film Analysis
======================================================
Various usage scenarios and examples.
"""

from tear_film_advanced import TearFilmConfig, TearFilmAnalyzer
import pandas as pd


# ============================================
# EXAMPLE 1: Basic Analysis with Defaults
# ============================================
def example_basic():
    """Most simple usage with default parameters."""
    print("=" * 60)
    print("EXAMPLE 1: Basic Analysis")
    print("=" * 60)
    
    config = TearFilmConfig(
        video_path="C:/Users/Asus/Desktop/tear_fluid/assests/AYDIN_MEHMET TUNAHAN_Right_2026_07_26-14_01_16.mkv",
        output_csv="example_basic.csv",
        show_visualization=False  # Fast processing
    )
    
    analyzer = TearFilmAnalyzer(config)
    analyzer.analyze_video()
    
    # Load and print summary
    df = pd.read_csv(config.output_csv)
    print(f"\n✅ Analysis complete!")
    print(f"   Total particles: {len(df)}")
    print(f"   Mean velocity: {df['mms_velocity'].mean():.4f}")


# ============================================
# EXAMPLE 2: High Sensitivity (Catch More Particles)
# ============================================
def example_high_sensitivity():
    """More sensitive detection for subtle particles."""
    print("\n" + "=" * 60)
    print("EXAMPLE 2: High Sensitivity Mode")
    print("=" * 60)
    
    config = TearFilmConfig(
        video_path="C:/Users/Asus/Desktop/tear_fluid/assests/AYDIN_MEHMET TUNAHAN_Right_2026_07_26-14_01_16.mkv",
        
        # Lower thresholds for more sensitivity
        thresh_k=2.0,  # Lower = more particles
        min_particle_area=1,
        max_particle_area=80,
        floor_threshold=0.1,
        
        # More aggressive blink detection
        blink_z_threshold=5.0,  # Higher = less blinks detected
        
        output_csv="example_sensitive.csv",
        show_visualization=False
    )
    
    analyzer = TearFilmAnalyzer(config)
    analyzer.analyze_video()
    
    df = pd.read_csv(config.output_csv)
    print(f"\n✅ High sensitivity analysis complete!")
    print(f"   Total particles: {len(df)}")


# ============================================
# EXAMPLE 3: High Specificity (Reduce False Positives)
# ============================================
def example_high_specificity():
    """More selective detection to minimize false positives."""
    print("\n" + "=" * 60)
    print("EXAMPLE 3: High Specificity Mode")
    print("=" * 60)
    
    config = TearFilmConfig(
        video_path="C:/Users/Asus/Desktop/tear_fluid/assests/AYDIN_MEHMET TUNAHAN_Right_2026_07_26-14_01_16.mkv",
        
        # Higher thresholds for selectivity
        thresh_k=4.5,  # Higher = fewer particles
        min_particle_area=2,
        max_particle_area=40,
        floor_threshold=1.0,
        
        # Larger glare buffer
        glare_buffer_radius=40,
        
        # Stricter blink detection
        blink_z_threshold=3.0,  # Lower = more blinks detected
        
        output_csv="example_specific.csv",
        show_visualization=False
    )
    
    analyzer = TearFilmAnalyzer(config)
    analyzer.analyze_video()
    
    df = pd.read_csv(config.output_csv)
    print(f"\n✅ High specificity analysis complete!")
    print(f"   Total particles: {len(df)}")


# ============================================
# EXAMPLE 4: Batch Processing Multiple Videos
# ============================================
def example_batch_processing():
    """Process multiple videos with same parameters."""
    print("\n" + "=" * 60)
    print("EXAMPLE 4: Batch Processing")
    print("=" * 60)
    
    # List of videos to process
    videos = [
        "C:/Users/Asus/Desktop/tear_fluid/assests/AYDIN_MEHMET TUNAHAN_Right_2026_07_26-14_01_16.mkv",
        # Add more video paths here...
    ]
    
    # Common configuration
    base_config = {
        'thresh_k': 3.0,
        'glare_buffer_radius': 30,
        'show_visualization': False
    }
    
    results_summary = []
    
    for i, video_path in enumerate(videos):
        print(f"\nProcessing video {i+1}/{len(videos)}: {video_path}")
        
        config = TearFilmConfig(
            video_path=video_path,
            output_csv=f"batch_result_{i+1}.csv",
            **base_config
        )
        
        try:
            analyzer = TearFilmAnalyzer(config)
            analyzer.analyze_video()
            
            # Collect statistics
            df = pd.read_csv(config.output_csv)
            results_summary.append({
                'video': video_path,
                'total_particles': len(df),
                'unique_particles': df['particle_id'].nunique(),
                'mean_velocity': df['mms_velocity'].mean(),
                'epochs': df['epoch'].nunique()
            })
            
        except Exception as e:
            print(f"❌ Error processing {video_path}: {e}")
            results_summary.append({
                'video': video_path,
                'error': str(e)
            })
    
    # Save summary
    summary_df = pd.DataFrame(results_summary)
    summary_df.to_csv("batch_summary.csv", index=False)
    print("\n✅ Batch processing complete! Summary saved to batch_summary.csv")
    print(summary_df)


# ============================================
# EXAMPLE 5: Custom Analysis with Callback
# ============================================
def example_custom_callback():
    """Advanced: Custom processing with epoch callback."""
    print("\n" + "=" * 60)
    print("EXAMPLE 5: Custom Callback (Advanced)")
    print("=" * 60)
    
    config = TearFilmConfig(
        video_path="C:/Users/Asus/Desktop/tear_fluid/assests/AYDIN_MEHMET TUNAHAN_Right_2026_07_26-14_01_16.mkv",
        output_csv="example_callback.csv",
        show_visualization=False
    )
    
    analyzer = TearFilmAnalyzer(config)
    
    # Modify analyzer for custom processing
    original_process = analyzer._process_epochs
    
    def custom_process():
        print("\n🔧 Custom preprocessing...")
        # Add your custom logic here
        original_process()
        print("🔧 Custom postprocessing...")
    
    analyzer._process_epochs = custom_process
    analyzer.analyze_video()
    
    print("\n✅ Custom analysis complete!")


# ============================================
# EXAMPLE 6: Parameter Comparison Study
# ============================================
def example_parameter_comparison():
    """Compare results with different parameter sets."""
    print("\n" + "=" * 60)
    print("EXAMPLE 6: Parameter Comparison Study")
    print("=" * 60)
    
    video_path = "C:/Users/Asus/Desktop/tear_fluid/assests/AYDIN_MEHMET TUNAHAN_Right_2026_07_26-14_01_16.mkv"
    
    # Different parameter sets to test
    parameter_sets = [
        {'name': 'Conservative', 'thresh_k': 4.0, 'glare_buffer_radius': 40},
        {'name': 'Moderate', 'thresh_k': 3.0, 'glare_buffer_radius': 30},
        {'name': 'Aggressive', 'thresh_k': 2.0, 'glare_buffer_radius': 20},
    ]
    
    comparison_results = []
    
    for params in parameter_sets:
        print(f"\nTesting: {params['name']}")
        
        config = TearFilmConfig(
            video_path=video_path,
            thresh_k=params['thresh_k'],
            glare_buffer_radius=params['glare_buffer_radius'],
            output_csv=f"comparison_{params['name'].lower()}.csv",
            show_visualization=False
        )
        
        analyzer = TearFilmAnalyzer(config)
        analyzer.analyze_video()
        
        df = pd.read_csv(config.output_csv)
        comparison_results.append({
            'Parameter Set': params['name'],
            'thresh_k': params['thresh_k'],
            'glare_buffer': params['glare_buffer_radius'],
            'Total Detections': len(df),
            'Unique Particles': df['particle_id'].nunique(),
            'Mean Velocity': df['mms_velocity'].mean()
        })
    
    # Display comparison
    comparison_df = pd.DataFrame(comparison_results)
    print("\n" + "=" * 60)
    print("PARAMETER COMPARISON RESULTS")
    print("=" * 60)
    print(comparison_df.to_string(index=False))
    
    comparison_df.to_csv("parameter_comparison.csv", index=False)


# ============================================
# EXAMPLE 7: Export for Power Law Fitting
# ============================================
def example_export_for_powerlaw():
    """Export data in format ready for power-law fitting."""
    print("\n" + "=" * 60)
    print("EXAMPLE 7: Export for Power Law Analysis")
    print("=" * 60)
    
    # First run the analysis
    config = TearFilmConfig(
        video_path="C:/Users/Asus/Desktop/tear_fluid/assests/AYDIN_MEHMET TUNAHAN_Right_2026_07_26-14_01_16.mkv",
        output_csv="for_powerlaw.csv",
        show_visualization=False
    )
    
    analyzer = TearFilmAnalyzer(config)
    analyzer.analyze_video()
    
    # Load and prepare for power-law fitting
    df = pd.read_csv("for_powerlaw.csv")
    
    # Clean data: remove outliers (> 40 particles per time point)
    time_counts = df.groupby('time_sec').size()
    valid_times = time_counts[time_counts <= 40].index
    clean_df = df[df['time_sec'].isin(valid_times)]
    
    # Calculate mean velocity per time point
    mean_velocity = clean_df.groupby('time_sec')['mms_velocity'].mean().reset_index()
    mean_velocity.columns = ['time_sec', 'mean_mms_velocity']
    
    # Save for veri_analizi.py
    mean_velocity.to_csv("powerlaw_ready.csv", index=False)
    
    print(f"\n✅ Data prepared for power-law fitting!")
    print(f"   Original records: {len(df)}")
    print(f"   Clean records: {len(clean_df)}")
    print(f"   Time points: {len(mean_velocity)}")
    print(f"   Saved to: powerlaw_ready.csv")


# ============================================
# MAIN: Run Examples
# ============================================
if __name__ == "__main__":
    import sys
    
    examples = {
        '1': ('Basic Analysis', example_basic),
        '2': ('High Sensitivity', example_high_sensitivity),
        '3': ('High Specificity', example_high_specificity),
        '4': ('Batch Processing', example_batch_processing),
        '5': ('Custom Callback', example_custom_callback),
        '6': ('Parameter Comparison', example_parameter_comparison),
        '7': ('Export for Power Law', example_export_for_powerlaw),
    }
    
    print("\n" + "=" * 60)
    print("TEAR FILM ANALYSIS - EXAMPLE USAGE")
    print("=" * 60)
    print("\nAvailable examples:")
    for key, (name, _) in examples.items():
        print(f"  {key}. {name}")
    print("  A. Run all examples")
    print("  Q. Quit")
    
    if len(sys.argv) > 1:
        choice = sys.argv[1].upper()
    else:
        choice = input("\nSelect example (1-7, A, Q): ").strip().upper()
    
    if choice == 'Q':
        print("Exiting...")
        sys.exit(0)
    elif choice == 'A':
        for key, (name, func) in examples.items():
            try:
                func()
            except Exception as e:
                print(f"❌ Error in {name}: {e}")
    elif choice in examples:
        _, func = examples[choice]
        func()
    else:
        print("Invalid choice!")
        sys.exit(1)
    
    print("\n" + "=" * 60)
    print("ALL EXAMPLES COMPLETE")
    print("=" * 60)
