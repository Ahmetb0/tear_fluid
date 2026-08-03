"""
Streamlit UI for Tear Film Analysis Parameter Titration
========================================================
Interactive parameter tuning interface for tear film analysis.

Usage:
    streamlit run tear_film_ui.py
"""

import streamlit as st
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from tear_film_advanced import (
    TearFilmConfig, TearFilmAnalyzer, 
    ParticleDetector, GlareExcluder, BlinkDetector, ValidationOptimizer,
    compute_power_law_decay
)

# Try to import streamlit-image-coordinates (optional for better UX)
try:
    from streamlit_image_coordinates import streamlit_image_coordinates
    HAS_IMAGE_COORDINATES = True
except ImportError:
    HAS_IMAGE_COORDINATES = False
    st.warning(
        "⚠️ streamlit-image-coordinates not installed. "
        "Install with: `pip install streamlit-image-coordinates` for better ground truth annotation."
    )

st.set_page_config(
    page_title="Tear Film Analysis UI",
    page_icon="💧",
    layout="wide"
)

# Session state initialization
if 'video_loaded' not in st.session_state:
    st.session_state.video_loaded = False
if 'sample_frame' not in st.session_state:
    st.session_state.sample_frame = None
if 'config' not in st.session_state:
    st.session_state.config = TearFilmConfig()
if 'epochs' not in st.session_state:
    st.session_state.epochs = None
if 'safe_frames' not in st.session_state:
    st.session_state.safe_frames = []
if 'blink_ranges' not in st.session_state:
    st.session_state.blink_ranges = []
if 'num_frames' not in st.session_state:
    st.session_state.num_frames = 0


def load_sample_frame(video_path: str, frame_idx: int = 30):
    """Load a sample frame from video for titration."""
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()
    
    if ret:
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), \
               cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return None, None


def get_frame_epoch_info(frame_idx: int, epochs, safe_frames):
    """
    Get epoch information for a given frame.
    
    Returns:
        dict: Information about frame safety and epoch
    """
    if frame_idx not in safe_frames:
        return {
            'is_safe': False,
            'epoch_idx': None,
            'epoch_progress': None,
            'message': '⚠️ UNSAFE: This frame is in a blink or transition period!'
        }
    
    # Find which epoch this frame belongs to
    for i, epoch in enumerate(epochs):
        if epoch.start_frame <= frame_idx < epoch.end_frame:
            progress = (frame_idx - epoch.start_frame) / (epoch.end_frame - epoch.start_frame) * 100
            return {
                'is_safe': True,
                'epoch_idx': i,
                'epoch_progress': progress,
                'epoch_length': epoch.end_frame - epoch.start_frame,
                'message': f'✅ Safe Frame | Epoch {i+1}/{len(epochs)} | {progress:.0f}% through epoch'
            }
    
    return {
        'is_safe': False,
        'epoch_idx': None,
        'epoch_progress': None,
        'message': '⚠️ Frame outside valid epochs'
    }


def preprocess_video_epochs(video_path: str, config: TearFilmConfig):
    """
    Preprocess video to detect blinks and segment epochs.
    This runs once when video is loaded and caches results in session state.
    
    Returns:
        tuple: (epochs, safe_frames, blink_ranges, num_frames)
    """
    import time
    start = time.time()
    
    # Get video info
    cap = cv2.VideoCapture(video_path)
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = config.fps
    cap.release()
    
    # Update config with video FPS
    config.fps = fps
    
    # Detect blinks
    from tear_film_advanced import BlinkDetector, EpochSegmenter
    blink_detector = BlinkDetector(config)
    blink_ranges = blink_detector.detect_blinks(video_path)
    
    # Segment epochs
    epoch_segmenter = EpochSegmenter(config)
    epochs = epoch_segmenter.segment_epochs(num_frames, blink_ranges)
    
    # Create safe frames list (flatten all epoch frames)
    safe_frames = []
    for epoch in epochs:
        safe_frames.extend(range(epoch.start_frame, epoch.end_frame))
    
    elapsed = time.time() - start
    
    return epochs, safe_frames, blink_ranges, num_frames, elapsed


def visualize_detection(gray_frame, config, glare_excluder, particle_detector):
    """Visualize particle detection on sample frame."""
    # Try to detect fixation lights
    fixation_detected = glare_excluder.detect_fixation_lights(gray_frame)
    
    if not fixation_detected:
        st.warning("⚠️ Fixation lights not detected - using full frame (no glare exclusion)")
        glare_mask = np.ones(gray_frame.shape, dtype=np.uint8)
    else:
        glare_mask = glare_excluder.create_glare_mask(gray_frame.shape)
    
    particles = particle_detector.detect_particles(gray_frame, glare_mask)
    vis_frame = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2RGB)
    
    if glare_excluder.has_valid_fixation_lights:
        cv2.circle(vis_frame, glare_excluder.superior_light, 
                   config.glare_buffer_radius, (255, 0, 0), 2)
        cv2.circle(vis_frame, glare_excluder.inferior_light, 
                   config.glare_buffer_radius, (255, 0, 0), 2)
        cv2.line(vis_frame, glare_excluder.superior_light,
                 glare_excluder.inferior_light, (255, 0, 0), 2)
    
    # Draw particles
    for p in particles:
        x, y = int(p['x']), int(p['y'])
        area = p['area']
        cv2.circle(vis_frame, (x, y), 3, (0, 255, 0), -1)
        cv2.putText(vis_frame, f"{area:.0f}", (x+5, y-5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 255), 1)
    
    # Add info overlay
    info_text = f"Particles Detected: {len(particles)}"
    cv2.putText(vis_frame, info_text, (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2)
    
    return vis_frame, len(particles), particles


# ===== SIDEBAR: Configuration =====
st.sidebar.title("💧 Tear Film Analysis")
st.sidebar.markdown("---")

# Video selection
st.sidebar.header("1️⃣ Video Selection")
video_path = st.sidebar.text_input(
    "Video Path",
    value="C:/Users/Asus/Desktop/tear_fluid/assests/AYDIN_MEHMET TUNAHAN_Right_2026_07_26-14_01_16.mkv"
)

if st.sidebar.button("Load Video"):
    if Path(video_path).exists():
        with st.sidebar.spinner("Loading video and detecting epochs..."):
            # Load sample frame
            color_frame, gray_frame = load_sample_frame(video_path, frame_idx=30)
            
            if color_frame is not None:
                st.session_state.sample_frame = (color_frame, gray_frame)
                st.session_state.config.video_path = video_path
                
                # Preprocess: Detect blinks and segment epochs
                epochs, safe_frames, blink_ranges, num_frames, elapsed = preprocess_video_epochs(
                    video_path, 
                    st.session_state.config
                )
                
                # Cache results in session state
                st.session_state.epochs = epochs
                st.session_state.safe_frames = safe_frames
                st.session_state.blink_ranges = blink_ranges
                st.session_state.num_frames = num_frames
                st.session_state.video_loaded = True
                
                # Show results
                st.sidebar.success(f"✅ Video loaded! ({elapsed:.1f}s)")
                st.sidebar.info(
                    f"📊 Analysis:\n"
                    f"- Total frames: {num_frames}\n"
                    f"- Blink events: {len(blink_ranges)}\n"
                    f"- Valid epochs: {len(epochs)}\n"
                    f"- Safe frames: {len(safe_frames)} ({len(safe_frames)/num_frames*100:.1f}%)"
                )
            else:
                st.sidebar.error("❌ Failed to load frame")
    else:
        st.sidebar.error("❌ Video file not found")

st.sidebar.markdown("---")

# ===== MAIN AREA: Tabs =====
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🎛️ Titration", 
    "🔍 Blink Detection", 
    "🚀 Run Analysis",
    "📊 Results",
    "🔬 Optimization"
])

# ===== TAB 1: TITRATION =====
with tab1:
    st.header("🎛️ Parameter Titration")
    st.markdown("Adjust parameters and see real-time detection results on a sample frame.")
    
    if not st.session_state.video_loaded:
        st.warning("⚠️ Please load a video first from the sidebar!")
    elif len(st.session_state.safe_frames) == 0:
        st.error("❌ No valid epochs found in this video!")
        st.warning(
            "**No analyzable open-eye intervals detected.**\n\n"
            "This could mean:\n"
            "- Video has too many blinks\n"
            "- Blink detection is too sensitive (increase `blink_z_threshold`)\n"
            "- Video is too short\n\n"
            "Please try adjusting blink detection parameters or use a different video."
        )
    else:
        # Safe Frame Selector
        st.markdown("### 🎯 Frame Selection (Safe Frames Only)")
        
        # Get middle frame from safe frames as default
        default_frame_idx = st.session_state.safe_frames[len(st.session_state.safe_frames)//2]
        
        # Use select_slider with only safe frames
        selected_frame = st.select_slider(
            "Select Frame for Analysis",
            options=st.session_state.safe_frames,
            value=default_frame_idx,
            help="Only frames from valid open-eye epochs are shown"
        )
        
        # Show frame info
        frame_info = get_frame_epoch_info(
            selected_frame, 
            st.session_state.epochs,
            st.session_state.safe_frames
        )
        
        if frame_info['is_safe']:
            st.success(frame_info['message'])
        else:
            st.error(frame_info['message'])
        
        # Load the selected frame
        color_frame, gray_frame = load_sample_frame(
            st.session_state.config.video_path, 
            selected_frame
        )
        st.session_state.sample_frame = (color_frame, gray_frame)
        
        st.markdown("---")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Particle Detection Parameters")
            cfg = st.session_state.config
            
            thresh_k = st.slider(
                "Adaptive Threshold Multiplier (thresh_k)",
                min_value=1.0, max_value=8.0,
                value=float(cfg.thresh_k), step=0.1,
                help="Higher = more selective (fewer particles)"
            )
            
            min_area = st.slider(
                "Minimum Particle Area (pixels²)",
                min_value=1, max_value=10,
                value=int(cfg.min_particle_area), step=1
            )
            
            max_area = st.slider(
                "Maximum Particle Area (pixels²)",
                min_value=20, max_value=200,
                value=int(cfg.max_particle_area), step=5
            )
            
            glare_buffer = st.slider(
                "Glare Buffer Radius (pixels)",
                min_value=10, max_value=80,
                value=int(cfg.glare_buffer_radius), step=5,
                help="Exclusion zone around fixation lights"
            )
            
            st.subheader("Bandpass Filter")
            
            sigma_small = st.slider(
                "Small Sigma",
                min_value=0.5, max_value=3.0,
                value=float(cfg.sigma_small), step=0.1
            )
            
            sigma_large = st.slider(
                "Large Sigma",
                min_value=3.0, max_value=15.0,
                value=float(cfg.sigma_large), step=0.5
            )
            
            floor_thresh = st.slider(
                "Floor Threshold",
                min_value=0.0, max_value=5.0,
                value=float(cfg.floor_threshold), step=0.1
            )
            
            # Keep session config in sync with sliders (BUG-S1/S2)
            cfg.thresh_k = thresh_k
            cfg.min_particle_area = min_area
            cfg.max_particle_area = max_area
            cfg.glare_buffer_radius = glare_buffer
            cfg.sigma_small = sigma_small
            cfg.sigma_large = sigma_large
            cfg.floor_threshold = floor_thresh
            
            if st.button("🔄 Apply & Visualize", type="primary"):
                # Config already synced above; run preview only
                _, gray_frame = st.session_state.sample_frame
                glare_excluder = GlareExcluder(st.session_state.config)
                particle_detector = ParticleDetector(st.session_state.config)
                
                result = visualize_detection(
                    gray_frame, 
                    st.session_state.config, 
                    glare_excluder, 
                    particle_detector
                )
                
                if result:
                    vis_frame, num_particles, particles = result
                    st.session_state.vis_result = (vis_frame, num_particles, particles)
        
        with col2:
            st.subheader("Detection Preview")
            
            if 'vis_result' in st.session_state:
                vis_frame, num_particles, particles = st.session_state.vis_result
                
                st.image(vis_frame, caption=f"Detected {num_particles} particles", 
                        use_container_width=True)
                
                # Statistics
                st.metric("Total Particles", num_particles)
                
                if particles:
                    areas = [p['area'] for p in particles]
                    col_a, col_b, col_c = st.columns(3)
                    with col_a:
                        st.metric("Min Area", f"{min(areas):.1f}")
                    with col_b:
                        st.metric("Mean Area", f"{np.mean(areas):.1f}")
                    with col_c:
                        st.metric("Max Area", f"{max(areas):.1f}")
                    
                    # Histogram
                    fig, ax = plt.subplots(figsize=(6, 3))
                    ax.hist(areas, bins=20, color='skyblue', edgecolor='black')
                    ax.set_xlabel('Particle Area (pixels²)')
                    ax.set_ylabel('Count')
                    ax.set_title('Particle Size Distribution')
                    st.pyplot(fig)
            else:
                st.info("Click 'Apply & Visualize' to see detection results")

# ===== TAB 2: BLINK DETECTION =====
with tab2:
    st.header("🔍 Blink Detection Preview")
    st.markdown("Analyze brightness signal and detect blinks before running full analysis.")
    
    if not st.session_state.video_loaded:
        st.warning("⚠️ Please load a video first from the sidebar!")
    else:
        # Show cached results if available
        if st.session_state.epochs is not None:
            st.success("✅ Blink detection already performed (cached from video load)")
            
            col_summary1, col_summary2, col_summary3, col_summary4 = st.columns(4)
            with col_summary1:
                st.metric("Total Frames", st.session_state.num_frames)
            with col_summary2:
                st.metric("Blink Events", len(st.session_state.blink_ranges))
            with col_summary3:
                st.metric("Valid Epochs", len(st.session_state.epochs))
            with col_summary4:
                safe_pct = len(st.session_state.safe_frames) / st.session_state.num_frames * 100
                st.metric("Safe Frames", f"{safe_pct:.1f}%")
            
            st.info("💡 These results are used for safe frame selection in Titration and Optimization tabs.")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("Blink Detection Parameters")
            
            z_threshold = st.slider(
                "Z-Score Threshold",
                min_value=2.0, max_value=8.0, value=4.0, step=0.5,
                help="Lower = more sensitive"
            )
            
            pad_frames = st.slider(
                "Padding Frames",
                min_value=0, max_value=5,
                value=int(st.session_state.config.blink_pad_frames), step=1
            )
            
            min_epoch = st.slider(
                "Minimum Epoch Length",
                min_value=3, max_value=20, value=5, step=1
            )
            
            button_text = "🔄 Re-analyze with New Parameters" if st.session_state.epochs else "🔍 Analyze Blinks"
            
            if st.button(button_text, type="primary"):
                st.session_state.config.blink_z_threshold = z_threshold
                st.session_state.config.blink_pad_frames = pad_frames
                st.session_state.config.min_epoch_length = min_epoch
                
                with st.spinner("Re-analyzing brightness signal and epochs..."):
                    # Re-run full epoch preprocessing
                    epochs, safe_frames, blink_ranges, num_frames, elapsed = preprocess_video_epochs(
                        video_path,
                        st.session_state.config
                    )
                    
                    # Update cache
                    st.session_state.epochs = epochs
                    st.session_state.safe_frames = safe_frames
                    st.session_state.blink_ranges = blink_ranges
                    st.session_state.num_frames = num_frames
                    
                    # Also keep signals for visualization
                    blink_detector = BlinkDetector(st.session_state.config)
                    st.session_state.blink_results = {
                        'ranges': blink_ranges,
                        'signals': blink_detector.frame_signals
                    }
                    
                    st.success(f"✅ Re-analysis complete! ({elapsed:.1f}s)")
                    st.info(
                        f"Updated cache:\n"
                        f"- Blink events: {len(blink_ranges)}\n"
                        f"- Valid epochs: {len(epochs)}\n"
                        f"- Safe frames: {len(safe_frames)} ({len(safe_frames)/num_frames*100:.1f}%)"
                    )
                    # Streamlit will auto-rerun on state change
        
        with col2:
            if 'blink_results' in st.session_state:
                results = st.session_state.blink_results
                blink_ranges = results['ranges']
                signals = results['signals']
                
                st.success(f"✅ Detected {len(blink_ranges)} blink events")
                
                # Plot brightness signal
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
                
                frames = np.arange(len(signals))
                mean_values = [s.mean for s in signals]
                bright_counts = [s.bright_count for s in signals]
                
                # Mean brightness
                ax1.plot(frames, mean_values, linewidth=1, color='blue', alpha=0.7)
                ax1.set_ylabel('Mean Brightness')
                ax1.set_title('Frame Brightness Signal')
                ax1.grid(True, alpha=0.3)
                
                # Bright pixel count
                ax2.plot(frames, bright_counts, linewidth=1, color='orange', alpha=0.7)
                ax2.set_ylabel('Bright Pixel Count')
                ax2.set_xlabel('Frame Number')
                ax2.grid(True, alpha=0.3)
                
                # Highlight blink ranges
                for start, end in blink_ranges:
                    ax1.axvspan(start, end, alpha=0.3, color='red')
                    ax2.axvspan(start, end, alpha=0.3, color='red')
                
                plt.tight_layout()
                st.pyplot(fig)
                
                # Epoch summary
                st.subheader("Epoch Summary")
                cap = cv2.VideoCapture(video_path)
                num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                cap.release()
                
                from tear_film_advanced import EpochSegmenter
                segmenter = EpochSegmenter(st.session_state.config)
                epochs = segmenter.segment_epochs(num_frames, blink_ranges)
                
                epoch_data = []
                for i, epoch in enumerate(epochs):
                    epoch_data.append({
                        'Epoch': i+1,
                        'Start Frame': epoch.start_frame,
                        'End Frame': epoch.end_frame,
                        'Length': epoch.length(),
                        'Duration (s)': f"{epoch.length() / st.session_state.config.fps:.2f}"
                    })
                
                st.dataframe(pd.DataFrame(epoch_data), use_container_width=True)

# ===== TAB 3: RUN ANALYSIS =====
with tab3:
    st.header("🚀 Run Full Analysis")
    st.markdown("Run complete tear film analysis with configured parameters.")
    
    if not st.session_state.video_loaded:
        st.warning("⚠️ Please load a video first from the sidebar!")
    else:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Output Configuration")
            
            output_csv = st.text_input(
                "Output CSV Filename",
                value="tear_film_analysis_ui.csv"
            )
            
            show_viz = st.checkbox("Show Real-time Visualization", value=False,
                                  help="Warning: May slow down processing")
            
            st.subheader("Current Parameters")
            param_summary = f"""
            **Blink Detection:**
            - Z-threshold: {st.session_state.config.blink_z_threshold}
            - Pad frames: {st.session_state.config.blink_pad_frames}
            - Min epoch: {st.session_state.config.min_epoch_length}
            
            **Particle Detection:**
            - thresh_k: {st.session_state.config.thresh_k}
            - Min area: {st.session_state.config.min_particle_area}
            - Max area: {st.session_state.config.max_particle_area}
            - Glare buffer: {st.session_state.config.glare_buffer_radius}
            """
            st.markdown(param_summary)
            
            if st.button("▶️ Start Analysis", type="primary"):
                st.session_state.config.output_csv = output_csv
                st.session_state.config.show_visualization = show_viz
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                try:
                    status_text.text("Initializing analyzer...")
                    analyzer = TearFilmAnalyzer(st.session_state.config)
                    
                    status_text.text("Running analysis... (see terminal for details)")
                    analyzer.analyze_video()
                    
                    progress_bar.progress(100)
                    status_text.text("✅ Analysis complete!")
                    
                    st.session_state.analysis_complete = True
                    st.session_state.output_file = output_csv
                    
                except Exception as e:
                    st.error(f"❌ Error during analysis: {str(e)}")
        
        with col2:
            if 'analysis_complete' in st.session_state and st.session_state.analysis_complete:
                st.success("✅ Analysis completed successfully!")
                
                # Load and display results
                df = pd.read_csv(st.session_state.output_file)
                
                st.metric("Total Records", len(df))
                
                col_a, col_b, col_c = st.columns(3)
                with col_a:
                    st.metric("Unique Particles", df['particle_id'].nunique())
                with col_b:
                    st.metric("Epochs", df['epoch'].nunique())
                with col_c:
                    st.metric("Mean Velocity", f"{df['mms_velocity'].mean():.4f}")
                
                st.download_button(
                    label="📥 Download Results CSV",
                    data=df.to_csv(index=False).encode('utf-8'),
                    file_name=st.session_state.output_file,
                    mime='text/csv'
                )

# ===== TAB 4: RESULTS =====
with tab4:
    st.header("📊 Results Visualization")
    
    uploaded_file = st.file_uploader("Upload analysis results CSV", type=['csv'])
    
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        
        st.subheader("Summary Statistics")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Records", len(df))
        with col2:
            st.metric("Particles", df['particle_id'].nunique())
        with col3:
            st.metric("Epochs", df['epoch'].nunique())
        with col4:
            st.metric("Duration", f"{df['time_sec'].max():.1f}s")
        
        # Power-Law Decay Curve (Time Since Blink)
        st.subheader("Power-Law Decay Curve (Medical Literature Model)")
        
        # Check if time_since_blink_s column exists
        if 'time_since_blink_s' in df.columns:
            bin_size = st.slider("Binning Interval (seconds)", 
                                min_value=0.05, max_value=0.5, value=0.15, step=0.05,
                                help="Group data into time intervals for smoothing")
            
            if 'include_in_power_law_fit' in df.columns:
                n_excluded = (~df['include_in_power_law_fit'].astype(bool)).sum()
                if n_excluded > 0:
                    st.info(
                        f"ℹ️ Power-law fit excludes **{n_excluded}** rows from epochs "
                        f"starting at frame 0 (non-post-blink). All data remain in CSV."
                    )
            
            power_law_result = compute_power_law_decay(df, bin_size=bin_size)
            
            if power_law_result:
                # Create figure with power-law curve
                fig, ax = plt.subplots(figsize=(12, 5))
                
                # Plot binned data as scatter points
                ax.scatter(power_law_result['binned_time'], 
                          power_law_result['binned_velocity'],
                          s=80, alpha=0.6, color='steelblue', 
                          label='Binned Median Velocity', zorder=3)
                
                # Plot fitted power-law curve
                ax.plot(power_law_result['binned_time'], 
                       power_law_result['fitted_curve'],
                       linewidth=3, color='crimson', 
                       label=f"Fitted: {power_law_result['equation']}", zorder=4)
                
                # Add R² to plot
                ax.text(0.02, 0.98, 
                       f"R² = {power_law_result['r_squared']:.4f}\n"
                       f"α = {power_law_result['alpha']:.3f}\n"
                       f"β = {power_law_result['beta']:.3f}",
                       transform=ax.transAxes, fontsize=11,
                       verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
                
                ax.set_xlabel('Time Since Blink (seconds)', fontsize=12)
                ax.set_ylabel('Velocity (mm/s)', fontsize=12)
                ax.set_title('Tear Film Velocity Decay: v = α × t^(-β)', fontsize=13, fontweight='bold')
                ax.grid(True, alpha=0.3, linestyle='--')
                ax.legend(loc='upper right', fontsize=10)
                
                st.pyplot(fig)
                
                # Display statistics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Alpha (α)", f"{power_law_result['alpha']:.3f}")
                with col2:
                    st.metric("Beta (β)", f"{power_law_result['beta']:.3f}")
                with col3:
                    st.metric("R² Score", f"{power_law_result['r_squared']:.4f}")
                with col4:
                    st.metric("Bins Used", power_law_result['num_bins'])
                
                st.info(
                    "📊 **Power-Law Model Interpretation:**\n"
                    "- **α (alpha)**: Initial velocity coefficient - higher values indicate faster initial tear film spread\n"
                    "- **β (beta)**: Decay exponent - typical range 0.3-0.8 for healthy tear film\n"
                    "- **R²**: Goodness of fit - values >0.8 indicate excellent model fit"
                )
            else:
                st.warning("⚠️ Unable to compute power-law curve. Check data quality.")
        else:
            st.warning("⚠️ time_since_blink_s column not found. Please re-run analysis with updated code.")
        
        # Legacy view (old absolute time plot)
        with st.expander("📉 Legacy View: Absolute Time (for comparison only)"):
            st.caption("⚠️ This view shows artifacts from concatenating different epochs")
            velocity_time = df.groupby('time_sec')['mms_velocity'].mean().reset_index()
            
            fig, ax = plt.subplots(figsize=(12, 3))
            ax.plot(velocity_time['time_sec'], velocity_time['mms_velocity'], 
                    linewidth=1.5, color='gray', alpha=0.6)
            ax.set_xlabel('Absolute Time (seconds)')
            ax.set_ylabel('Mean MMS Velocity')
            ax.set_title('Mean Velocity Over Absolute Time (Not Recommended)')
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
        
        # Epoch comparison
        st.subheader("Velocity by Epoch")
        fig, ax = plt.subplots(figsize=(10, 5))
        df.boxplot(column='mms_velocity', by='epoch', ax=ax)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('MMS Velocity')
        ax.set_title('Velocity Distribution by Epoch')
        plt.suptitle('')
        st.pyplot(fig)
        
        # Data table
        st.subheader("Raw Data (first 100 rows)")
        st.dataframe(df.head(100), use_container_width=True)

# ===== TAB 5: OPTIMIZATION =====
with tab5:
    st.header("🔬 Parameter Optimization (Grid Search)")
    st.markdown("""
    Automatically find optimal detection parameters using ground truth validation.
    
    **Workflow:**
    1. Select a safe frame (from valid epochs)
    2. Manually annotate true particle positions (ground truth)
    3. Run grid search to find parameters with best F1 score
    4. Apply optimized settings to your analysis
    """)
    
    if not st.session_state.video_loaded:
        st.warning("⚠️ Please load a video first from the sidebar!")
    elif len(st.session_state.safe_frames) == 0:
        st.error("❌ No valid epochs found in this video!")
        st.warning(
            "**No analyzable open-eye intervals detected.**\n\n"
            "Cannot perform optimization without valid frames. "
            "Please adjust blink detection parameters or use a different video."
        )
    else:
        # Safe Frame Selector for Optimization
        st.markdown("### 🎯 Frame Selection (Safe Frames Only)")
        st.info(
            "💡 **Important:** Select a frame with typical particle density "
            "from a stable open-eye period for best optimization results."
        )
        
        # Get middle frame from safe frames as default
        default_opt_frame = st.session_state.safe_frames[len(st.session_state.safe_frames)//2]
        
        # Use select_slider with only safe frames
        selected_opt_frame = st.select_slider(
            "Select Frame for Optimization",
            options=st.session_state.safe_frames,
            value=default_opt_frame,
            help="Only frames from valid open-eye epochs",
            key="optimization_frame_selector"
        )
        
        # Show frame info
        opt_frame_info = get_frame_epoch_info(
            selected_opt_frame, 
            st.session_state.epochs,
            st.session_state.safe_frames
        )
        
        col_info1, col_info2 = st.columns([3, 1])
        with col_info1:
            if opt_frame_info['is_safe']:
                st.success(opt_frame_info['message'])
            else:
                st.error(opt_frame_info['message'])
        with col_info2:
            st.metric("Frame #", selected_opt_frame)
        
        # Load the selected frame
        color_frame, gray_frame = load_sample_frame(
            st.session_state.config.video_path, 
            selected_opt_frame
        )
        st.session_state.sample_frame = (color_frame, gray_frame)
        
        st.markdown("---")
        
        # FULL WIDTH LAYOUT - No columns for better image display
        st.subheader("1️⃣ Ground Truth Annotation")
        
        st.info("""
        📝 **Manual Annotation Instructions:**
        - Click on the image below to mark true particle positions
        - Each click adds a point to ground truth
        - You can also upload a CSV file with (x, y) coordinates
        """)
        
        # Display sample frame for annotation
        if st.session_state.sample_frame:
            color_frame, gray_frame = st.session_state.sample_frame
            
            # Initialize ground truth in session state
            if 'ground_truth' not in st.session_state:
                st.session_state.ground_truth = []
            
            # Initialize last click tracker to prevent duplicate processing
            if 'last_click_coords' not in st.session_state:
                st.session_state.last_click_coords = None
            
            # Store original frame dimensions
            original_height, original_width = color_frame.shape[:2]
            
            # Draw existing ground truth points on frame
            annotated_frame = color_frame.copy()
            for idx, (gt_x, gt_y) in enumerate(st.session_state.ground_truth):
                # Draw circle at each ground truth point
                cv2.circle(annotated_frame, (int(gt_x), int(gt_y)), 
                          radius=8, color=(0, 255, 0), thickness=2)
                # Draw point number
                cv2.putText(annotated_frame, f"{idx+1}", 
                           (int(gt_x)+12, int(gt_y)-12),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Clickable image for ground truth annotation
            st.markdown("**🖱️ Click on image to mark particles:**")
            st.info(f"📐 Image: {original_width} × {original_height} px | Click directly on particles to annotate")
            
            if HAS_IMAGE_COORDINATES:
                # Convert BGR to RGB
                annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                
                # CRITICAL: NO width/height parameters
                # Let it be fully responsive to container
                clicked_point = streamlit_image_coordinates(
                    annotated_frame_rgb,  # Use numpy array directly
                    key="ground_truth_image"
                )
                
                # Debug info
                if clicked_point is not None:
                    st.caption(f"🎯 Click: ({clicked_point['x']:.0f}, {clicked_point['y']:.0f}) | Image dimensions: {original_width}×{original_height}")
                
                # Process click immediately
                if clicked_point is not None:
                    x_coord = clicked_point["x"]
                    y_coord = clicked_point["y"]
                    
                    # Create unique identifier for this click
                    current_click = (x_coord, y_coord)
                    
                    # Check if this is truly a NEW click
                    if current_click != st.session_state.last_click_coords:
                        # Validate bounds
                        if 0 <= x_coord < original_width and 0 <= y_coord < original_height:
                            # Check for nearby existing points
                            is_duplicate = any(
                                abs(x - x_coord) < 5 and abs(y - y_coord) < 5 
                                for x, y in st.session_state.ground_truth
                            )
                            
                            if not is_duplicate:
                                # Add to ground truth
                                st.session_state.ground_truth.append((x_coord, y_coord))
                                # Update last click tracker
                                st.session_state.last_click_coords = current_click
                                st.success(f"✅ Added point #{len(st.session_state.ground_truth)}: ({x_coord:.0f}, {y_coord:.0f})")
                                # Force rerun to show the new marker immediately
                                st.rerun()
                            else:
                                st.info(f"ℹ️ Point already exists near ({x_coord:.0f}, {y_coord:.0f})")
                        else:
                            st.warning(f"⚠️ Click outside image bounds")
            else:
                # Fallback: display image without click functionality
                fallback_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                st.image(fallback_rgb, 
                        caption="Install streamlit-image-coordinates for clickable annotation", 
                        use_container_width=True)
                st.info("💡 Install streamlit-image-coordinates: `pip install streamlit-image-coordinates`")
            
            # Ground Truth Management
            st.markdown("---")
            
            # Display info
            info_col1, info_col2 = st.columns(2)
            with info_col1:
                st.metric("📍 Annotated Points", len(st.session_state.ground_truth))
            with info_col2:
                st.metric("📏 Image Size", f"{original_width} × {original_height} px")
            
            # Management buttons (wider layout)
            btn_col1, btn_col2 = st.columns(2)
            with btn_col1:
                if st.button("↩️ Undo Last Point", 
                            disabled=len(st.session_state.ground_truth)==0,
                            use_container_width=True,
                            key="undo_btn"):
                    if st.session_state.ground_truth:
                        removed = st.session_state.ground_truth.pop()
                        st.session_state.last_click_coords = None  # Reset click tracker
                        # Force rerun to update the image
                        st.rerun()
            
            with btn_col2:
                if st.button("🗑️ Clear All Points", 
                            disabled=len(st.session_state.ground_truth)==0,
                            use_container_width=True,
                            key="clear_btn"):
                    st.session_state.ground_truth = []
                    st.session_state.last_click_coords = None  # Reset click tracker
                    # Force rerun to update the image
                    st.rerun()
            
            # CSV upload option
            st.markdown("---")
            st.markdown("**Alternative: Upload CSV File**")
            uploaded_gt = st.file_uploader(
                "Upload ground truth CSV (must have 'x' and 'y' columns)", 
                type=['csv'],
                key="gt_csv_upload"
            )
            if uploaded_gt:
                gt_df = pd.read_csv(uploaded_gt)
                if 'x' in gt_df.columns and 'y' in gt_df.columns:
                    st.session_state.ground_truth = list(zip(gt_df['x'], gt_df['y']))
                    st.session_state.last_click_coords = None  # Reset click tracker
                    st.rerun()  # Force rerun to show uploaded points
                else:
                    st.error("❌ CSV must have 'x' and 'y' columns")
            
            # Show annotated points list
            if st.session_state.ground_truth:
                st.markdown("---")
                with st.expander(f"📋 View All {len(st.session_state.ground_truth)} Annotated Points", expanded=False):
                    points_df = pd.DataFrame(
                        st.session_state.ground_truth,
                        columns=['x', 'y']
                    )
                    points_df.index += 1  # Start from 1
                    points_df.index.name = 'Point #'
                    st.dataframe(points_df, use_container_width=True)
                
                # Visualize ground truth overlay
                st.markdown("---")
                st.markdown("**Ground Truth Overlay (Preview):**")
                vis_gt = color_frame.copy()
                for x, y in st.session_state.ground_truth:
                    cv2.circle(vis_gt, (int(x), int(y)), 5, (255, 0, 0), 2)
                    cv2.circle(vis_gt, (int(x), int(y)), 2, (255, 255, 0), -1)
                vis_gt_rgb = cv2.cvtColor(vis_gt, cv2.COLOR_BGR2RGB)
                st.image(vis_gt_rgb, caption="Ground Truth Overlay", use_container_width=True)
            else:
                st.info("No ground truth points added yet.")
        
        # ===== GRID SEARCH PARAMETERS (MOVED BELOW IMAGE) =====
        st.markdown("---")
        st.subheader("2️⃣ Grid Search Optimization")
        
        if len(st.session_state.get('ground_truth', [])) == 0:
            st.warning("⚠️ Please annotate ground truth points above first!")
        else:
            st.success(f"✅ Ground truth ready: {len(st.session_state.ground_truth)} particles")
            
            # Optimization parameters
            st.markdown("---")
            st.subheader("⚙️ Grid Search Parameters")
            
            # Two-column layout for parameter ranges
            st.markdown("#### Parameter Ranges")
            col_left, col_right = st.columns([1, 1], gap="large")
            
            with col_left:
                st.markdown("##### thresh_k")
                thresh_k_min = st.slider("Minimum", 1.0, 5.0, 1.5, 0.5, key="tk_min")
                thresh_k_max = st.slider("Maximum", 3.0, 10.0, 8.0, 0.5, key="tk_max")
                thresh_k_step = st.slider("Step Size", 0.1, 1.0, 0.5, 0.1, key="tk_step")
            
            with col_right:
                st.markdown("##### floor_threshold")
                floor_min = st.slider("Minimum", 0.0, 1.0, 0.0, 0.1, key="fl_min")
                floor_max = st.slider("Maximum", 0.5, 3.0, 2.0, 0.25, key="fl_max")
                floor_step = st.slider("Step Size", 0.1, 0.5, 0.25, 0.05, key="fl_step")
            
            st.markdown("---")
            match_tolerance = st.slider("🎯 Matching Tolerance (pixels)", 
                                       1.0, 20.0, 5.0, 0.5,
                                       help="Maximum distance between detected and ground truth particles for a match")
            
            num_combinations = len(np.arange(thresh_k_min, thresh_k_max, thresh_k_step)) * \
                             len(np.arange(floor_min, floor_max, floor_step))
            
            st.info(f"📊 Will test **{num_combinations}** parameter combinations (estimated time: ~{num_combinations * 0.5:.0f} seconds)")
            
            st.markdown("---")
            if st.button("🚀 Start Grid Search Optimization", 
                        type="primary", 
                        use_container_width=True,
                        key="start_opt_btn"):
                _, gray_frame = st.session_state.sample_frame
                
                # Setup validation optimizer
                optimizer_config = TearFilmConfig(**st.session_state.config.__dict__)
                optimizer_config.validation_match_tolerance = match_tolerance
                
                optimizer = ValidationOptimizer(optimizer_config)
                optimizer.set_ground_truth(st.session_state.ground_truth)
                
                # Prepare frame and glare mask
                glare_excluder = GlareExcluder(optimizer_config)
                fixation_detected = glare_excluder.detect_fixation_lights(gray_frame)
                
                if not fixation_detected:
                    st.warning("⚠️ Could not detect fixation lights! Will proceed WITHOUT glare exclusion.")
                    st.info("💡 All pixels will be included in analysis. If results are poor, try adjusting video brightness or contrast.")
                    # Create empty glare mask (all pixels valid)
                    glare_mask = np.ones(gray_frame.shape, dtype=bool)
                else:
                    st.success("✅ Fixation lights detected successfully")
                    glare_mask = glare_excluder.create_glare_mask(gray_frame.shape)
                    
                    # Run optimization
                    with st.spinner("Running grid search... This may take a minute."):
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        # Redirect print output (simplified version)
                        best_result = optimizer.suggest_settings(
                            gray_frame, 
                            glare_mask,
                            thresh_k_range=(thresh_k_min, thresh_k_max, thresh_k_step),
                            floor_range=(floor_min, floor_max, floor_step),
                            verbose=False  # Avoid console clutter
                        )
                        
                        progress_bar.progress(100)
                        status_text.text("✅ Optimization complete!")
                    
                    # Display results
                    st.markdown("---")
                    st.success("🎉 Optimization Complete!")
                    
                    st.markdown("## 🏆 Best Parameters Found")
                    
                    # Main F1 Score - Most prominent
                    st.metric(
                        label="🎯 F1 Score (Optimal Balance)", 
                        value=f"{best_result['f1']:.4f}",
                        help="Harmonic mean of Precision and Recall - Higher is better"
                    )
                    
                    # Best parameters
                    st.markdown("### Optimal Parameter Values")
                    param_col1, param_col2 = st.columns(2, gap="large")
                    with param_col1:
                        st.metric("⚙️ Best thresh_k", f"{best_result['thresh_k']:.2f}")
                    with param_col2:
                        st.metric("🔧 Best floor_threshold", f"{best_result['floor_threshold']:.2f}")
                    
                    # Detailed metrics
                    st.markdown("---")
                    st.markdown("### Detailed Performance Metrics")
                    metric_col_a, metric_col_b, metric_col_c, metric_col_d = st.columns(4, gap="medium")
                    with metric_col_a:
                        st.metric("Precision", f"{best_result['precision']:.3f}")
                    with metric_col_b:
                        st.metric("Recall", f"{best_result['recall']:.3f}")
                    with metric_col_c:
                        st.metric("✅ True Positives", best_result['tp'])
                    with metric_col_d:
                        st.metric("❌ False Positives", best_result['fp'])
                    
                    # Save to session state
                    st.session_state.optimized_params = best_result
                    st.session_state.optimization_results = optimizer.optimization_results
                    st.session_state.last_optimizer = optimizer
                    
                    # Action buttons (full width for better visibility)
                    st.markdown("---")
                    st.markdown("#### Actions")
                    
                    if st.button("✅ Apply These Optimized Settings to Config", 
                                type="primary", 
                                use_container_width=True,
                                key="apply_opt_btn"):
                        opt = st.session_state.get('last_optimizer')
                        if opt is not None and opt.best_params is not None:
                            optimized_cfg = opt.apply_best_settings()
                            st.session_state.config.thresh_k = optimized_cfg.thresh_k
                            st.session_state.config.floor_threshold = optimized_cfg.floor_threshold
                        else:
                            st.session_state.config.thresh_k = best_result['thresh_k']
                            st.session_state.config.floor_threshold = best_result['floor_threshold']
                        st.session_state.config.validation_match_tolerance = match_tolerance
                        st.success(
                            f"✅ Applied: thresh_k={st.session_state.config.thresh_k:.2f}, "
                            f"floor_threshold={st.session_state.config.floor_threshold:.2f}, "
                            f"match_tolerance={st.session_state.config.validation_match_tolerance:.1f}px"
                        )
                        st.rerun()
                    
                    if st.button("📥 Export Full Grid Search Results to CSV", 
                                use_container_width=True,
                                key="export_opt_btn"):
                        optimizer.export_results("optimization_grid_search.csv")
                        st.success("📄 Exported to: optimization_grid_search.csv")
                    
                    # Heatmap visualization
                    if 'optimization_results' in st.session_state:
                        st.markdown("---")
                        st.markdown("### 📊 F1 Score Heatmap")
                        st.caption("Visual representation of all tested parameter combinations")
                        
                        results_df = pd.DataFrame(st.session_state.optimization_results)
                        pivot = results_df.pivot(index='thresh_k', 
                                                columns='floor_threshold', 
                                                values='f1')
                        
                        fig, ax = plt.subplots(figsize=(12, 7))
                        im = ax.imshow(pivot.values, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
                        
                        # Set ticks
                        ax.set_xticks(np.arange(len(pivot.columns)))
                        ax.set_yticks(np.arange(len(pivot.index)))
                        ax.set_xticklabels([f"{x:.2f}" for x in pivot.columns], rotation=45)
                        ax.set_yticklabels([f"{y:.2f}" for y in pivot.index])
                        
                        # Labels and title
                        ax.set_xlabel('floor_threshold', fontsize=12, fontweight='bold')
                        ax.set_ylabel('thresh_k', fontsize=12, fontweight='bold')
                        ax.set_title('F1 Score Heatmap (Grid Search Results)', 
                                    fontsize=14, fontweight='bold', pad=20)
                        
                        # Colorbar
                        cbar = plt.colorbar(im, ax=ax, label='F1 Score')
                        cbar.set_label('F1 Score', fontsize=11, fontweight='bold')
                        
                        # Mark best point
                        best_idx_y = np.where(pivot.index == best_result['thresh_k'])[0][0]
                        best_idx_x = np.where(pivot.columns == best_result['floor_threshold'])[0][0]
                        ax.scatter(best_idx_x, best_idx_y, 
                                  marker='*', s=500, c='blue', 
                                  edgecolors='white', linewidths=2,
                                  label='Best Parameters')
                        ax.legend(loc='upper right', fontsize=10)
                        
                        plt.tight_layout()
                        st.pyplot(fig)

# ===== FOOTER =====
st.sidebar.markdown("---")

# Show current epoch status if video loaded
if st.session_state.video_loaded and st.session_state.epochs is not None:
    st.sidebar.markdown("### 📊 Current Video Status")
    
    with st.sidebar.expander("Epoch Information", expanded=False):
        st.write(f"**Total Frames:** {st.session_state.num_frames}")
        st.write(f"**Blink Events:** {len(st.session_state.blink_ranges)}")
        st.write(f"**Valid Epochs:** {len(st.session_state.epochs)}")
        st.write(f"**Safe Frames:** {len(st.session_state.safe_frames)} "
                f"({len(st.session_state.safe_frames)/st.session_state.num_frames*100:.1f}%)")
        
        if len(st.session_state.epochs) > 0:
            st.markdown("**Epoch Details:**")
            for i, epoch in enumerate(st.session_state.epochs):
                st.write(f"- Epoch {i+1}: frames {epoch.start_frame}-{epoch.end_frame} "
                        f"({epoch.length()} frames)")

st.sidebar.markdown("---")
st.sidebar.markdown("""
### 📖 Quick Tips
- 🎥 **Load video** first - epochs are auto-detected
- 🎯 **Safe frames** only in Titration & Optimization
- 🔍 **Blink Detection** tab to adjust sensitivity
- 🚀 **Run Analysis** when parameters are ready
- 📊 **Results** for post-analysis visualization

**Version:** 2.1.0 (Safe Frame Selection)  
**Author:** Tear Film Research Lab
""")
