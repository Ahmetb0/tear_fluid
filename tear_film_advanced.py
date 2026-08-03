"""
Advanced Tear Film Analysis System
===================================
Refactored OOP-based implementation inspired by PTLib JavaScript library.

Features:
1. Blink Detection & Epoch Segmentation (Z-score based brightness analysis)
2. Glare Exclusion (buffer zones around fixation lights)
3. Adaptive Bandpass Filtering (local mean/std thresholding)
4. Modular Parameter Management (ready for UI integration)
5. FWHM-based Particle Shape Analysis (major/minor radius, orientation, elongation)
6. Automatic Parameter Optimization (ValidationOptimizer with grid search)

Author: Tear Film Research Lab
Date: 2026
Version: 2.1.0
"""

import cv2
import numpy as np
import csv
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
from scipy.ndimage import gaussian_filter
import warnings
warnings.filterwarnings('ignore')


@dataclass
class TearFilmConfig:
    """Configuration parameters for tear film analysis."""
    
    # Video parameters
    video_path: str = ""
    fps: float = 30.0
    
    # Blink detection parameters
    blink_z_threshold: float = 4.0
    blink_pad_frames: int = 3
    min_epoch_length: int = 5
    
    # Glare exclusion parameters
    glare_buffer_radius: int = 30  # pixels around fixation lights
    ref_light_threshold: int = 200
    min_ref_light_distance: float = 30.0
    
    # Particle detection parameters (adaptive bandpass)
    sigma_small: float = 1.0
    sigma_large: float = 6.0
    thresh_k: float = 3.0  # adaptive threshold multiplier
    local_window_size: int = 41
    floor_threshold: float = 0.5
    min_particle_area: int = 1
    max_particle_area: int = 50
    
    # FWHM (Full Width at Half Maximum) parameters
    fwhm_enabled: bool = True  # Enable FWHM shape analysis
    fwhm_search_radius: int = 4  # Pixels to search for peak
    fwhm_rel_threshold: float = 0.5  # Relative threshold (0.5 = half-max)
    fwhm_max_radius: int = 12  # Maximum particle radius for FWHM
    
    # Tracking parameters
    max_tracking_distance: float = 0.2  # normalized units
    
    # Validation/Optimization parameters
    validation_match_tolerance: float = 5.0  # Pixels for ground truth matching
    
    # Output parameters
    output_csv: str = "tear_film_analysis.csv"
    show_visualization: bool = True


class FrameSignal:
    """Container for frame-level brightness statistics."""
    
    def __init__(self, mean: float, bright_count: int):
        self.mean = mean
        self.bright_count = bright_count


class BlinkDetector:
    """
    Detects eye blinks using robust Z-score analysis on brightness signal.
    
    Implements the algorithm from PTLib.detectBlinks():
    - Uses median-based robust Z-score (not max-based normalization)
    - Detects large deviations in both mean brightness and bright pixel count
    - Handles both brightness spikes (closed lid reflection) and dips
    """
    
    def __init__(self, config: TearFilmConfig):
        self.config = config
        self.frame_signals: List[FrameSignal] = []
    
    @staticmethod
    def compute_robust_z_score(values: np.ndarray) -> np.ndarray:
        """
        Compute robust Z-score using median and MAD (Median Absolute Deviation).
        More stable than mean/std for outlier detection.
        """
        if len(values) < 3:
            return np.zeros_like(values)
        
        median = np.median(values)
        mad = np.median(np.abs(values - median))
        
        # MAD to std conversion factor for normal distribution
        mad_to_std = 1.4826
        robust_std = mad * mad_to_std
        
        if robust_std < 1e-6:
            return np.zeros_like(values)
        
        return (values - median) / robust_std
    
    def compute_frame_signal(self, frame: np.ndarray, roi_mask: Optional[np.ndarray] = None) -> FrameSignal:
        """
        Compute brightness statistics for a single frame.
        
        Args:
            frame: Grayscale image
            roi_mask: Optional binary mask for ROI
            
        Returns:
            FrameSignal with mean brightness and bright pixel count
        """
        if roi_mask is not None:
            pixels = frame[roi_mask > 0]
        else:
            pixels = frame.ravel()
        
        mean_brightness = float(np.mean(pixels))
        bright_count = int(np.sum(pixels > 200))
        
        return FrameSignal(mean_brightness, bright_count)
    
    def detect_blinks(self, video_path: str, roi_mask: Optional[np.ndarray] = None) -> List[Tuple[int, int]]:
        """
        Detect blink frames in video using Z-score analysis.
        
        Args:
            video_path: Path to video file
            roi_mask: Optional ROI mask
            
        Returns:
            List of (start_frame, end_frame) tuples for blink ranges
        """
        cap = cv2.VideoCapture(video_path)
        self.frame_signals = []
        
        print("Computing brightness signal for blink detection...")
        frame_idx = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            signal = self.compute_frame_signal(gray, roi_mask)
            self.frame_signals.append(signal)
            frame_idx += 1
            
            if frame_idx % 100 == 0:
                print(f"  Processed {frame_idx} frames...")
        
        cap.release()
        print(f"Total frames: {len(self.frame_signals)}")
        
        # Extract signals
        mean_values = np.array([s.mean for s in self.frame_signals])
        bright_counts = np.array([s.bright_count for s in self.frame_signals])
        
        # Compute robust Z-scores
        mean_z = self.compute_robust_z_score(mean_values)
        bright_z = self.compute_robust_z_score(bright_counts)
        
        # Flag frames with high Z-scores
        flagged = (np.abs(mean_z) > self.config.blink_z_threshold) | \
                  (np.abs(bright_z) > self.config.blink_z_threshold)
        
        # Group flagged frames into ranges
        blink_ranges = []
        i = 0
        n = len(flagged)
        
        while i < n:
            if flagged[i]:
                j = i
                while j < n and flagged[j]:
                    j += 1
                
                # Add padding
                start = max(0, i - self.config.blink_pad_frames)
                end = min(n - 1, j - 1 + self.config.blink_pad_frames)
                blink_ranges.append((start, end))
                i = j
            else:
                i += 1
        
        # Merge overlapping ranges
        if blink_ranges:
            blink_ranges.sort(key=lambda x: x[0])
            merged = [list(blink_ranges[0])]
            
            for start, end in blink_ranges[1:]:
                if start <= merged[-1][1] + 1:
                    merged[-1][1] = max(merged[-1][1], end)
                else:
                    merged.append([start, end])
            
            blink_ranges = [tuple(r) for r in merged]
        
        print(f"Detected {len(blink_ranges)} blink events")
        return blink_ranges


@dataclass
class Epoch:
    """Represents a continuous open-eye interval."""
    start_frame: int
    end_frame: int
    
    def length(self) -> int:
        return self.end_frame - self.start_frame


class EpochSegmenter:
    """
    Segments video into open-eye epochs based on blink detection.
    Implements PTLib.segmentEpochs() logic.
    """
    
    def __init__(self, config: TearFilmConfig):
        self.config = config
    
    def segment_epochs(self, num_frames: int, blink_ranges: List[Tuple[int, int]]) -> List[Epoch]:
        """
        Create epochs (open-eye intervals) between blinks.
        
        Args:
            num_frames: Total number of frames in video
            blink_ranges: List of (start, end) blink frame ranges
            
        Returns:
            List of Epoch objects
        """
        cuts = [0]
        for start, end in blink_ranges:
            cuts.extend([start, end + 1])
        cuts.append(num_frames)
        
        epochs = []
        for k in range(0, len(cuts) - 1, 2):
            start = cuts[k]
            end = cuts[k + 1]
            
            if end - start >= self.config.min_epoch_length:
                epochs.append(Epoch(start, end))
        
        print(f"Segmented into {len(epochs)} valid epochs")
        total_frames = sum(e.length() for e in epochs)
        print(f"  Total analyzable frames: {total_frames}/{num_frames} ({100*total_frames/num_frames:.1f}%)")
        
        return epochs


class GlareExcluder:
    """
    Identifies and excludes glare regions (fixation lights) from analysis.
    Implements PTLib.fixationLightMask() and excludeGlare() logic.
    """
    
    def __init__(self, config: TearFilmConfig):
        self.config = config
        self.superior_light: Optional[Tuple[int, int]] = None
        self.inferior_light: Optional[Tuple[int, int]] = None
        self.normalization_distance: Optional[float] = None
        self.lights_valid: bool = False
    
    def clear_fixation_lights(self) -> None:
        """Reset fixation light state (e.g. at epoch boundary or after failed detection)."""
        self.superior_light = None
        self.inferior_light = None
        self.normalization_distance = None
        self.lights_valid = False
    
    @property
    def has_valid_fixation_lights(self) -> bool:
        """True only when detection succeeded and geometry was validated."""
        return (
            self.lights_valid
            and self.superior_light is not None
            and self.inferior_light is not None
            and self.normalization_distance is not None
            and self.normalization_distance > self.config.min_ref_light_distance
        )
    
    def detect_fixation_lights(self, frame: np.ndarray) -> bool:
        """
        Detect two fixation light positions.
        
        Args:
            frame: Grayscale frame
            
        Returns:
            True if both lights detected and geometry validated.
            On failure, internal state is cleared (no partial assignment).
        """
        blurred = cv2.GaussianBlur(frame, (5, 5), 0)
        _, thresh = cv2.threshold(
            blurred, self.config.ref_light_threshold, 255, cv2.THRESH_BINARY
        )
        
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:2]
        
        if len(contours) < 2:
            self.clear_fixation_lights()
            return False
        
        # Extract centers (do not assign to self until validation passes)
        centers = []
        for cnt in contours:
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                centers.append((cx, cy))
        
        if len(centers) < 2:
            self.clear_fixation_lights()
            return False
        
        centers.sort(key=lambda p: p[1])
        superior = centers[0]
        inferior = centers[1]
        
        dx = superior[0] - inferior[0]
        dy = superior[1] - inferior[1]
        norm_distance = float(np.sqrt(dx * dx + dy * dy))
        
        x_diff = abs(dx)
        y_diff = abs(dy)
        
        geometry_ok = (
            norm_distance > self.config.min_ref_light_distance
            and y_diff > (x_diff * 2)
        )
        
        if not geometry_ok:
            self.clear_fixation_lights()
            print(
                f"  [Fixation] Geometry check failed: "
                f"distance={norm_distance:.1f}px (min={self.config.min_ref_light_distance}), "
                f"vertical/horizontal ratio={y_diff / max(x_diff, 1e-6):.2f} (need >2.0)"
            )
            return False
        
        self.superior_light = superior
        self.inferior_light = inferior
        self.normalization_distance = norm_distance
        self.lights_valid = True
        print(
            f"  [Fixation] OK: superior={superior}, inferior={inferior}, "
            f"norm_distance={norm_distance:.1f}px"
        )
        return True
    
    def create_glare_mask(self, frame_shape: Tuple[int, int]) -> np.ndarray:
        """
        Create binary mask excluding glare regions.
        
        Args:
            frame_shape: (height, width) of frame
            
        Returns:
            Binary mask (1 = valid, 0 = excluded)
        """
        h, w = frame_shape
        mask = np.ones((h, w), dtype=np.uint8)
        
        if not self.has_valid_fixation_lights:
            return mask
        
        buffer = self.config.glare_buffer_radius
        
        # Exclude circular regions around each light
        cv2.circle(mask, self.superior_light, buffer, 0, -1)
        cv2.circle(mask, self.inferior_light, buffer, 0, -1)
        
        return mask


class ParticleDetector:
    """
    Adaptive particle detection using bandpass filtering and local statistics.
    Implements PTLib.particleEnhancedView() and PTLib.detectParticles() logic.
    """
    
    def __init__(self, config: TearFilmConfig):
        self.config = config
    
    def bandpass_filter(self, image: np.ndarray) -> np.ndarray:
        """
        Apply bandpass filter using Difference of Gaussians (DoG).
        
        Args:
            image: Grayscale image
            
        Returns:
            Bandpass filtered image
        """
        small_blur = gaussian_filter(image.astype(np.float32), 
                                     sigma=self.config.sigma_small)
        large_blur = gaussian_filter(image.astype(np.float32), 
                                     sigma=self.config.sigma_large)
        
        # Difference of Gaussians
        dog = small_blur - large_blur
        
        return dog
    
    def local_mean_std(self, image: np.ndarray, window_size: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute local mean and standard deviation using box filters.
        
        Args:
            image: Input image
            window_size: Window size for local statistics
            
        Returns:
            (local_mean, local_std) arrays
        """
        # Use integral image for fast box filtering
        kernel = np.ones((window_size, window_size), np.float32) / (window_size * window_size)
        
        local_mean = cv2.filter2D(image, -1, kernel)
        local_mean_sq = cv2.filter2D(image**2, -1, kernel)
        
        # Variance = E[X^2] - E[X]^2
        local_var = np.maximum(local_mean_sq - local_mean**2, 0)
        local_std = np.sqrt(local_var)
        
        return local_mean, local_std
    
    def detect_particles(self, frame: np.ndarray, glare_mask: np.ndarray) -> List[Dict]:
        """
        Detect particles using adaptive local thresholding on bandpass image.
        
        Args:
            frame: Grayscale frame
            glare_mask: Binary mask (1 = valid region)
            
        Returns:
            List of particle detections with properties
        """
        # Apply bandpass filter
        dog_image = self.bandpass_filter(frame)
        
        # Compute local statistics
        local_mean, local_std = self.local_mean_std(
            dog_image, 
            self.config.local_window_size
        )
        
        # Adaptive threshold: pixel > mean + k*std + floor
        threshold_map = local_mean + \
                       self.config.thresh_k * local_std + \
                       self.config.floor_threshold
        
        # Create binary mask
        binary = ((dog_image > threshold_map) & (glare_mask > 0)).astype(np.uint8)
        
        # Find connected components
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        particles = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            
            if self.config.min_particle_area <= area <= self.config.max_particle_area:
                # Compute centroid
                M = cv2.moments(cnt)
                if M["m00"] != 0:
                    cx = M["m10"] / M["m00"]
                    cy = M["m01"] / M["m00"]
                    
                    # Compute second moments for ellipse fitting
                    mu20 = M["mu20"] / M["m00"]
                    mu02 = M["mu02"] / M["m00"]
                    mu11 = M["mu11"] / M["m00"]
                    
                    particle = {
                        'x': cx,
                        'y': cy,
                        'area': area,
                        'mu20': mu20,
                        'mu02': mu02,
                        'mu11': mu11,
                        'contour': cnt
                    }
                    
                    # Add FWHM-based shape analysis if enabled
                    if self.config.fwhm_enabled:
                        fwhm_props = self.estimate_particle_extent(dog_image, cx, cy)
                        particle.update(fwhm_props)
                    
                    particles.append(particle)
        
        return particles
    
    def estimate_particle_extent(self, dog_image: np.ndarray, cx: float, cy: float) -> Dict:
        """
        Estimate particle physical extent using FWHM (Full Width at Half Maximum).
        
        This method implements PTLib.estimateParticleExtent() to measure particle
        shape properties based on intensity profile rather than just binary area.
        
        Args:
            dog_image: Bandpass filtered image (DoG)
            cx, cy: Initial centroid coordinates
            
        Returns:
            Dictionary with FWHM-based shape measurements:
            - major_radius: Major axis radius (pixels)
            - minor_radius: Minor axis radius (pixels)
            - orientation: Orientation angle (radians)
            - elongation: Elongation ratio (major/minor)
            - fwhm_area: FWHM-based area
            - peak_value: Peak intensity value
        """
        h, w = dog_image.shape
        search_r = self.config.fwhm_search_radius
        max_radius = self.config.fwhm_max_radius
        rel_thresh = self.config.fwhm_rel_threshold
        
        # Find local peak within search radius
        ix0, iy0 = int(round(cx)), int(round(cy))
        peak_x, peak_y, peak_v = ix0, iy0, -np.inf
        
        for dy in range(-search_r, search_r + 1):
            for dx in range(-search_r, search_r + 1):
                x, y = ix0 + dx, iy0 + dy
                if 0 <= x < w and 0 <= y < h:
                    v = dog_image[y, x]
                    if v > peak_v:
                        peak_v = v
                        peak_x, peak_y = x, y
        
        # If peak is not positive, return degenerate case
        if peak_v <= 0:
            return {
                'major_radius': 1.0,
                'minor_radius': 1.0,
                'orientation': 0.0,
                'elongation': 1.0,
                'fwhm_area': 1.0,
                'peak_value': float(peak_v),
                'fwhm_degenerate': True
            }
        
        # FWHM threshold (half-maximum)
        thresh = peak_v * rel_thresh
        
        # Define search region
        x0 = max(0, peak_x - max_radius)
        x1 = min(w, peak_x + max_radius + 1)
        y0 = max(0, peak_y - max_radius)
        y1 = min(h, peak_y + max_radius + 1)
        
        # Flood fill from peak to find FWHM region (vectorized approach)
        visited = np.zeros((h, w), dtype=bool)
        stack = [(peak_x, peak_y)]
        visited[peak_y, peak_x] = True
        
        sum_x, sum_y = 0.0, 0.0
        sum_xx, sum_yy, sum_xy = 0.0, 0.0, 0.0
        area = 0
        
        while stack:
            cx2, cy2 = stack.pop()
            sum_x += cx2
            sum_y += cy2
            sum_xx += cx2 * cx2
            sum_yy += cy2 * cy2
            sum_xy += cx2 * cy2  # Note: typo in original JS (cx2*cy2 should be), keeping consistent
            area += 1
            
            # Check 4-connected neighbors
            for nx, ny in [(cx2-1, cy2), (cx2+1, cy2), (cx2, cy2-1), (cx2, cy2+1)]:
                if x0 <= nx < x1 and y0 <= ny < y1:
                    if not visited[ny, nx] and dog_image[ny, nx] >= thresh:
                        visited[ny, nx] = True
                        stack.append((nx, ny))
        
        if area == 0:
            return {
                'major_radius': 1.0,
                'minor_radius': 1.0,
                'orientation': 0.0,
                'elongation': 1.0,
                'fwhm_area': 1.0,
                'peak_value': float(peak_v),
                'fwhm_degenerate': True
            }
        
        # Compute FWHM centroid
        ecx = sum_x / area
        ecy = sum_y / area
        
        # Compute second central moments (equivalent ellipse)
        Ixx = sum_xx / area - ecx * ecx
        Iyy = sum_yy / area - ecy * ecy
        Ixy = sum_xy / area - ecx * ecy
        
        # Eigenvalues of moment matrix
        mid = (Ixx + Iyy) / 2.0
        spread = np.sqrt(max(0, ((Ixx - Iyy) / 2.0) ** 2 + Ixy * Ixy))
        
        # Major and minor radii (factor of 2 for equivalent ellipse)
        major_r = 2.0 * np.sqrt(max(0, mid + spread))
        minor_r = 2.0 * np.sqrt(max(0, mid - spread))
        
        # Orientation (radians)
        orientation_rad = 0.5 * np.arctan2(2.0 * Ixy, Ixx - Iyy)
        
        # Elongation ratio
        if minor_r > 1e-6:
            elongation = major_r / minor_r
        else:
            elongation = 99.0 if major_r > 1e-6 else 1.0
        
        return {
            'major_radius': float(major_r),
            'minor_radius': float(minor_r),
            'orientation': float(orientation_rad),
            'elongation': float(elongation),
            'fwhm_area': float(area),
            'peak_value': float(peak_v),
            'fwhm_degenerate': False
        }


class ParticleTracker:
    """
    Greedy nearest-neighbor particle tracking with normalized coordinates.
    Implements tracking logic similar to PTLib.linkTracks().
    """
    
    def __init__(self, config: TearFilmConfig):
        self.config = config
        self.tracked_objects: Dict[int, Tuple[float, float]] = {}
        self.next_id = 0
        self.trajectories: Dict[int, List[Tuple[float, float]]] = {}
        self.last_frame_idx: Optional[int] = None
    
    def normalize_coordinates(self, x: float, y: float, 
                            ref_point: Tuple[int, int], 
                            norm_distance: float) -> Tuple[float, float]:
        """
        Normalize coordinates relative to reference point and distance.
        
        Args:
            x, y: Pixel coordinates
            ref_point: Reference point (superior light)
            norm_distance: Normalization distance
            
        Returns:
            (x_norm, y_norm) in normalized units
        """
        x_norm = (x - ref_point[0]) / norm_distance
        y_norm = (y - ref_point[1]) / norm_distance
        return x_norm, y_norm
    
    def denormalize_coordinates(self, x_norm: float, y_norm: float,
                               ref_point: Tuple[int, int],
                               norm_distance: float) -> Tuple[int, int]:
        """Convert normalized coordinates back to pixel coordinates."""
        x = int(x_norm * norm_distance + ref_point[0])
        y = int(y_norm * norm_distance + ref_point[1])
        return x, y
    
    def update(self, particles: List[Dict], 
               ref_point: Tuple[int, int],
               norm_distance: float,
               fps: float,
               frame_idx: int) -> Tuple[Dict[int, Tuple[float, float]], Dict[int, float]]:
        """
        Update tracking with new particle detections.
        
        Args:
            particles: List of detected particles
            ref_point: Reference point for normalization
            norm_distance: Normalization distance
            fps: Frame rate for velocity calculation
            frame_idx: Current video frame index (for dynamic delta_t)
            
        Returns:
            (tracked_objects, velocities) dictionaries
        """
        # Normalize particle coordinates
        normalized_particles = []
        for p in particles:
            x_norm, y_norm = self.normalize_coordinates(
                p['x'], p['y'], ref_point, norm_distance
            )
            normalized_particles.append((x_norm, y_norm))
        
        velocities = {}
        
        # First frame: initialize all particles
        if len(self.tracked_objects) == 0:
            for x_norm, y_norm in normalized_particles:
                self.tracked_objects[self.next_id] = (x_norm, y_norm)
                self.trajectories[self.next_id] = [(x_norm, y_norm)]
                self.next_id += 1
            self.last_frame_idx = frame_idx
            return self.tracked_objects, velocities
        
        # Real elapsed time since last processed frame (handles skipped frames)
        if self.last_frame_idx is not None:
            delta_frames = max(1, frame_idx - self.last_frame_idx)
        else:
            delta_frames = 1
        delta_t = delta_frames / fps
        updated_objects = {}
        max_dist = self.config.max_tracking_distance
        
        for new_point in normalized_particles:
            best_match_id = None
            best_distance = float('inf')
            
            for obj_id, old_point in self.tracked_objects.items():
                dx = new_point[0] - old_point[0]
                dy = new_point[1] - old_point[1]
                distance = np.sqrt(dx*dx + dy*dy)
                
                if distance < best_distance and distance < max_dist:
                    best_distance = distance
                    best_match_id = obj_id
            
            if best_match_id is not None:
                # Matched existing track
                updated_objects[best_match_id] = new_point
                
                # MMS velocity: normalized distance per 0.1 second
                mms_velocity = (best_distance / delta_t) * 0.1
                velocities[best_match_id] = mms_velocity
                
                # Update trajectory
                if best_match_id in self.trajectories:
                    self.trajectories[best_match_id].append(new_point)
                
                # Remove from old tracks
                del self.tracked_objects[best_match_id]
            else:
                # New track
                updated_objects[self.next_id] = new_point
                self.trajectories[self.next_id] = [new_point]
                self.next_id += 1
        
        self.tracked_objects = updated_objects
        self.last_frame_idx = frame_idx
        return self.tracked_objects, velocities
    
    def reset(self):
        """Reset tracking state (called on blinks)."""
        self.tracked_objects = {}
        self.trajectories = {}
        self.last_frame_idx = None


# Detection parameters that grid search may optimize and Apply should sync
OPTIMIZABLE_DETECTION_PARAMS = (
    'thresh_k',
    'floor_threshold',
    'min_particle_area',
    'max_particle_area',
    'glare_buffer_radius',
    'sigma_small',
    'sigma_large',
    'validation_match_tolerance',
)


class ValidationOptimizer:
    """
    Automatic parameter optimization using ground truth validation.
    
    This class performs grid search over detection parameters to find
    optimal settings that maximize F1 score against manually annotated
    ground truth particles.
    """
    
    def __init__(self, config: TearFilmConfig):
        self.config = config
        self.ground_truth_points: List[Tuple[float, float]] = []
        self.best_params: Optional[Dict] = None
        self.optimization_results: List[Dict] = []
    
    def set_ground_truth(self, points: List[Tuple[float, float]]) -> None:
        """
        Set ground truth particle positions for validation.
        
        Args:
            points: List of (x, y) pixel coordinates of true particles
        """
        self.ground_truth_points = points
        print(f"Ground truth set: {len(points)} particles")
    
    def match_detections(self, detected_particles: List[Dict], 
                        tolerance: float) -> Tuple[int, int, int]:
        """
        Match detected particles to ground truth within tolerance distance.
        
        Args:
            detected_particles: List of detected particle dictionaries
            tolerance: Maximum distance (pixels) for a match
            
        Returns:
            (true_positives, false_positives, false_negatives) tuple
        """
        if len(self.ground_truth_points) == 0:
            return 0, len(detected_particles), 0
        
        gt_matched = set()
        det_matched = set()
        
        # For each ground truth point, find closest detection
        for i, (gt_x, gt_y) in enumerate(self.ground_truth_points):
            best_dist = float('inf')
            best_det_idx = None
            
            for j, particle in enumerate(detected_particles):
                if j in det_matched:
                    continue
                
                dx = particle['x'] - gt_x
                dy = particle['y'] - gt_y
                dist = np.sqrt(dx*dx + dy*dy)
                
                if dist < best_dist and dist <= tolerance:
                    best_dist = dist
                    best_det_idx = j
            
            if best_det_idx is not None:
                gt_matched.add(i)
                det_matched.add(best_det_idx)
        
        true_positives = len(gt_matched)
        false_positives = len(detected_particles) - len(det_matched)
        false_negatives = len(self.ground_truth_points) - len(gt_matched)
        
        return true_positives, false_positives, false_negatives
    
    def compute_metrics(self, tp: int, fp: int, fn: int) -> Dict[str, float]:
        """
        Compute precision, recall, and F1 score.
        
        Args:
            tp: True positives
            fp: False positives
            fn: False negatives
            
        Returns:
            Dictionary with precision, recall, f1 metrics
        """
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'tp': tp,
            'fp': fp,
            'fn': fn
        }
    
    def evaluate_parameters(self, frame: np.ndarray, glare_mask: np.ndarray,
                          thresh_k: float, floor_threshold: float) -> Dict:
        """
        Evaluate a specific parameter combination against ground truth.
        
        Args:
            frame: Grayscale frame to process
            glare_mask: Glare exclusion mask
            thresh_k: Adaptive threshold multiplier
            floor_threshold: Floor threshold value
            
        Returns:
            Dictionary with parameters and metrics
        """
        # Create temporary config
        temp_config = TearFilmConfig(**self.config.__dict__)
        temp_config.thresh_k = thresh_k
        temp_config.floor_threshold = floor_threshold
        
        # Detect particles with these parameters
        detector = ParticleDetector(temp_config)
        particles = detector.detect_particles(frame, glare_mask)
        
        # Match against ground truth
        tp, fp, fn = self.match_detections(
            particles, 
            self.config.validation_match_tolerance
        )
        
        # Compute metrics
        metrics = self.compute_metrics(tp, fp, fn)
        
        return {
            'thresh_k': thresh_k,
            'floor_threshold': floor_threshold,
            'num_detected': len(particles),
            **metrics
        }
    
    def suggest_settings(self, frame: np.ndarray, glare_mask: np.ndarray,
                        thresh_k_range: Tuple[float, float, float] = (1.5, 8.0, 0.5),
                        floor_range: Tuple[float, float, float] = (0.0, 2.0, 0.25),
                        verbose: bool = True) -> Dict:
        """
        Perform grid search to find optimal detection parameters.
        
        Args:
            frame: Grayscale frame to optimize on
            glare_mask: Glare exclusion mask
            thresh_k_range: (min, max, step) for thresh_k
            floor_range: (min, max, step) for floor_threshold
            verbose: Print progress
            
        Returns:
            Dictionary with best parameters and metrics
        """
        if len(self.ground_truth_points) == 0:
            raise ValueError("Ground truth not set! Call set_ground_truth() first.")
        
        # Generate parameter grid
        thresh_k_values = np.arange(*thresh_k_range)
        floor_values = np.arange(*floor_range)
        
        total_combinations = len(thresh_k_values) * len(floor_values)
        
        if verbose:
            print(f"\n{'='*60}")
            print("PARAMETER OPTIMIZATION (Grid Search)")
            print(f"{'='*60}")
            print(f"Ground truth particles: {len(self.ground_truth_points)}")
            print(f"Testing {total_combinations} combinations:")
            print(f"  thresh_k: {thresh_k_range}")
            print(f"  floor_threshold: {floor_range}")
            print(f"{'='*60}\n")
        
        self.optimization_results = []
        best_f1 = -1.0
        best_result = None
        
        for i, thresh_k in enumerate(thresh_k_values):
            for j, floor_thresh in enumerate(floor_values):
                result = self.evaluate_parameters(
                    frame, glare_mask, thresh_k, floor_thresh
                )
                self.optimization_results.append(result)
                
                if result['f1'] > best_f1:
                    best_f1 = result['f1']
                    best_result = result
                
                if verbose and (len(self.optimization_results) % 10 == 0):
                    progress = len(self.optimization_results) / total_combinations * 100
                    print(f"Progress: {progress:.1f}% | "
                          f"Best F1 so far: {best_f1:.3f} "
                          f"(k={best_result['thresh_k']:.2f}, "
                          f"floor={best_result['floor_threshold']:.2f})")
        
        self.best_params = best_result
        
        if verbose:
            print(f"\n{'='*60}")
            print("OPTIMIZATION COMPLETE")
            print(f"{'='*60}")
            print(f"Best F1 Score: {best_result['f1']:.4f}")
            print(f"Best Parameters:")
            print(f"  thresh_k: {best_result['thresh_k']:.2f}")
            print(f"  floor_threshold: {best_result['floor_threshold']:.2f}")
            print(f"\nMetrics:")
            print(f"  Precision: {best_result['precision']:.4f}")
            print(f"  Recall: {best_result['recall']:.4f}")
            print(f"  F1 Score: {best_result['f1']:.4f}")
            print(f"  True Positives: {best_result['tp']}")
            print(f"  False Positives: {best_result['fp']}")
            print(f"  False Negatives: {best_result['fn']}")
            print(f"  Detected: {best_result['num_detected']}")
            print(f"{'='*60}\n")
        
        return best_result
    
    def apply_best_settings(self) -> TearFilmConfig:
        """
        Create a new config with optimized parameters.
        
        Returns:
            Updated TearFilmConfig with best parameters
        """
        if self.best_params is None:
            raise ValueError("No optimization performed yet! Call suggest_settings() first.")
        
        optimized_config = TearFilmConfig(**self.config.__dict__)
        optimized_config.thresh_k = self.best_params['thresh_k']
        optimized_config.floor_threshold = self.best_params['floor_threshold']
        for key in OPTIMIZABLE_DETECTION_PARAMS:
            if key in self.best_params:
                setattr(optimized_config, key, self.best_params[key])
        
        print(
            f"[Optimizer] Applied best settings: "
            f"thresh_k={optimized_config.thresh_k:.3f}, "
            f"floor_threshold={optimized_config.floor_threshold:.3f}, "
            f"F1={self.best_params.get('f1', 0):.4f}"
        )
        
        return optimized_config
    
    def export_results(self, filename: str = "optimization_results.csv") -> None:
        """Export all optimization results to CSV for analysis."""
        if len(self.optimization_results) == 0:
            print("No optimization results to export.")
            return
        
        import csv
        with open(filename, 'w', newline='') as f:
            fieldnames = ['thresh_k', 'floor_threshold', 'num_detected', 
                         'precision', 'recall', 'f1', 'tp', 'fp', 'fn']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.optimization_results)
        
        print(f"Optimization results exported to {filename}")


class TearFilmAnalyzer:
    """
    Main orchestrator for advanced tear film analysis.
    Coordinates all components for complete video processing.
    """
    
    def __init__(self, config: TearFilmConfig):
        self.config = config
        self.blink_detector = BlinkDetector(config)
        self.epoch_segmenter = EpochSegmenter(config)
        self.glare_excluder = GlareExcluder(config)
        self.particle_detector = ParticleDetector(config)
        self.particle_tracker = ParticleTracker(config)
        
        self.epochs: List[Epoch] = []
        self.results: List[Dict] = []
    
    def analyze_video(self) -> None:
        """
        Complete analysis pipeline:
        1. Detect blinks and segment epochs
        2. Process each epoch frame-by-frame
        3. Save results to CSV
        """
        print("="*60)
        print("ADVANCED TEAR FILM ANALYSIS")
        print("="*60)
        
        # Step 1: Blink detection
        print("\n[1/4] Detecting blinks...")
        blink_ranges = self.blink_detector.detect_blinks(self.config.video_path)
        
        # Step 2: Epoch segmentation
        print("\n[2/4] Segmenting epochs...")
        cap = cv2.VideoCapture(self.config.video_path)
        num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = self.config.fps
        self.config.fps = fps
        cap.release()
        
        self.epochs = self.epoch_segmenter.segment_epochs(num_frames, blink_ranges)
        
        if len(self.epochs) == 0:
            print("ERROR: No valid epochs found. Video may be too short or blinks too frequent.")
            return
        
        # Step 3: Process epochs
        print(f"\n[3/4] Processing {len(self.epochs)} epochs...")
        self._process_epochs()
        
        # Step 4: Save results
        print(f"\n[4/4] Saving results to {self.config.output_csv}...")
        n_fit = sum(1 for r in self.results if r.get('include_in_power_law_fit', True))
        n_skip = len(self.results) - n_fit
        if n_skip:
            print(
                f"  CSV rows: {len(self.results)} total | "
                f"{n_fit} eligible for power-law fit | "
                f"{n_skip} excluded (non-post-blink epochs)"
            )
        self._save_results()
        
        print("\n" + "="*60)
        print("ANALYSIS COMPLETE")
        print(f"Total particles tracked: {len(self.results)}")
        print(f"Output file: {self.config.output_csv}")
        print("="*60)
    
    def _process_epochs(self) -> None:
        """Process all epochs frame by frame."""
        cap = cv2.VideoCapture(self.config.video_path)
        
        for epoch_idx, epoch in enumerate(self.epochs):
            print(f"\n  Epoch {epoch_idx+1}/{len(self.epochs)}: frames {epoch.start_frame}-{epoch.end_frame}")
            
            # Reset tracker and fixation state for new epoch (retry detection each epoch)
            self.particle_tracker.reset()
            self.glare_excluder.clear_fixation_lights()
            
            # Epoch 0 from frame 0 is not a true post-blink interval for power-law pooling
            include_in_power_law = epoch.start_frame > 0
            if not include_in_power_law:
                print(
                    f"  [Epoch] start_frame=0: records kept in CSV but excluded from "
                    f"power-law fit (not post-blink)"
                )
            
            # Seek to epoch start
            cap.set(cv2.CAP_PROP_POS_FRAMES, epoch.start_frame)
            
            tracked: Dict[int, Tuple[float, float]] = {}
            velocities: Dict[int, float] = {}
            particles: List[Dict] = []
            glare_mask = np.ones((1, 1), dtype=np.uint8)
            
            for frame_idx in range(epoch.start_frame, epoch.end_frame):
                ret, frame = cap.read()
                if not ret:
                    break
                
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                # Retry fixation detection every frame until valid (BUG-F1 fix)
                if not self.glare_excluder.has_valid_fixation_lights:
                    if not self.glare_excluder.detect_fixation_lights(gray):
                        if (frame_idx - epoch.start_frame) % 25 == 0:
                            print(
                                f"    [Fixation] Frame {frame_idx}: no valid lights, "
                                f"skipping (will retry)"
                            )
                        continue
                
                glare_mask = self.glare_excluder.create_glare_mask(gray.shape)
                particles = self.particle_detector.detect_particles(gray, glare_mask)
                
                if self.glare_excluder.has_valid_fixation_lights:
                    tracked, velocities = self.particle_tracker.update(
                        particles,
                        self.glare_excluder.superior_light,
                        self.glare_excluder.normalization_distance,
                        self.config.fps,
                        frame_idx
                    )
                    
                    time_sec = frame_idx / self.config.fps
                    time_since_blink_s = (frame_idx - epoch.start_frame) / self.config.fps
                    
                    particle_lookup = {}
                    for p in particles:
                        p_x_norm, p_y_norm = self.particle_tracker.normalize_coordinates(
                            p['x'], p['y'],
                            self.glare_excluder.superior_light,
                            self.glare_excluder.normalization_distance
                        )
                        key = (round(p_x_norm, 4), round(p_y_norm, 4))
                        particle_lookup[key] = p
                    
                    for particle_id, (x_norm, y_norm) in tracked.items():
                        velocity = velocities.get(particle_id, 0.0)
                        
                        if velocity > 0:
                            result = {
                                'frame': frame_idx,
                                'time_sec': time_sec,
                                'time_since_blink_s': time_since_blink_s,
                                'include_in_power_law_fit': include_in_power_law,
                                'particle_id': particle_id,
                                'x_norm': x_norm,
                                'y_norm': y_norm,
                                'mms_velocity': velocity,
                                'epoch': epoch_idx
                            }
                            
                            key = (round(x_norm, 4), round(y_norm, 4))
                            if key in particle_lookup and 'major_radius' in particle_lookup[key]:
                                p = particle_lookup[key]
                                result.update({
                                    'major_radius': p['major_radius'],
                                    'minor_radius': p['minor_radius'],
                                    'orientation': p['orientation'],
                                    'elongation': p['elongation'],
                                    'fwhm_area': p['fwhm_area'],
                                    'peak_value': p['peak_value']
                                })
                            
                            self.results.append(result)
                
                # Visualization
                if self.config.show_visualization and frame_idx % 5 == 0:
                    self._visualize_frame(frame, glare_mask, particles, tracked, velocities)
                
                if frame_idx % 50 == 0:
                    progress = (frame_idx - epoch.start_frame) / epoch.length() * 100
                    print(f"    Progress: {progress:.1f}%", end='\r')
        
        cap.release()
        if self.config.show_visualization:
            cv2.destroyAllWindows()
    
    def _visualize_frame(self, frame: np.ndarray, glare_mask: np.ndarray,
                        particles: List[Dict], tracked: Dict, velocities: Dict) -> None:
        """Draw visualization overlay on frame."""
        vis_frame = frame.copy()
        
        # Draw glare exclusion zones
        if self.glare_excluder.has_valid_fixation_lights:
            cv2.circle(vis_frame, self.glare_excluder.superior_light, 
                      self.config.glare_buffer_radius, (255, 0, 0), 2)
            cv2.circle(vis_frame, self.glare_excluder.inferior_light, 
                      self.config.glare_buffer_radius, (255, 0, 0), 2)
            cv2.line(vis_frame, self.glare_excluder.superior_light,
                    self.glare_excluder.inferior_light, (255, 0, 0), 1)
        
        # Draw tracked particles
        for particle_id, (x_norm, y_norm) in tracked.items():
            if self.glare_excluder.has_valid_fixation_lights:
                x, y = self.particle_tracker.denormalize_coordinates(
                    x_norm, y_norm,
                    self.glare_excluder.superior_light,
                    self.glare_excluder.normalization_distance
                )
                
                cv2.circle(vis_frame, (x, y), 4, (0, 255, 0), -1)
                
                velocity = velocities.get(particle_id, 0.0)
                if velocity > 0:
                    label = f"ID:{particle_id} V:{velocity:.3f}"
                    cv2.putText(vis_frame, label, (x+8, y-8),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
                
                # Draw trajectory
                if particle_id in self.particle_tracker.trajectories:
                    traj = self.particle_tracker.trajectories[particle_id]
                    if len(traj) > 1:
                        for i in range(1, len(traj)):
                            pt1 = self.particle_tracker.denormalize_coordinates(
                                traj[i-1][0], traj[i-1][1],
                                self.glare_excluder.superior_light,
                                self.glare_excluder.normalization_distance
                            )
                            pt2 = self.particle_tracker.denormalize_coordinates(
                                traj[i][0], traj[i][1],
                                self.glare_excluder.superior_light,
                                self.glare_excluder.normalization_distance
                            )
                            cv2.line(vis_frame, pt1, pt2, (0, 0, 255), 2)
        
        # Info overlay
        info_text = f"Particles: {len(tracked)} | FPS: {self.config.fps:.1f}"
        cv2.putText(vis_frame, info_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.imshow("Advanced Tear Film Analysis", vis_frame)
        cv2.waitKey(1)
    
    def _save_results(self) -> None:
        """Save analysis results to CSV file."""
        base_fields = [
            'frame', 'time_sec', 'time_since_blink_s', 'include_in_power_law_fit',
            'epoch', 'particle_id', 'x_norm', 'y_norm', 'mms_velocity'
        ]
        
        if self.config.fwhm_enabled and len(self.results) > 0:
            # Check if first result has FWHM data
            if 'major_radius' in self.results[0]:
                fwhm_fields = ['major_radius', 'minor_radius', 'orientation', 
                              'elongation', 'fwhm_area', 'peak_value']
                fieldnames = base_fields + fwhm_fields
            else:
                fieldnames = base_fields
        else:
            fieldnames = base_fields
        
        with open(self.config.output_csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            for result in self.results:
                row = {
                    'frame': result['frame'],
                    'time_sec': f"{result['time_sec']:.3f}",
                    'time_since_blink_s': f"{result['time_since_blink_s']:.3f}",
                    'include_in_power_law_fit': int(bool(result.get('include_in_power_law_fit', True))),
                    'epoch': result['epoch'],
                    'particle_id': result['particle_id'],
                    'x_norm': f"{result['x_norm']:.4f}",
                    'y_norm': f"{result['y_norm']:.4f}",
                    'mms_velocity': f"{result['mms_velocity']:.4f}"
                }
                
                # Add FWHM data if available
                if 'major_radius' in result:
                    row.update({
                        'major_radius': f"{result['major_radius']:.4f}",
                        'minor_radius': f"{result['minor_radius']:.4f}",
                        'orientation': f"{result['orientation']:.4f}",
                        'elongation': f"{result['elongation']:.4f}",
                        'fwhm_area': f"{result['fwhm_area']:.2f}",
                        'peak_value': f"{result['peak_value']:.4f}"
                    })
                
                writer.writerow(row)


def compute_power_law_decay(df: 'pd.DataFrame', bin_size: float = 0.15, 
                           time_col: str = 'time_since_blink_s',
                           velocity_col: str = 'mms_velocity') -> dict:
    """
    Compute power-law decay curve fitting for tear film velocity.
    
    This function implements the power-law decay model from medical literature:
        v = α * t^(-β)
    
    where:
        - v: velocity (mm/s)
        - t: time since blink (seconds)
        - α (alpha): initial velocity coefficient
        - β (beta): decay exponent
    
    Args:
        df: DataFrame with analysis results (must have time_since_blink_s and mms_velocity)
        bin_size: Time bin size in seconds (default: 0.15s)
        time_col: Name of time column (default: 'time_since_blink_s')
        velocity_col: Name of velocity column (default: 'mms_velocity')
    
    Returns:
        Dictionary containing:
        - 'binned_time': Array of bin center times
        - 'binned_velocity': Array of median velocities per bin
        - 'alpha': Fitted alpha parameter
        - 'beta': Fitted beta parameter
        - 'fitted_curve': Fitted velocity values at binned times
        - 'r_squared': R² goodness of fit
        - 'equation': String representation of fitted equation
    """
    import pandas as pd
    from scipy.optimize import curve_fit
    
    # Exclude non-post-blink epochs (e.g. epoch starting at frame 0) from fit pool
    fit_source = df.copy()
    if 'include_in_power_law_fit' in fit_source.columns:
        excluded = (~fit_source['include_in_power_law_fit'].astype(bool)).sum()
        fit_source = fit_source[fit_source['include_in_power_law_fit'].astype(bool)]
        if excluded > 0:
            print(
                f"[Power-Law] Excluded {excluded} rows from non-post-blink epochs "
                f"(include_in_power_law_fit=0)"
            )
    
    # Filter valid data (positive time and velocity)
    valid_data = fit_source[(fit_source[time_col] > 0) & (fit_source[velocity_col] > 0)].copy()
    
    if len(valid_data) == 0:
        print("⚠️ No valid data for power-law fitting")
        return None
    
    # Create time bins
    max_time = valid_data[time_col].max()
    bins = np.arange(0, max_time + bin_size, bin_size)
    valid_data['time_bin'] = pd.cut(valid_data[time_col], bins=bins)
    
    # Calculate median velocity per bin
    binned_stats = valid_data.groupby('time_bin', observed=True)[velocity_col].agg(['median', 'mean', 'count'])
    binned_stats = binned_stats[binned_stats['count'] >= 3]  # Require at least 3 points per bin
    
    if len(binned_stats) < 4:
        print("⚠️ Insufficient bins for curve fitting (need at least 4 bins with ≥3 points)")
        return None
    
    # Extract bin centers and median velocities
    bin_centers = np.array([interval.mid for interval in binned_stats.index])
    median_velocities = binned_stats['median'].values
    
    # Remove any NaN or zero values
    valid_mask = (~np.isnan(bin_centers)) & (~np.isnan(median_velocities)) & (bin_centers > 0) & (median_velocities > 0)
    bin_centers = bin_centers[valid_mask]
    median_velocities = median_velocities[valid_mask]
    
    if len(bin_centers) < 4:
        print("⚠️ Insufficient valid bins after filtering")
        return None
    
    # Define power-law function: v = alpha * t^(-beta)
    def power_law(t, alpha, beta):
        return alpha * np.power(t, -beta)
    
    try:
        # Curve fitting with reasonable initial guesses
        # Initial guess: alpha ~ first velocity, beta ~ 0.5 (typical for tear film)
        initial_guess = [median_velocities[0], 0.5]
        
        # Fit the curve
        params, covariance = curve_fit(
            power_law, 
            bin_centers, 
            median_velocities,
            p0=initial_guess,
            bounds=([0.01, 0.01], [100, 3.0]),  # Reasonable physical bounds
            maxfev=5000
        )
        
        alpha, beta = params
        
        # Calculate fitted curve
        fitted_velocities = power_law(bin_centers, alpha, beta)
        
        # Calculate R² (coefficient of determination)
        ss_res = np.sum((median_velocities - fitted_velocities) ** 2)
        ss_tot = np.sum((median_velocities - np.mean(median_velocities)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        # Create equation string
        equation = f"v = {alpha:.3f} × t^(-{beta:.3f})"
        
        return {
            'binned_time': bin_centers,
            'binned_velocity': median_velocities,
            'alpha': alpha,
            'beta': beta,
            'fitted_curve': fitted_velocities,
            'r_squared': r_squared,
            'equation': equation,
            'bin_size': bin_size,
            'num_bins': len(bin_centers)
        }
        
    except Exception as e:
        print(f"❌ Power-law fitting failed: {e}")
        return None


def main():
    """Main entry point with example usage."""
    
    # Configuration
    config = TearFilmConfig(
        video_path="C:/Users/Asus/Desktop/tear_fluid/assests/AYDIN_MEHMET TUNAHAN_Right_2026_07_26-14_01_16.mkv",
        
        # Blink detection parameters
        blink_z_threshold=4.0,
        blink_pad_frames=3,
        min_epoch_length=5,
        
        # Glare exclusion
        glare_buffer_radius=30,
        
        # Particle detection (titration parameters)
        thresh_k=3.0,
        min_particle_area=1,
        max_particle_area=50,
        
        # Output
        output_csv="tear_film_analysis_advanced.csv",
        show_visualization=True
    )
    
    # Run analysis
    analyzer = TearFilmAnalyzer(config)
    analyzer.analyze_video()


if __name__ == "__main__":
    main()
