"""
Track tear-film particles across frames using the trained U-Net.

Loads unet_tear_film.pth, segments each frame, extracts centroids, matches
particles between consecutive frames, and exports velocities + trajectory video.

Usage:
    python track_particles.py
    python track_particles.py --fps 30 --mm-per-pixel 0.01
"""

from __future__ import annotations

import argparse
import csv
import logging
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch

from train_unet import (
    DEFAULT_IMAGES_DIR,
    DEFAULT_MODEL_OUT,
    IMAGE_SIZE,
    PROJECT_ROOT,
    UNet,
)

DEFAULT_CSV_OUT = PROJECT_ROOT / "particle_tracks.csv"
DEFAULT_VIDEO_DIR = PROJECT_ROOT / "output" / "tracking"
THRESHOLD = 0.2
FRAME_PATTERN = re.compile(r"^(?P<prefix>.+)_frame_(?P<index>\d+)$", re.IGNORECASE)


@dataclass
class Particle:
    """Detected particle in a single frame."""

    x: float
    y: float
    area: float


@dataclass
class TrackPoint:
    video_id: str
    frame_number: int
    frame_index: int
    filename: str
    particle_id: int
    x: float
    y: float
    time_sec: float
    velocity_px_per_frame: float = 0.0
    velocity_px_per_sec: float = 0.0
    velocity_mm_per_sec: float = 0.0


@dataclass
class ParticleTracker:
    """Greedy nearest-neighbour tracker for consecutive frames."""

    max_distance: float = 50.0
    next_id: int = 0
    active: Dict[int, Tuple[float, float]] = field(default_factory=dict)
    histories: Dict[int, List[Tuple[int, float, float]]] = field(default_factory=dict)

    def reset(self) -> None:
        self.next_id = 0
        self.active = {}
        self.histories = {}

    def update(
        self,
        particles: List[Particle],
        frame_number: int,
    ) -> Dict[int, Particle]:
        """Match detections to existing tracks; return particle_id -> Particle."""
        if not self.active:
            assigned: Dict[int, Particle] = {}
            for p in particles:
                pid = self.next_id
                self.next_id += 1
                self.active[pid] = (p.x, p.y)
                self.histories.setdefault(pid, []).append((frame_number, p.x, p.y))
                assigned[pid] = p
            return assigned

        updated: Dict[int, Particle] = {}
        remaining = dict(self.active)

        for p in particles:
            best_id: Optional[int] = None
            best_dist = float("inf")

            for pid, (ox, oy) in remaining.items():
                dist = float(np.hypot(p.x - ox, p.y - oy))
                if dist < best_dist and dist <= self.max_distance:
                    best_dist = dist
                    best_id = pid

            if best_id is not None:
                updated[best_id] = p
                self.histories[best_id].append((frame_number, p.x, p.y))
                del remaining[best_id]
            else:
                pid = self.next_id
                self.next_id += 1
                updated[pid] = p
                self.histories.setdefault(pid, []).append((frame_number, p.x, p.y))

        self.active = {pid: (p.x, p.y) for pid, p in updated.items()}
        return updated


# ---------------------------------------------------------------------------
# Model & segmentation
# ---------------------------------------------------------------------------

def load_model(checkpoint_path: Path, device: torch.device) -> UNet:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = UNet(
        in_channels=checkpoint.get("in_channels", 3),
        out_channels=checkpoint.get("out_channels", 1),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def predict_mask_fullres(
    model: UNet,
    image_bgr: np.ndarray,
    device: torch.device,
    threshold: float = THRESHOLD,
    image_size: int = IMAGE_SIZE,
) -> np.ndarray:
    """Run U-Net inference; return binary mask at original image resolution."""
    h, w = image_bgr.shape[:2]
    resized = cv2.resize(image_bgr, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    tensor = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).to(device)

    probs = torch.sigmoid(model(tensor)).squeeze().cpu().numpy()
    mask_small = (probs >= threshold).astype(np.uint8) * 255
    return cv2.resize(mask_small, (w, h), interpolation=cv2.INTER_NEAREST)


def extract_particles(
    mask: np.ndarray,
    min_area: int = 4,
) -> List[Particle]:
    """Find connected components and return centroids in pixel coordinates."""
    num_labels, _, stats, centroids = cv2.connectedComponentsWithStats(
        mask, connectivity=8
    )
    particles: List[Particle] = []
    for label in range(1, num_labels):
        area = stats[label, cv2.CC_STAT_AREA]
        if area < min_area:
            continue
        cx, cy = centroids[label]
        particles.append(Particle(x=float(cx), y=float(cy), area=float(area)))
    return particles


# ---------------------------------------------------------------------------
# Frame grouping & sorting
# ---------------------------------------------------------------------------

def parse_frame_path(path: Path) -> Tuple[str, int]:
    """Return (video_prefix, frame_number) from a frame filename stem."""
    match = FRAME_PATTERN.match(path.stem)
    if not match:
        return path.stem, 0
    return match.group("prefix"), int(match.group("index"))


def group_frames_by_video(frames_dir: Path) -> Dict[str, List[Path]]:
    """Group image paths by video prefix and sort by frame number."""
    groups: Dict[str, List[Path]] = defaultdict(list)
    extensions = {".jpg", ".jpeg", ".png", ".bmp"}

    for path in frames_dir.iterdir():
        if path.is_file() and path.suffix.lower() in extensions:
            video_id, _ = parse_frame_path(path)
            groups[video_id].append(path)

    for video_id in groups:
        groups[video_id].sort(key=lambda p: parse_frame_path(p)[1])

    return dict(sorted(groups.items()))


# ---------------------------------------------------------------------------
# Velocity & export
# ---------------------------------------------------------------------------

def compute_velocity(
    x_prev: float,
    y_prev: float,
    x_curr: float,
    y_curr: float,
    delta_frames: int,
    fps: float,
    mm_per_pixel: Optional[float],
) -> Tuple[float, float, float]:
    """Return px/frame, px/s, mm/s velocities."""
    if delta_frames <= 0:
        return 0.0, 0.0, 0.0

    dist_px = float(np.hypot(x_curr - x_prev, y_curr - y_prev))
    vel_px_frame = dist_px / delta_frames
    vel_px_sec = vel_px_frame * fps
    vel_mm_sec = vel_px_sec * mm_per_pixel if mm_per_pixel else 0.0
    return vel_px_frame, vel_px_sec, vel_mm_sec


def save_csv(records: List[TrackPoint], csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "video_id",
        "frame_number",
        "frame_index",
        "filename",
        "time_sec",
        "particle_id",
        "centroid_x",
        "centroid_y",
        "velocity_px_per_frame",
        "velocity_px_per_sec",
        "velocity_mm_per_sec",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in records:
            writer.writerow(
                {
                    "video_id": r.video_id,
                    "frame_number": r.frame_number,
                    "frame_index": r.frame_index,
                    "filename": r.filename,
                    "time_sec": f"{r.time_sec:.4f}",
                    "particle_id": r.particle_id,
                    "centroid_x": f"{r.x:.2f}",
                    "centroid_y": f"{r.y:.2f}",
                    "velocity_px_per_frame": f"{r.velocity_px_per_frame:.4f}",
                    "velocity_px_per_sec": f"{r.velocity_px_per_sec:.4f}",
                    "velocity_mm_per_sec": f"{r.velocity_mm_per_sec:.4f}",
                }
            )


def draw_trajectories(
    frame_bgr: np.ndarray,
    tracker: ParticleTracker,
    assigned: Dict[int, Particle],
    colors: Dict[int, Tuple[int, int, int]],
) -> np.ndarray:
    """Draw particle centroids and recent trajectory trails."""
    vis = frame_bgr.copy()

    for pid, history in tracker.histories.items():
        if len(history) < 2:
            continue
        color = colors.setdefault(pid, _id_color(pid))
        pts = [(int(x), int(y)) for _, x, y in history[-20:]]
        for i in range(1, len(pts)):
            cv2.line(vis, pts[i - 1], pts[i], color, 1, cv2.LINE_AA)

    for pid, p in assigned.items():
        color = colors.setdefault(pid, _id_color(pid))
        cv2.circle(vis, (int(p.x), int(p.y)), 4, color, -1, cv2.LINE_AA)
        cv2.putText(
            vis,
            str(pid),
            (int(p.x) + 5, int(p.y) - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            color,
            1,
            cv2.LINE_AA,
        )
    return vis


def draw_tracking_overlay(
    frame_bgr: np.ndarray,
    mask: np.ndarray,
    tracker: ParticleTracker,
    assigned: Dict[int, Particle],
    colors: Dict[int, Tuple[int, int, int]],
) -> np.ndarray:
    """Draw mask contours, centroids, IDs, and trajectory trails."""
    vis = frame_bgr.copy()
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(vis, contours, -1, (0, 255, 0), 1, lineType=cv2.LINE_AA)

    tint = vis.copy()
    tint[mask > 0] = (0, 180, 0)
    vis = cv2.addWeighted(vis, 0.75, tint, 0.25, 0)

    for pid, history in tracker.histories.items():
        if len(history) < 2:
            continue
        color = colors.setdefault(pid, _id_color(pid))
        pts = [(int(x), int(y)) for _, x, y in history[-25:]]
        for i in range(1, len(pts)):
            cv2.line(vis, pts[i - 1], pts[i], color, 2, cv2.LINE_AA)

    for pid, p in assigned.items():
        color = colors.setdefault(pid, _id_color(pid))
        cv2.circle(vis, (int(p.x), int(p.y)), 5, color, -1, cv2.LINE_AA)
        cv2.putText(
            vis,
            f"ID{pid}",
            (int(p.x) + 6, int(p.y) - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )
    return vis


def _id_color(particle_id: int) -> Tuple[int, int, int]:
    rng = np.random.default_rng(particle_id)
    return tuple(int(c) for c in rng.integers(60, 255, size=3))


def write_tracking_video(
    frames: List[np.ndarray],
    output_path: Path,
    fps: float,
) -> None:
    if not frames:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    h, w = frames[0].shape[:2]
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (w, h),
    )
    for frame in frames:
        writer.write(frame)
    writer.release()


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def process_video_sequence(
    video_id: str,
    frame_paths: List[Path],
    model: UNet,
    device: torch.device,
    fps: float,
    mm_per_pixel: Optional[float],
    threshold: float,
    max_track_distance: float,
    min_particle_area: int,
    make_video: bool,
) -> Tuple[List[TrackPoint], List[np.ndarray]]:
    tracker = ParticleTracker(max_distance=max_track_distance)
    records: List[TrackPoint] = []
    vis_frames: List[np.ndarray] = []
    colors: Dict[int, Tuple[int, int, int]] = {}
    prev_positions: Dict[int, Tuple[float, float, int]] = {}

    for frame_index, path in enumerate(frame_paths):
        image = cv2.imread(str(path))
        if image is None:
            logging.warning("Could not read %s", path.name)
            continue

        _, frame_number = parse_frame_path(path)
        time_sec = frame_number / fps

        mask = predict_mask_fullres(model, image, device, threshold)
        particles = extract_particles(mask, min_area=min_particle_area)
        assigned = tracker.update(particles, frame_number)

        for pid, p in assigned.items():
            v_px_f, v_px_s, v_mm_s = 0.0, 0.0, 0.0
            if pid in prev_positions:
                px, py, p_frame = prev_positions[pid]
                delta = frame_number - p_frame
                v_px_f, v_px_s, v_mm_s = compute_velocity(
                    px, py, p.x, p.y, delta, fps, mm_per_pixel
                )

            records.append(
                TrackPoint(
                    video_id=video_id,
                    frame_number=frame_number,
                    frame_index=frame_index,
                    filename=path.name,
                    particle_id=pid,
                    x=p.x,
                    y=p.y,
                    time_sec=time_sec,
                    velocity_px_per_frame=v_px_f,
                    velocity_px_per_sec=v_px_s,
                    velocity_mm_per_sec=v_mm_s,
                )
            )
            prev_positions[pid] = (p.x, p.y, frame_number)

        if make_video:
            vis_frames.append(
                draw_tracking_overlay(image, mask, tracker, assigned, colors)
            )

    return records, vis_frames


def process_video_file(
    video_path: Path,
    output_path: Path,
    model: UNet,
    device: torch.device,
    video_id: Optional[str] = None,
    fps: Optional[float] = None,
    threshold: float = THRESHOLD,
    max_track_distance: float = 50.0,
    min_particle_area: int = 4,
    mm_per_pixel: Optional[float] = None,
    progress_callback: Optional[Callable[[int, int], None]] = None,
    frame_indices: Optional[List[int]] = None,
) -> List[TrackPoint]:
    """
    Process a video frame-by-frame: segment, track, visualize.

    If *frame_indices* is given, only those source frame numbers are processed
    (e.g. blink-filtered safe frames). Velocity uses real frame gaps.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    video_id = video_id or video_path.stem
    native_fps = cap.get(cv2.CAP_PROP_FPS)
    fps = fps or (native_fps if native_fps and native_fps > 0 else 30.0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    tracker = ParticleTracker(max_distance=max_track_distance)
    records: List[TrackPoint] = []
    colors: Dict[int, Tuple[int, int, int]] = {}
    prev_positions: Dict[int, Tuple[float, float, int]] = {}

    if frame_indices is None:
        indices_to_process = list(range(total_frames))
    else:
        indices_to_process = sorted(set(frame_indices))
    n_process = len(indices_to_process)

    try:
        for seq_idx, source_frame in enumerate(indices_to_process):
            cap.set(cv2.CAP_PROP_POS_FRAMES, source_frame)
            ret, frame = cap.read()
            if not ret:
                continue

            mask = predict_mask_fullres(model, frame, device, threshold)
            particles = extract_particles(mask, min_area=min_particle_area)
            assigned = tracker.update(particles, source_frame)
            time_sec = source_frame / fps

            for pid, p in assigned.items():
                v_px_f, v_px_s, v_mm_s = 0.0, 0.0, 0.0
                if pid in prev_positions:
                    px, py, p_idx = prev_positions[pid]
                    delta = source_frame - p_idx
                    v_px_f, v_px_s, v_mm_s = compute_velocity(
                        px, py, p.x, p.y, delta, fps, mm_per_pixel
                    )

                records.append(
                    TrackPoint(
                        video_id=video_id,
                        frame_number=source_frame,
                        frame_index=seq_idx,
                        filename=f"frame_{source_frame:04d}",
                        particle_id=pid,
                        x=p.x,
                        y=p.y,
                        time_sec=time_sec,
                        velocity_px_per_frame=v_px_f,
                        velocity_px_per_sec=v_px_s,
                        velocity_mm_per_sec=v_mm_s,
                    )
                )
                prev_positions[pid] = (p.x, p.y, source_frame)

            vis = draw_tracking_overlay(frame, mask, tracker, assigned, colors)
            writer.write(vis)

            if progress_callback is not None:
                progress_callback(seq_idx + 1, n_process)
    finally:
        cap.release()
        writer.release()

    return records


def enrich_tracking_dataframe(
    df,
    *,
    epochs: Sequence,
    fps: float,
):
    """
    Add blink-relative columns for medical report / FDM / power-law analysis.

    Requires epoch list from BlinkDetector (start_frame, end_frame).
    """
    from medical_report import enrich_tracks_with_blink_timing

    return enrich_tracks_with_blink_timing(df, epochs, fps)


def trackpoints_to_dataframe(records: List[TrackPoint]):
    """Convert track records to a pandas DataFrame."""
    import pandas as pd

    if not records:
        return pd.DataFrame(
            columns=[
                "video_id", "frame_number", "frame_index", "filename",
                "time_sec", "particle_id", "centroid_x", "centroid_y",
                "velocity_px_per_frame", "velocity_px_per_sec", "velocity_mm_per_sec",
            ]
        )
    return pd.DataFrame(
        [
            {
                "video_id": r.video_id,
                "frame_number": r.frame_number,
                "frame_index": r.frame_index,
                "filename": r.filename,
                "time_sec": r.time_sec,
                "particle_id": r.particle_id,
                "centroid_x": r.x,
                "centroid_y": r.y,
                "velocity_px_per_frame": r.velocity_px_per_frame,
                "velocity_px_per_sec": r.velocity_px_per_sec,
                "velocity_mm_per_sec": r.velocity_mm_per_sec,
            }
            for r in records
        ]
    )


def run_tracking(
    model_path: Path = DEFAULT_MODEL_OUT,
    frames_dir: Path = DEFAULT_IMAGES_DIR,
    csv_out: Path = DEFAULT_CSV_OUT,
    video_dir: Path = DEFAULT_VIDEO_DIR,
    fps: float = 30.0,
    mm_per_pixel: Optional[float] = None,
    threshold: float = THRESHOLD,
    max_track_distance: float = 50.0,
    min_particle_area: int = 4,
    preview_videos: bool = True,
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info("Device: %s", device)

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path.resolve()}")

    model = load_model(model_path, device)
    groups = group_frames_by_video(frames_dir)

    if not groups:
        raise FileNotFoundError(f"No frames in {frames_dir.resolve()}")

    logging.info("Found %d video sequence(s), %d total frames",
                 len(groups), sum(len(v) for v in groups.values()))

    all_records: List[TrackPoint] = []

    for video_id, paths in groups.items():
        logging.info("Processing %s (%d frames)...", video_id, len(paths))
        records, vis_frames = process_video_sequence(
            video_id=video_id,
            frame_paths=paths,
            model=model,
            device=device,
            fps=fps,
            mm_per_pixel=mm_per_pixel,
            threshold=threshold,
            max_track_distance=max_track_distance,
            min_particle_area=min_particle_area,
            make_video=preview_videos,
        )
        all_records.extend(records)

        if preview_videos and vis_frames:
            video_out = video_dir / f"{video_id}_tracks.mp4"
            write_tracking_video(vis_frames, video_out, fps)
            logging.info("  Trajectory video -> %s", video_out.resolve())

    save_csv(all_records, csv_out)
    logging.info("Saved %d track records -> %s", len(all_records), csv_out.resolve())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Track tear-film particles with U-Net segmentation."
    )
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_OUT)
    parser.add_argument("--frames-dir", type=Path, default=DEFAULT_IMAGES_DIR)
    parser.add_argument("--csv-out", type=Path, default=DEFAULT_CSV_OUT)
    parser.add_argument("--video-dir", type=Path, default=DEFAULT_VIDEO_DIR)
    parser.add_argument("--fps", type=float, default=30.0,
                        help="Video frame rate for time/velocity conversion")
    parser.add_argument("--mm-per-pixel", type=float, default=None,
                        help="Calibration: mm per pixel (optional, enables mm/s)")
    parser.add_argument("--threshold", type=float, default=THRESHOLD)
    parser.add_argument("--max-track-distance", type=float, default=50.0,
                        help="Max pixel distance for frame-to-frame matching")
    parser.add_argument("--min-particle-area", type=int, default=4)
    parser.add_argument("--no-video", action="store_true",
                        help="Skip trajectory video output")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    args = parse_args()
    run_tracking(
        model_path=args.model,
        frames_dir=args.frames_dir,
        csv_out=args.csv_out,
        video_dir=args.video_dir,
        fps=args.fps,
        mm_per_pixel=args.mm_per_pixel,
        threshold=args.threshold,
        max_track_distance=args.max_track_distance,
        min_particle_area=args.min_particle_area,
        preview_videos=not args.no_video,
    )


if __name__ == "__main__":
    main()
