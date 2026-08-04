"""
Extract uniformly sampled frames from videos for U-Net manual labeling.

Default source: project `assests/` folder with the 5 patient eye videos used
for U-Net training. Output goes to `./data/unet_raw_frames`.

Usage:
    python extract_frames.py
    python extract_frames.py --target-frames 165 --format jpg
    python extract_frames.py --all-videos --input ./assests
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - optional dependency
    tqdm = None

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}
DEFAULT_INPUT_DIR = Path("./assests")
DEFAULT_OUTPUT_DIR = Path("./data/unet_raw_frames")
DEFAULT_TARGET_FRAMES = 165
DEFAULT_MIN_FRAMES = 150
DEFAULT_MAX_FRAMES = 180

# U-Net labeling set — project patient videos in assests/
PROJECT_VIDEOS = (
    "ATALAY_ERAY_Left_2026_07_26-14_25_35.mkv",
    "AYDIN_MEHMET TUNAHAN_Left_2026_07_26-14_01_38.mkv",
    "BOZKURT_AHMET_Left_2026_07_26-14_09_10.mkv",
    "BOZKURT_AHMET_Right_2026_07_26-14_08_40.mkv",
    "AYDIN_MEHMET TUNAHAN_Right_2026_07_26-14_01_16.mkv",
)


@dataclass
class FrameExtractionConfig:
    """Configuration for uniform frame extraction."""

    input_dir: Path
    output_dir: Path
    target_frames: int = DEFAULT_TARGET_FRAMES
    image_format: str = "jpg"  # "jpg" or "png"
    jpg_quality: int = 95
    margin_frames: int = 5  # skip first/last N frames when possible
    video_names: Optional[Sequence[str]] = field(default_factory=lambda: list(PROJECT_VIDEOS))

    def __post_init__(self) -> None:
        self.input_dir = Path(self.input_dir)
        self.output_dir = Path(self.output_dir)
        self.image_format = self.image_format.lower().lstrip(".")
        if self.image_format not in {"jpg", "jpeg", "png"}:
            raise ValueError("image_format must be 'jpg' or 'png'")
        if not (DEFAULT_MIN_FRAMES <= self.target_frames <= DEFAULT_MAX_FRAMES):
            logging.warning(
                "target_frames=%d is outside recommended range [%d, %d]",
                self.target_frames,
                DEFAULT_MIN_FRAMES,
                DEFAULT_MAX_FRAMES,
            )


@dataclass
class VideoInfo:
    """Metadata for a single source video."""

    path: Path
    frame_count: int


@dataclass
class ExtractionResult:
    """Summary of one video extraction pass."""

    video_path: Path
    requested_frames: int
    saved_frames: int
    frame_indices: List[int]
    output_paths: List[Path]


def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def sanitize_stem(stem: str) -> str:
    """Make a filesystem-safe, lowercase identifier from a video filename."""
    cleaned = stem.strip().lower()
    cleaned = re.sub(r"[^\w\-]+", "_", cleaned, flags=re.UNICODE)
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")
    return cleaned or "video"


def discover_videos(
    input_dir: Path,
    video_names: Optional[Sequence[str]] = None,
) -> List[Path]:
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    if video_names:
        videos: List[Path] = []
        missing: List[str] = []
        for name in video_names:
            path = input_dir / name
            if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS:
                videos.append(path)
            else:
                missing.append(name)
        if missing:
            raise FileNotFoundError(
                f"Video(s) not found in {input_dir}:\n  "
                + "\n  ".join(missing)
            )
        return videos

    return sorted(
        path
        for path in input_dir.iterdir()
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
    )


def get_video_frame_count(video_path: Path) -> int:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    if frame_count <= 0:
        raise RuntimeError(f"Invalid frame count for video: {video_path}")
    return frame_count


def collect_video_info(video_paths: Sequence[Path]) -> List[VideoInfo]:
    infos: List[VideoInfo] = []
    iterator: Iterable[Path] = video_paths
    if tqdm is not None:
        iterator = tqdm(video_paths, desc="Scanning videos", unit="video")

    for path in iterator:
        count = get_video_frame_count(path)
        infos.append(VideoInfo(path=path, frame_count=count))
        logging.debug("%s -> %d frames", path.name, count)

    return infos


def distribute_frame_budget(
    video_infos: Sequence[VideoInfo],
    target_frames: int,
) -> List[int]:
    """
    Split target frame count across videos as evenly as possible.

    Remainder frames are assigned one-by-one to the longest videos first so
    shorter clips are not overloaded.
    """
    n_videos = len(video_infos)
    if n_videos == 0:
        return []

    base = target_frames // n_videos
    remainder = target_frames % n_videos

    budgets = [base] * n_videos
    if remainder:
        ranked_indices = sorted(
            range(n_videos),
            key=lambda i: video_infos[i].frame_count,
            reverse=True,
        )
        for idx in ranked_indices[:remainder]:
            budgets[idx] += 1

    return budgets


def compute_uniform_frame_indices(
    frame_count: int,
    num_frames: int,
    margin_frames: int = 5,
) -> List[int]:
    """
    Pick evenly spaced, non-consecutive frame indices within a video.

    Uses np.linspace over a safe interior range when the video is long enough;
    otherwise falls back to linspace over the full clip.
    """
    if num_frames <= 0:
        return []

    if frame_count <= 0:
        return []

    num_frames = min(num_frames, frame_count)

    if frame_count > 2 * margin_frames + num_frames:
        start = margin_frames
        end = frame_count - margin_frames - 1
    else:
        start = 0
        end = frame_count - 1

    if num_frames == 1:
        return [int(round((start + end) / 2))]

    raw_indices = np.linspace(start, end, num=num_frames)
    indices = [int(round(i)) for i in raw_indices]

    # Guarantee uniqueness while preserving order
    unique_indices: List[int] = []
    seen = set()
    for idx in indices:
        idx = max(0, min(frame_count - 1, idx))
        if idx not in seen:
            seen.add(idx)
            unique_indices.append(idx)

    if len(unique_indices) < num_frames:
        all_candidates = list(range(start, end + 1))
        for candidate in all_candidates:
            if candidate not in seen:
                unique_indices.append(candidate)
                seen.add(candidate)
            if len(unique_indices) >= num_frames:
                break
        unique_indices.sort()

    return unique_indices[:num_frames]


def build_output_path(
    output_dir: Path,
    video_stem: str,
    frame_index: int,
    image_format: str,
) -> Path:
    ext = "jpg" if image_format in {"jpg", "jpeg"} else "png"
    filename = f"{video_stem}_frame_{frame_index:04d}.{ext}"
    return output_dir / filename


def save_frame(
    frame: np.ndarray,
    output_path: Path,
    image_format: str,
    jpg_quality: int,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if image_format == "png":
        params = [cv2.IMWRITE_PNG_COMPRESSION, 3]
        ext = ".png"
    else:
        params = [cv2.IMWRITE_JPEG_QUALITY, jpg_quality]
        ext = ".jpg"

    if output_path.suffix.lower() != ext:
        output_path = output_path.with_suffix(ext)

    if not cv2.imwrite(str(output_path), frame, params):
        raise RuntimeError(f"Failed to write frame: {output_path}")


class VideoFrameExtractor:
    """Extract uniformly sampled frames from a folder of videos."""

    def __init__(self, config: FrameExtractionConfig) -> None:
        self.config = config

    def run(self) -> Tuple[List[ExtractionResult], int]:
        self.config.output_dir.mkdir(parents=True, exist_ok=True)

        videos = discover_videos(self.config.input_dir, self.config.video_names)
        if not videos:
            raise FileNotFoundError(
                f"No video files found in {self.config.input_dir} "
                f"(supported: {', '.join(sorted(VIDEO_EXTENSIONS))})"
            )

        logging.info("Found %d video(s) in %s", len(videos), self.config.input_dir)
        video_infos = collect_video_info(videos)
        budgets = distribute_frame_budget(video_infos, self.config.target_frames)

        total_saved = 0
        results: List[ExtractionResult] = []

        video_iter: Iterable[Tuple[VideoInfo, int]] = zip(video_infos, budgets)
        if tqdm is not None:
            video_iter = tqdm(
                list(zip(video_infos, budgets)),
                desc="Extracting frames",
                unit="video",
            )

        for info, budget in video_iter:
            result = self._extract_from_video(info, budget)
            results.append(result)
            total_saved += result.saved_frames

        logging.info(
            "Done. Saved %d frame(s) to %s",
            total_saved,
            self.config.output_dir,
        )
        return results, total_saved

    def _extract_from_video(
        self,
        info: VideoInfo,
        num_frames: int,
    ) -> ExtractionResult:
        stem = sanitize_stem(info.path.stem)
        indices = compute_uniform_frame_indices(
            frame_count=info.frame_count,
            num_frames=num_frames,
            margin_frames=self.config.margin_frames,
        )

        logging.info(
            "Processing %s | total_frames=%d | extracting=%d",
            info.path.name,
            info.frame_count,
            len(indices),
        )

        cap = cv2.VideoCapture(str(info.path))
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {info.path}")

        saved_paths: List[Path] = []
        frame_iter: Iterable[int] = indices
        if tqdm is not None and len(indices) > 1:
            frame_iter = tqdm(
                indices,
                desc=f"{info.path.name}",
                leave=False,
                unit="frame",
            )

        for frame_idx in frame_iter:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ok, frame = cap.read()
            if not ok or frame is None:
                logging.warning(
                    "Could not read frame %d from %s; skipping",
                    frame_idx,
                    info.path.name,
                )
                continue

            output_path = build_output_path(
                self.config.output_dir,
                stem,
                frame_idx,
                self.config.image_format,
            )
            save_frame(
                frame,
                output_path,
                self.config.image_format,
                self.config.jpg_quality,
            )
            saved_paths.append(output_path)

        cap.release()

        return ExtractionResult(
            video_path=info.path,
            requested_frames=num_frames,
            saved_frames=len(saved_paths),
            frame_indices=indices,
            output_paths=saved_paths,
        )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract uniformly sampled frames from videos for U-Net labeling."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help=f"Input directory containing video files (default: {DEFAULT_INPUT_DIR})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory for extracted frames (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--videos",
        nargs="*",
        default=None,
        metavar="NAME",
        help=(
            "Specific video filenames inside --input. "
            f"Default: {len(PROJECT_VIDEOS)} project videos in assests/"
        ),
    )
    parser.add_argument(
        "--all-videos",
        action="store_true",
        help="Process every supported video in --input instead of the project list",
    )
    parser.add_argument(
        "--target-frames",
        type=int,
        default=DEFAULT_TARGET_FRAMES,
        help=(
            "Total number of frames to extract across all videos "
            f"(recommended {DEFAULT_MIN_FRAMES}-{DEFAULT_MAX_FRAMES}, default: {DEFAULT_TARGET_FRAMES})"
        ),
    )
    parser.add_argument(
        "--format",
        dest="image_format",
        choices=["jpg", "jpeg", "png"],
        default="jpg",
        help="Output image format (default: jpg)",
    )
    parser.add_argument(
        "--jpg-quality",
        type=int,
        default=95,
        help="JPEG quality 1-100 when --format jpg (default: 95)",
    )
    parser.add_argument(
        "--margin-frames",
        type=int,
        default=5,
        help="Skip this many frames at start/end when possible (default: 5)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    setup_logging(verbose=args.verbose)

    if args.all_videos:
        video_names = None
    elif args.videos is not None:
        video_names = args.videos
    else:
        video_names = list(PROJECT_VIDEOS)

    config = FrameExtractionConfig(
        input_dir=args.input,
        output_dir=args.output,
        target_frames=args.target_frames,
        image_format=args.image_format,
        jpg_quality=args.jpg_quality,
        margin_frames=args.margin_frames,
        video_names=video_names,
    )

    extractor = VideoFrameExtractor(config)

    try:
        results, total_saved = extractor.run()
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        logging.error("%s", exc)
        return 1

    logging.info("Summary:")
    for result in results:
        logging.info(
            "  %s -> saved %d/%d frames",
            result.video_path.name,
            result.saved_frames,
            result.requested_frames,
        )

    if total_saved == 0:
        logging.error("No frames were saved.")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
