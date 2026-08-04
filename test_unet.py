"""
Run inference with the trained U-Net and save a comparison figure.

Usage:
    python test_unet.py
    python test_unet.py --num-samples 5 --seed 42
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from train_unet import (
    DEFAULT_IMAGES_DIR,
    DEFAULT_MASKS_DIR,
    DEFAULT_MODEL_OUT,
    IMAGE_SIZE,
    PROJECT_ROOT,
    UNet,
)

DEFAULT_OUTPUT = PROJECT_ROOT / "test_sonuclari.png"
THRESHOLD = 0.2


def load_model(checkpoint_path: Path, device: torch.device) -> UNet:
    """Load trained U-Net weights from a .pth checkpoint."""
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Model not found: {checkpoint_path.resolve()}")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    in_channels = checkpoint.get("in_channels", 3)
    out_channels = checkpoint.get("out_channels", 1)

    model = UNet(in_channels=in_channels, out_channels=out_channels)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def preprocess_image(image_bgr: np.ndarray, size: int = IMAGE_SIZE) -> tuple[np.ndarray, torch.Tensor]:
    """Resize and normalize image for inference; return display RGB and tensor."""
    image_resized = cv2.resize(image_bgr, (size, size), interpolation=cv2.INTER_LINEAR)
    image_rgb = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)

    tensor = torch.from_numpy(image_rgb.astype(np.float32) / 255.0)
    tensor = tensor.permute(2, 0, 1).unsqueeze(0)  # 1,C,H,W
    return image_rgb, tensor


def load_ground_truth_mask(mask_path: Path, size: int = IMAGE_SIZE) -> np.ndarray:
    """Load binary ground-truth mask resized to *size* × *size*."""
    if not mask_path.exists():
        return np.zeros((size, size), dtype=np.uint8)

    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return np.zeros((size, size), dtype=np.uint8)

    mask = cv2.resize(mask, (size, size), interpolation=cv2.INTER_NEAREST)
    return (mask >= 127).astype(np.uint8) * 255


@torch.no_grad()
def predict(
    model: UNet,
    image_tensor: torch.Tensor,
    device: torch.device,
    threshold: float = THRESHOLD,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Run inference; return binary mask (uint8) and probability heatmap (float32, 0–1).
    """
    image_tensor = image_tensor.to(device)
    logits = model(image_tensor)
    probs = torch.sigmoid(logits).squeeze().cpu().numpy()

    binary = (probs >= threshold).astype(np.uint8) * 255
    return binary, probs.astype(np.float32)


def select_random_images(images_dir: Path, num_samples: int, seed: int) -> list[Path]:
    """Pick *num_samples* random image paths from *images_dir*."""
    extensions = {".jpg", ".jpeg", ".png", ".bmp"}
    candidates = sorted(
        p for p in images_dir.iterdir()
        if p.is_file() and p.suffix.lower() in extensions
    )
    if not candidates:
        raise FileNotFoundError(f"No images found in {images_dir.resolve()}")

    rng = random.Random(seed)
    k = min(num_samples, len(candidates))
    return rng.sample(candidates, k)


def create_comparison_figure(
    samples: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]],
    output_path: Path,
) -> None:
    """
    Save a grid figure: each row = original | ground truth | prediction | prob heatmap.

    *samples* items are (original_rgb, gt_mask, pred_mask, prob_map, title).
    """
    n_rows = len(samples)
    fig, axes = plt.subplots(n_rows, 4, figsize=(16, 4 * n_rows))
    if n_rows == 1:
        axes = np.expand_dims(axes, axis=0)

    col_titles = [
        "Orijinal Görüntü",
        "Gerçek Maske (Label Studio)",
        "Model Tahmini",
        "Olasılık Haritası",
    ]

    for row, (original, gt_mask, pred_mask, prob_map, title) in enumerate(samples):
        panels = [
            (original, None, {}),
            (gt_mask, "gray", {"vmin": 0, "vmax": 255}),
            (pred_mask, "gray", {"vmin": 0, "vmax": 255}),
            (prob_map, "viridis", {"vmin": 0.0, "vmax": 1.0}),
        ]

        for col, (data, cmap, kwargs) in enumerate(panels):
            ax = axes[row, col]
            if cmap is None:
                ax.imshow(data)
            else:
                im = ax.imshow(data, cmap=cmap, **kwargs)
                if col == 3:
                    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            ax.axis("off")
            if row == 0:
                ax.set_title(col_titles[col], fontsize=11, fontweight="bold")
        axes[row, 0].set_ylabel(title, fontsize=9, rotation=0, labelpad=90, va="center")

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def run_inference(
    model_path: Path = DEFAULT_MODEL_OUT,
    images_dir: Path = DEFAULT_IMAGES_DIR,
    masks_dir: Path = DEFAULT_MASKS_DIR,
    output_path: Path = DEFAULT_OUTPUT,
    num_samples: int = 5,
    seed: int = 42,
    threshold: float = THRESHOLD,
) -> Path:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = load_model(model_path, device)
    image_paths = select_random_images(images_dir, num_samples, seed)

    samples: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]] = []

    for image_path in image_paths:
        image_bgr = cv2.imread(str(image_path))
        if image_bgr is None:
            print(f"Warning: could not read {image_path.name}, skipping.")
            continue

        original_rgb, tensor = preprocess_image(image_bgr)
        gt_mask = load_ground_truth_mask(masks_dir / f"{image_path.stem}.png")
        pred_mask, prob_map = predict(model, tensor, device, threshold)

        samples.append((original_rgb, gt_mask, pred_mask, prob_map, image_path.name))

    if not samples:
        raise RuntimeError("No samples could be processed.")

    create_comparison_figure(samples, output_path)
    print(f"Saved {len(samples)} comparisons -> {output_path.resolve()}")
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test U-Net on random tear-film frames.")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_OUT)
    parser.add_argument("--images-dir", type=Path, default=DEFAULT_IMAGES_DIR)
    parser.add_argument("--masks-dir", type=Path, default=DEFAULT_MASKS_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--num-samples", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threshold", type=float, default=THRESHOLD)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_inference(
        model_path=args.model,
        images_dir=args.images_dir,
        masks_dir=args.masks_dir,
        output_path=args.output,
        num_samples=args.num_samples,
        seed=args.seed,
        threshold=args.threshold,
    )


if __name__ == "__main__":
    main()
