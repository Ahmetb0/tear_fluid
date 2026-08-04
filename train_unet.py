"""
Train a U-Net on Label Studio polygon annotations for tear-film particle segmentation.

Usage:
    python train_unet.py
    python train_unet.py --epochs 30 --batch-size 8
    python train_unet.py --regenerate-masks
    python train_unet.py --allow-cpu          # debug only, no GPU
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split

# ---------------------------------------------------------------------------
# Paths & constants  (relative to this script / project root)
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent

DEFAULT_ANNOTATIONS = PROJECT_ROOT / "data" / "annotations.json"
DEFAULT_IMAGES_DIR = PROJECT_ROOT / "data" / "unet_raw_frames"
DEFAULT_MASKS_DIR = PROJECT_ROOT / "data" / "masks"
DEFAULT_MODEL_OUT = PROJECT_ROOT / "unet_tear_film.pth"

IMAGE_SIZE = 256
DEFAULT_EPOCHS = 100         # longer training to overcome class imbalance
DEFAULT_BATCH_SIZE = 4       # safe for laptop GPU VRAM; max recommended = 8
MAX_BATCH_SIZE = 8
BCE_WEIGHT = 0.2
DICE_WEIGHT = 0.8


# ---------------------------------------------------------------------------
# Annotation → mask pipeline
# ---------------------------------------------------------------------------

def extract_image_filename(task: dict) -> str:
    """Resolve local image filename from a Label Studio task entry."""
    upload = task.get("file_upload") or ""
    if upload:
        # Label Studio prefix: "<8-char-hash>-original_filename.jpg"
        if len(upload) > 9 and upload[8] == "-":
            return upload[9:]
        return upload

    image_path = task.get("data", {}).get("image", "")
    name = Path(image_path).name
    if len(name) > 9 and name[8] == "-":
        return name[9:]
    return name


def _percentage_points_to_pixels(
    points: Sequence[Sequence[float]],
    width: int,
    height: int,
) -> np.ndarray:
    """Convert Label Studio percentage polygon points to pixel coordinates."""
    pts = np.array(
        [[p[0] / 100.0 * width, p[1] / 100.0 * height] for p in points],
        dtype=np.float32,
    )
    return np.round(pts).astype(np.int32)


def polygons_from_task(task: dict) -> List[dict]:
    """Extract all polygon annotation dicts from a Label Studio task."""
    polygons: List[dict] = []
    for annotation in task.get("annotations") or []:
        if annotation.get("was_cancelled"):
            continue
        for item in annotation.get("result") or []:
            if item.get("type") not in {"polygonlabels", "polygon"}:
                continue
            value = item.get("value") or {}
            points = value.get("points")
            if not points:
                continue
            polygons.append(
                {
                    "points": points,
                    "width": item.get("original_width"),
                    "height": item.get("original_height"),
                }
            )
    return polygons


def create_binary_mask(
    polygons: Sequence[dict],
    width: int,
    height: int,
) -> np.ndarray:
    """Rasterize polygons into a single binary mask (uint8, 0 or 255)."""
    mask = np.zeros((height, width), dtype=np.uint8)
    for poly in polygons:
        poly_w = int(poly.get("width") or width)
        poly_h = int(poly.get("height") or height)
        pts = _percentage_points_to_pixels(poly["points"], poly_w, poly_h)
        if pts.shape[0] >= 3:
            cv2.fillPoly(mask, [pts], 255)
    return mask


def resize_image_and_mask(
    image: np.ndarray,
    mask: np.ndarray,
    size: int = IMAGE_SIZE,
) -> Tuple[np.ndarray, np.ndarray]:
    """Resize image (linear) and mask (nearest) to size × size."""
    image_resized = cv2.resize(image, (size, size), interpolation=cv2.INTER_LINEAR)
    mask_resized = cv2.resize(mask, (size, size), interpolation=cv2.INTER_NEAREST)
    return image_resized, mask_resized


def generate_masks_from_annotations(
    annotations_path: Path,
    images_dir: Path,
    masks_dir: Path,
    image_size: int = IMAGE_SIZE,
) -> List[Path]:
    """
    Parse Label Studio JSON and write 256×256 binary masks to *masks_dir*.

    Returns list of image paths that were processed successfully.
    """
    masks_dir.mkdir(parents=True, exist_ok=True)

    with annotations_path.open(encoding="utf-8") as f:
        tasks = json.load(f)

    image_index = {p.name.lower(): p for p in images_dir.glob("*") if p.is_file()}
    processed: List[Path] = []
    skipped = 0

    for task in tasks:
        filename = extract_image_filename(task)
        image_path = images_dir / filename
        if not image_path.exists():
            image_path = image_index.get(filename.lower())
        if image_path is None or not image_path.exists():
            logging.warning("Image not found for task: %s", filename)
            skipped += 1
            continue

        image = cv2.imread(str(image_path))
        if image is None:
            logging.warning("Could not read image: %s", image_path)
            skipped += 1
            continue

        img_h, img_w = image.shape[:2]
        polygons = polygons_from_task(task)
        mask = create_binary_mask(polygons, img_w, img_h)
        _, mask = resize_image_and_mask(image, mask, image_size)

        mask_path = masks_dir / f"{image_path.stem}.png"
        cv2.imwrite(str(mask_path), mask)
        processed.append(image_path)

    logging.info(
        "Mask generation done: %d saved, %d skipped -> %s",
        len(processed),
        skipped,
        masks_dir,
    )
    return processed


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class TearFilmSegmentationDataset(Dataset):
    """Pairs images from *images_dir* with binary masks from *masks_dir*."""

    def __init__(
        self,
        image_paths: Sequence[Path],
        masks_dir: Path,
        image_size: int = IMAGE_SIZE,
    ) -> None:
        self.image_paths = list(image_paths)
        self.masks_dir = masks_dir
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        image_path = self.image_paths[idx]
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Failed to read image: {image_path}")

        mask_path = self.masks_dir / f"{image_path.stem}.png"
        if mask_path.exists():
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        else:
            mask = np.zeros(image.shape[:2], dtype=np.uint8)

        image, mask = resize_image_and_mask(image, mask, self.image_size)

        # BGR -> RGB, normalize to [0, 1]
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        image_tensor = torch.from_numpy(image_rgb).permute(2, 0, 1)  # C,H,W

        mask_binary = (mask >= 127).astype(np.float32)
        mask_tensor = torch.from_numpy(mask_binary).unsqueeze(0)  # 1,H,W

        return image_tensor, mask_tensor


# ---------------------------------------------------------------------------
# U-Net
# ---------------------------------------------------------------------------

class DoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class UNet(nn.Module):
    """Standard U-Net for binary segmentation."""

    def __init__(self, in_channels: int = 3, out_channels: int = 1) -> None:
        super().__init__()
        self.down1 = DoubleConv(in_channels, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.down2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.down3 = DoubleConv(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        self.down4 = DoubleConv(256, 512)
        self.pool4 = nn.MaxPool2d(2)

        self.bottleneck = DoubleConv(512, 1024)

        self.up4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec4 = DoubleConv(1024, 512)
        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(256, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(128, 64)

        self.out_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        c1 = self.down1(x)
        c2 = self.down2(self.pool1(c1))
        c3 = self.down3(self.pool2(c2))
        c4 = self.down4(self.pool3(c3))

        bn = self.bottleneck(self.pool4(c4))

        x = self.up4(bn)
        x = self.dec4(torch.cat([x, c4], dim=1))
        x = self.up3(x)
        x = self.dec3(torch.cat([x, c3], dim=1))
        x = self.up2(x)
        x = self.dec2(torch.cat([x, c2], dim=1))
        x = self.up1(x)
        x = self.dec1(torch.cat([x, c1], dim=1))
        return self.out_conv(x)


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

class DiceBCELoss(nn.Module):
    """Combined BCE + Dice loss; higher Dice weight penalises 'all-black' predictions."""

    def __init__(
        self,
        bce_weight: float = BCE_WEIGHT,
        dice_weight: float = DICE_WEIGHT,
        smooth: float = 1.0,
    ) -> None:
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(logits, targets)

        probs = torch.sigmoid(logits)
        probs_flat = probs.view(probs.size(0), -1)
        targets_flat = targets.view(targets.size(0), -1)

        intersection = (probs_flat * targets_flat).sum(dim=1)
        dice_score = (2.0 * intersection + self.smooth) / (
            probs_flat.sum(dim=1) + targets_flat.sum(dim=1) + self.smooth
        )
        dice_loss = 1.0 - dice_score.mean()

        return self.bce_weight * bce + self.dice_weight * dice_loss


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

@dataclass
class TrainConfig:
    annotations: Path = DEFAULT_ANNOTATIONS
    images_dir: Path = DEFAULT_IMAGES_DIR
    masks_dir: Path = DEFAULT_MASKS_DIR
    model_out: Path = DEFAULT_MODEL_OUT
    image_size: int = IMAGE_SIZE
    epochs: int = DEFAULT_EPOCHS
    batch_size: int = DEFAULT_BATCH_SIZE
    lr: float = 1e-4
    val_split: float = 0.2
    num_workers: int = 0
    seed: int = 42
    regenerate_masks: bool = False
    require_cuda: bool = True


def validate_paths(config: TrainConfig) -> None:
    """Ensure dataset paths exist and log resolved absolute locations."""
    paths = {
        "annotations": config.annotations,
        "images_dir": config.images_dir,
        "masks_dir": config.masks_dir,
    }
    for name, path in paths.items():
        resolved = path.resolve()
        logging.info("  %s -> %s", name, resolved)
        if name != "masks_dir" and not path.exists():
            raise FileNotFoundError(f"{name} not found: {resolved}")

    config.masks_dir.mkdir(parents=True, exist_ok=True)

    image_count = len(list(config.images_dir.glob("*.*")))
    logging.info("  images found: %d in %s", image_count, config.images_dir.resolve())
    if image_count == 0:
        raise FileNotFoundError(
            f"No images in {config.images_dir.resolve()}. "
            "Expected Label Studio frames in data/unet_raw_frames/"
        )


def resolve_device(require_cuda: bool = True) -> torch.device:
    """
    Select CUDA GPU when available; refuse CPU training by default.

    CPU training on U-Net would take days on a laptop — use --allow-cpu only
    for debugging without a GPU.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        logging.info("Device: CUDA (%s, %.1f GB VRAM)", gpu_name, vram_gb)
        return device

    if require_cuda:
        raise RuntimeError(
            "CUDA GPU not available. U-Net training on CPU would take days.\n"
            "  • Install NVIDIA drivers + CUDA-enabled PyTorch:\n"
            "    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124\n"
            "  • Debug on CPU only: python train_unet.py --allow-cpu"
        )

    logging.warning(
        "CUDA not available — falling back to CPU (--allow-cpu). "
        "Training will be extremely slow."
    )
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_dataloaders(config: TrainConfig) -> Tuple[DataLoader, DataLoader, int]:
    if config.regenerate_masks or not any(config.masks_dir.glob("*.png")):
        generate_masks_from_annotations(
            config.annotations,
            config.images_dir,
            config.masks_dir,
            config.image_size,
        )

    with config.annotations.open(encoding="utf-8") as f:
        tasks = json.load(f)

    image_paths: List[Path] = []
    for task in tasks:
        filename = extract_image_filename(task)
        path = config.images_dir / filename
        if path.exists():
            image_paths.append(path)

    if not image_paths:
        raise FileNotFoundError(
            f"No matching images found in {config.images_dir}. "
            "Check annotations.json and unet_raw_frames/ filenames."
        )

    dataset = TearFilmSegmentationDataset(
        image_paths, config.masks_dir, config.image_size
    )

    val_size = max(1, int(len(dataset) * config.val_split))
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(config.seed),
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, val_loader, len(dataset)


def train_model(config: TrainConfig) -> None:
    set_seed(config.seed)

    logging.info("Project root: %s", PROJECT_ROOT)
    logging.info("Validating paths...")
    validate_paths(config)

    device = resolve_device(require_cuda=config.require_cuda)

    logging.info(
        "Training config: epochs=%d, batch_size=%d, image_size=%d",
        config.epochs,
        config.batch_size,
        config.image_size,
    )

    train_loader, val_loader, num_samples = build_dataloaders(config)
    logging.info(
        "Dataset: %d samples | train batches: %d | val batches: %d",
        num_samples,
        len(train_loader),
        len(val_loader),
    )

    model = UNet(in_channels=3, out_channels=1).to(device)
    criterion = DiceBCELoss(bce_weight=BCE_WEIGHT, dice_weight=DICE_WEIGHT)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=5,
        min_lr=1e-6,
    )

    logging.info(
        "Loss weights: BCE=%.1f, Dice=%.1f | LR scheduler: ReduceLROnPlateau(patience=5)",
        BCE_WEIGHT,
        DICE_WEIGHT,
    )

    best_val_loss = float("inf")

    for epoch in range(1, config.epochs + 1):
        model.train()
        train_loss = 0.0
        for images, masks in train_loader:
            images = images.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, masks)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= max(len(train_loader), 1)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(device)
                masks = masks.to(device)
                logits = model(images)
                val_loss += criterion(logits, masks).item()

        val_loss /= max(len(val_loader), 1)
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]
        logging.info(
            "Epoch %03d/%d | train_loss=%.4f | val_loss=%.4f | lr=%.2e",
            epoch,
            config.epochs,
            train_loss,
            val_loss,
            current_lr,
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "val_loss": val_loss,
                    "image_size": config.image_size,
                    "in_channels": 3,
                    "out_channels": 1,
                },
                config.model_out,
            )

    logging.info("Training complete. Best val_loss=%.4f", best_val_loss)
    logging.info("Model saved to: %s", config.model_out.resolve())


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(
        description="Train U-Net on Label Studio tear-film annotations."
    )
    parser.add_argument("--annotations", type=Path, default=DEFAULT_ANNOTATIONS)
    parser.add_argument("--images-dir", type=Path, default=DEFAULT_IMAGES_DIR)
    parser.add_argument("--masks-dir", type=Path, default=DEFAULT_MASKS_DIR)
    parser.add_argument("--model-out", type=Path, default=DEFAULT_MODEL_OUT)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS,
                        help=f"Training epochs (default: {DEFAULT_EPOCHS})")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE,
                        help=f"Batch size, max {MAX_BATCH_SIZE} for laptop VRAM (default: {DEFAULT_BATCH_SIZE})")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--allow-cpu",
        action="store_true",
        help="Allow CPU training (very slow; for debugging only)",
    )
    parser.add_argument(
        "--regenerate-masks",
        action="store_true",
        help="Force regeneration of binary masks from annotations.json",
    )
    args = parser.parse_args()

    batch_size = min(max(1, args.batch_size), MAX_BATCH_SIZE)

    return TrainConfig(
        annotations=args.annotations,
        images_dir=args.images_dir,
        masks_dir=args.masks_dir,
        model_out=args.model_out,
        epochs=args.epochs,
        batch_size=batch_size,
        lr=args.lr,
        val_split=args.val_split,
        seed=args.seed,
        regenerate_masks=args.regenerate_masks,
        require_cuda=not args.allow_cpu,
    )


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    config = parse_args()
    train_model(config)


if __name__ == "__main__":
    main()
