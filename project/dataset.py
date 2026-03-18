import os
from pathlib import Path
from typing import List, Tuple

import glymur
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset


IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def _load_rgb_image(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".jp2":
        arr = np.array(glymur.Jp2k(str(path))[:], dtype=np.uint8)
        if arr.ndim == 2:
            arr = np.stack([arr, arr, arr], axis=-1)
        return arr
    with Image.open(path) as img:
        return np.array(img.convert("RGB"), dtype=np.uint8)


def _to_tensor(arr: np.ndarray) -> torch.Tensor:
    # Convert HWC uint8 [0,255] -> CHW float32 [0,1].
    return torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0


class J2KPairDataset(Dataset):
    def __init__(
        self,
        data_root: str,
        split: str,
        patch_size: int = 128,
        train: bool = True,
    ) -> None:
        self.data_root = Path(data_root)
        self.split = split
        self.patch_size = patch_size
        self.train = train

        split_dir = self.data_root / split
        self.comp_dir = split_dir / "compressed"
        self.gt_dir = split_dir / "ground_truth"

        if not self.comp_dir.exists() or not self.gt_dir.exists():
            raise FileNotFoundError(
                f"Expected directories not found: {self.comp_dir} and/or {self.gt_dir}"
            )

        gt_lookup = {}
        for gt_path in self.gt_dir.iterdir():
            if gt_path.suffix.lower() in IMG_EXTS:
                gt_lookup[gt_path.stem] = gt_path

        self.pairs: List[Tuple[Path, Path]] = []
        for comp_path in sorted(self.comp_dir.glob("*.jp2")):
            gt_path = gt_lookup.get(comp_path.stem)
            if gt_path is not None:
                self.pairs.append((comp_path, gt_path))

        if not self.pairs:
            raise RuntimeError(
                f"No compressed/ground-truth pairs found in split '{split}' under {self.data_root}"
            )

    def __len__(self) -> int:
        return len(self.pairs)

    def _paired_crop(self, comp: np.ndarray, gt: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        h = min(comp.shape[0], gt.shape[0])
        w = min(comp.shape[1], gt.shape[1])
        comp = comp[:h, :w]
        gt = gt[:h, :w]

        if h < self.patch_size or w < self.patch_size:
            # If image is too small, center-pad to patch size.
            out_c = np.zeros((self.patch_size, self.patch_size, 3), dtype=np.uint8)
            out_g = np.zeros((self.patch_size, self.patch_size, 3), dtype=np.uint8)
            y0 = (self.patch_size - h) // 2
            x0 = (self.patch_size - w) // 2
            out_c[y0 : y0 + h, x0 : x0 + w] = comp
            out_g[y0 : y0 + h, x0 : x0 + w] = gt
            return out_c, out_g

        if self.train:
            y = torch.randint(0, h - self.patch_size + 1, (1,)).item()
            x = torch.randint(0, w - self.patch_size + 1, (1,)).item()
        else:
            y = (h - self.patch_size) // 2
            x = (w - self.patch_size) // 2

        comp_crop = comp[y : y + self.patch_size, x : x + self.patch_size]
        gt_crop = gt[y : y + self.patch_size, x : x + self.patch_size]
        return comp_crop, gt_crop

    def __getitem__(self, index: int):
        comp_path, gt_path = self.pairs[index]
        comp = _load_rgb_image(comp_path)
        gt = _load_rgb_image(gt_path)

        comp, gt = self._paired_crop(comp, gt)

        if self.train and torch.rand(1).item() < 0.5:
            comp = np.flip(comp, axis=1).copy()
            gt = np.flip(gt, axis=1).copy()
        if self.train and torch.rand(1).item() < 0.5:
            comp = np.flip(comp, axis=0).copy()
            gt = np.flip(gt, axis=0).copy()

        return {
            "input": _to_tensor(comp),
            "target": _to_tensor(gt),
            "name": comp_path.stem,
        }
