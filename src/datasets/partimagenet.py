"""PartImageNet segmentation adapter."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from .base import BaseHierarchicalDataset, HierarchicalSample, HierarchyNode, ObjectInstance


class PartImageNetDataset(BaseHierarchicalDataset):
    """Load PartImageNet whole-object and part masks into the common format.

    The segmentation release stores RGB images in ``images/{split}``, part
    masks in ``annotations/{split}``, and whole-object masks in
    ``annotations/{split}_whole``. Pixel values are category IDs; the most
    frequent value in each mask is the background value.
    """

    def __init__(self, root: str | Path, split: str = "train", transform: Any = None, load_images: bool = True) -> None:
        super().__init__(transform=transform)
        self.root = Path(root)
        self.split = split
        self.load_images = load_images
        self.image_root = self.root / "images" / split
        self.part_root = self.root / "annotations" / split
        self.whole_root = self.root / "annotations" / f"{split}_whole"
        self.image_paths = tuple(sorted(self.image_root.glob("*")))

    def __len__(self) -> int:
        return len(self.image_paths)

    def get_sample(self, index: int) -> HierarchicalSample:
        image_path = self.image_paths[index]
        image_id = image_path.stem
        part_mask = self._read_mask(self.part_root / f"{image_id}.png")
        whole_mask = self._read_mask(self.whole_root / f"{image_id}.png")
        image = self._load_image(image_path) if self.load_images else None
        object_mask = self._foreground_mask(whole_mask)
        parts = tuple(self._part_nodes(part_mask))
        instance = ObjectInstance(
            instance_id=image_id,
            category_id=self._foreground_value(whole_mask),
            category_name=image_id.split("_")[0],
            bbox=self._bbox(object_mask),
            segmentation=object_mask,
            parts=parts,
            attributes={"split": self.split},
        )
        return HierarchicalSample(image_id, image_path, (instance,), image, {"split": self.split})

    @staticmethod
    def _read_mask(path: Path) -> np.ndarray:
        try:
            from PIL import Image
        except ImportError as error:
            raise ImportError("Loading PartImageNet masks requires Pillow. Install it in env before use.") from error
        if not path.exists():
            raise FileNotFoundError(f"Missing PartImageNet mask: {path}")
        return np.asarray(Image.open(path))

    @staticmethod
    def _background_value(mask: np.ndarray) -> int | None:
        values = Counter(mask.reshape(-1).tolist())
        return max(values, key=values.get) if values else None

    @classmethod
    def _foreground_value(cls, mask: np.ndarray) -> int | None:
        background = cls._background_value(mask)
        values = Counter(mask[mask != background].reshape(-1).tolist())
        return max(values, key=values.get) if values else None

    @classmethod
    def _foreground_mask(cls, mask: np.ndarray) -> np.ndarray:
        background = cls._background_value(mask)
        return mask != background if background is not None else mask.astype(bool)

    @classmethod
    def _part_nodes(cls, mask: np.ndarray) -> list[HierarchyNode]:
        background = cls._background_value(mask)
        nodes = []
        for category_id in np.unique(mask):
            if category_id == background:
                continue
            category_mask = mask == category_id
            nodes.append(HierarchyNode(
                node_id=f"part:{int(category_id)}",
                category_id=int(category_id),
                category_name=str(int(category_id)),
                bbox=cls._bbox(category_mask),
                segmentation=category_mask,
                attributes={"pixel_label": int(category_id)},
            ))
        return nodes

    @staticmethod
    def _bbox(mask: np.ndarray) -> tuple[float, float, float, float] | None:
        if not mask.any():
            return None
        y, x = np.nonzero(mask)
        return float(x.min()), float(y.min()), float(x.max() - x.min() + 1), float(y.max() - y.min() + 1)

    @staticmethod
    def _load_image(path: Path) -> Any:
        try:
            from PIL import Image
        except ImportError as error:
            raise ImportError("Loading PartImageNet images requires Pillow. Install it in env before use.") from error
        return Image.open(path).convert("RGB")