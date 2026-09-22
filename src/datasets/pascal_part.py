"""PASCAL-Part adapter for VOC2010 images and MATLAB annotations."""

from __future__ import annotations

from collections.abc import Mapping as MappingABC
from pathlib import Path
from typing import Any

import numpy as np

from .base import BaseHierarchicalDataset, HierarchicalSample, HierarchyNode, ObjectInstance


class PascalPartDataset(BaseHierarchicalDataset):
    """Load PASCAL-Part annotations into the common object-to-parts format."""

    def __init__(
        self,
        root: str | Path,
        annotations: str | Path,
        split: str = "trainval",
        transform: Any = None,
        load_images: bool = True,
    ) -> None:
        super().__init__(transform=transform)
        self.root = Path(root)
        self.annotation_root = self._resolve_annotation_root(Path(annotations))
        self.split = split
        self.load_images = load_images
        self.annotation_files = tuple(self._annotation_files(split))

    def __len__(self) -> int:
        return len(self.annotation_files)

    def get_sample(self, index: int) -> HierarchicalSample:
        annotation_path = self.annotation_files[index]
        payload = self._read_mat(annotation_path)
        annotation = payload["anno"]
        image_id = self._string_field(annotation, "imname", 0) or annotation_path.stem
        image_path = self._image_path(image_id)
        image = self._load_image(image_path) if self.load_images else None
        objects = tuple(self._objects(annotation, image_id))
        return HierarchicalSample(
            image_id=image_id,
            image_path=image_path,
            objects=objects,
            image=image,
            metadata={"annotation_path": annotation_path, "split": self.split},
        )

    def _annotation_files(self, split: str) -> list[Path]:
        files_by_id = {path.stem: path for path in sorted(self.annotation_root.glob("*.mat"))}
        image_ids = self._split_image_ids(split)
        if image_ids is None:
            return list(files_by_id.values())
        return [files_by_id[image_id] for image_id in image_ids if image_id in files_by_id]

    def _split_image_ids(self, split: str) -> list[str] | None:
        if split == "all":
            return None
        image_set_path = self.root.parent / "ImageSets" / "Main" / f"{split}.txt"
        if not image_set_path.exists():
            return None
        return [
            line.split()[0]
            for line in image_set_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    def _image_path(self, image_id: str) -> Path:
        path = Path(image_id)
        if path.suffix == "":
            path = path.with_suffix(".jpg")
        return self.root / path

    def _load_image(self, image_path: Path) -> Any:
        try:
            from PIL import Image
        except ImportError as error:
            raise ImportError("Loading PASCAL-Part images requires Pillow. Install it in env before use.") from error
        return Image.open(image_path).convert("RGB")

    @staticmethod
    def _read_mat(path: Path) -> MappingABC[str, Any]:
        try:
            from scipy.io import loadmat
        except ImportError as error:
            raise ImportError("Loading PASCAL-Part annotations requires scipy. Install it in env before use.") from error
        return loadmat(path, squeeze_me=True, struct_as_record=False)

    def _objects(self, annotation: Any, image_id: str) -> list[ObjectInstance]:
        objects = []
        for object_index, object_annotation in enumerate(self._sequence(self._field(annotation, "objects", 1))):
            object_id = f"{image_id}:object:{object_index}"
            object_mask = self._mask(self._field(object_annotation, "mask", 2))
            parts = tuple(self._parts(object_annotation, object_id))
            if object_mask is None and parts:
                object_mask = self._union_masks(part.segmentation for part in parts)
            category_name = self._string_field(object_annotation, "class", 0)
            category_id = self._scalar_field(object_annotation, "class_ind", 1)
            objects.append(ObjectInstance(
                instance_id=object_id,
                category_id=category_id,
                category_name=category_name,
                bbox=self._bbox_from_mask(object_mask),
                segmentation=object_mask,
                parts=parts,
                attributes={"source_index": object_index},
            ))
        return objects

    def _parts(self, object_annotation: Any, object_id: str) -> list[HierarchyNode]:
        nodes = []
        for part_index, part_annotation in enumerate(self._sequence(self._field(object_annotation, "parts", 3))):
            mask = self._mask(self._field(part_annotation, "mask", 1))
            part_name = self._string_field(part_annotation, "part_name", 0)
            node_id = f"{object_id}:part:{part_index}"
            nodes.append(HierarchyNode(
                node_id=node_id,
                category_id=part_name,
                category_name=part_name,
                parent_id=object_id,
                bbox=self._bbox_from_mask(mask),
                segmentation=mask,
                attributes={"source_index": part_index},
            ))
        return nodes

    def _resolve_annotation_root(self, annotation_path: Path) -> Path:
        if not annotation_path.is_absolute():
            project_relative_path = Path.cwd() / annotation_path
            annotation_path = project_relative_path if project_relative_path.exists() else self.root.parent / annotation_path
        nested_annotation_root = annotation_path / "Annotations_Part"
        if nested_annotation_root.is_dir():
            return nested_annotation_root
        return annotation_path

    @staticmethod
    def _field(value: Any, name: str, index: int | None = None) -> Any:
        if isinstance(value, MappingABC) and name in value:
            return value[name]
        if hasattr(value, name):
            return getattr(value, name)
        if isinstance(value, np.void) and value.dtype.names and name in value.dtype.names:
            return value[name]
        if index is not None:
            if isinstance(value, (list, tuple)) and len(value) > index:
                return value[index]
            if isinstance(value, np.ndarray) and value.dtype == object and value.size > index:
                return value.reshape(-1)[index]
        return None

    @classmethod
    def _string_field(cls, value: Any, name: str, index: int | None = None) -> str | None:
        field = cls._unwrap(cls._field(value, name, index))
        if field is None:
            return None
        if isinstance(field, bytes):
            return field.decode("utf-8")
        if isinstance(field, np.ndarray):
            if field.size == 0:
                return None
            if field.dtype.kind in ("S", "U"):
                return "".join(str(item) for item in field.reshape(-1)).strip()
            field = cls._unwrap(field)
        return str(field)

    @classmethod
    def _scalar_field(cls, value: Any, name: str, index: int | None = None) -> str | int | float | None:
        field = cls._unwrap(cls._field(value, name, index))
        if isinstance(field, np.generic):
            field = field.item()
        if isinstance(field, (str, int, float)):
            return field
        return None

    @classmethod
    def _mask(cls, value: Any) -> np.ndarray | None:
        mask = cls._unwrap(value)
        if not isinstance(mask, np.ndarray):
            mask = np.asarray(mask)
        if mask.size == 0:
            return None
        mask = np.squeeze(mask)
        if mask.ndim != 2:
            return None
        return mask.astype(bool)

    @staticmethod
    def _sequence(value: Any) -> list[Any]:
        if value is None:
            return []
        if isinstance(value, (list, tuple)):
            return list(value)
        if isinstance(value, np.ndarray):
            if value.size == 0:
                return []
            if value.dtype.kind in ("O", "V") or value.dtype.names:
                return list(value.reshape(-1))
        return [value]

    @classmethod
    def _unwrap(cls, value: Any) -> Any:
        while isinstance(value, np.ndarray) and value.size == 1 and value.dtype.kind in ("O", "V"):
            value = value.reshape(-1)[0]
        return value

    @staticmethod
    def _bbox_from_mask(mask: np.ndarray | None) -> tuple[float, float, float, float] | None:
        if mask is None or not mask.any():
            return None
        y_coords, x_coords = np.nonzero(mask)
        x_min = int(x_coords.min())
        y_min = int(y_coords.min())
        width = int(x_coords.max() - x_min + 1)
        height = int(y_coords.max() - y_min + 1)
        return (float(x_min), float(y_min), float(width), float(height))

    @staticmethod
    def _union_masks(masks: Any) -> np.ndarray | None:
        union = None
        for mask in masks:
            if not isinstance(mask, np.ndarray):
                continue
            union = mask.copy() if union is None else union | mask
        return union
