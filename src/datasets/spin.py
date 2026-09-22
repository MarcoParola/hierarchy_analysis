"""SPIN (SubPartImageNet) hierarchical segmentation adapter."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping

from .base import BaseHierarchicalDataset, HierarchicalSample, HierarchyNode, ObjectInstance, as_bbox


class SPINDataset(BaseHierarchicalDataset):
    """Load SPIN whole, part, and subpart COCO annotation files.

    SPIN publishes one COCO file per granularity. When explicit parent IDs are
    absent, this adapter preserves all annotations for the image and links each
    child to the same-image parent with the strongest mask or box overlap.
    """

    def __init__(self, root: str | Path, annotations: str | Path, split: str = "test", transform: Any = None, load_images: bool = True) -> None:
        super().__init__(transform=transform)
        self.root = Path(root)
        self.annotation_root = Path(annotations)
        if not self.annotation_root.is_absolute():
            self.annotation_root = Path.cwd() / self.annotation_root
        nested_annotation_root = self.annotation_root / "spin_jsons_for_coco"
        if nested_annotation_root.is_dir():
            self.annotation_root = nested_annotation_root
        self.split = split
        self.load_images = load_images
        self.levels = {
            level: self._read_json(self.annotation_root / f"spin_{split}_{level}s.json")
            for level in ("whole", "part", "subpart")
        }
        self.images = tuple(self.levels["subpart"].get("images", []))
        self.annotations_by_level: dict[str, dict[Any, list[Mapping[str, Any]]]] = {}
        for level, payload in self.levels.items():
            by_image: dict[Any, list[Mapping[str, Any]]] = defaultdict(list)
            for annotation in payload.get("annotations", []):
                by_image[annotation["image_id"]].append(annotation)
            self.annotations_by_level[level] = by_image
        self.categories = {
            level: {item["id"]: item for item in payload.get("categories", [])}
            for level, payload in self.levels.items()
        }

    @staticmethod
    def _read_json(path: Path) -> Mapping[str, Any]:
        with path.open(encoding="utf-8") as stream:
            return json.load(stream)

    def __len__(self) -> int:
        return len(self.images)

    def get_sample(self, index: int) -> HierarchicalSample:
        image_info = self.images[index]
        image_path = self._image_path(image_info["file_name"])
        image = self._load_image(image_path) if self.load_images else None
        objects = self._build_objects(image_info["id"])
        metadata = {key: value for key, value in image_info.items() if key != "file_name"}
        return HierarchicalSample(image_info["id"], image_path, tuple(objects), image, metadata)

    def _image_path(self, file_name: str) -> Path:
        path = Path(file_name)
        if path.suffix == "":
            path = path.with_suffix(".JPEG")
        return self.root / self.split / path

    def _load_image(self, image_path: Path) -> Any:
        try:
            from PIL import Image
        except ImportError as error:
            raise ImportError("Loading SPIN images requires Pillow. Install it in env before use.") from error
        return Image.open(image_path).convert("RGB")

    def _build_objects(self, image_id: Any) -> list[ObjectInstance]:
        whole = self.annotations_by_level["whole"].get(image_id, [])
        parts = self.annotations_by_level["part"].get(image_id, [])
        subparts = self.annotations_by_level["subpart"].get(image_id, [])
        parts_by_object = self._assign_children(whole, parts)
        subparts_by_part = self._assign_children(parts, subparts)
        objects = []
        for object_annotation in whole:
            object_id = object_annotation.get("id")
            object_parts = parts_by_object.get(object_id, [])
            object_part_nodes = [
                self._node("part", part, subparts_by_part.get(part.get("id"), []), parent_id=object_id)
                for part in object_parts
            ]
            category = self.categories["whole"].get(object_annotation.get("category_id"), {})
            objects.append(ObjectInstance(
                instance_id=object_annotation.get("id"),
                category_id=object_annotation.get("category_id"),
                category_name=self._category_name(category),
                bbox=as_bbox(object_annotation.get("bbox")),
                segmentation=object_annotation.get("segmentation"),
                parts=tuple(object_part_nodes),
                attributes=object_annotation,
            ))
        return objects

    def _node(
        self,
        level: str,
        annotation: Mapping[str, Any],
        children: list[Mapping[str, Any]],
        parent_id: Any = None,
    ) -> HierarchyNode:
        category = self.categories[level].get(annotation.get("category_id"), {})
        child_level = "subpart" if level == "part" else None
        return HierarchyNode(
            node_id=annotation.get("id"),
            category_id=annotation.get("category_id"),
            category_name=self._category_name(category),
            parent_id=self._parent_id(annotation, parent_id),
            bbox=as_bbox(annotation.get("bbox")),
            segmentation=annotation.get("segmentation"),
            children=tuple(
                self._node(child_level, child, [], parent_id=annotation.get("id"))
                for child in children
            ) if child_level else (),
            attributes=annotation,
        )

    @staticmethod
    def _category_name(category: Mapping[str, Any]) -> str | None:
        name = category.get("name")
        return None if name is None else str(name)

    def _assign_children(
        self,
        parents: list[Mapping[str, Any]],
        children: list[Mapping[str, Any]],
    ) -> dict[Any, list[Mapping[str, Any]]]:
        assigned: dict[Any, list[Mapping[str, Any]]] = defaultdict(list)
        if not parents:
            return assigned
        if len(parents) == 1:
            assigned[parents[0].get("id")].extend(children)
            return assigned

        parents_by_id = {parent.get("id"): parent for parent in parents}
        for child in children:
            explicit_parent_id = self._parent_id(child)
            if explicit_parent_id in parents_by_id:
                assigned[explicit_parent_id].append(child)
                continue
            parent = max(parents, key=lambda item: self._overlap_score(child, item))
            assigned[parent.get("id")].append(child)
        return assigned

    @staticmethod
    def _parent_id(annotation: Mapping[str, Any], default: Any = None) -> Any:
        for key in ("parent_id", "whole_id", "part_id", "obj_id"):
            if annotation.get(key) is not None:
                return annotation[key]
        return default

    def _overlap_score(self, annotation: Mapping[str, Any], parent: Mapping[str, Any]) -> float:
        mask_overlap = self._mask_overlap(annotation, parent)
        if mask_overlap is not None:
            return mask_overlap
        return self._bbox_overlap(annotation.get("bbox"), parent.get("bbox"))

    @staticmethod
    def _mask_overlap(annotation: Mapping[str, Any], parent: Mapping[str, Any]) -> float | None:
        annotation_segmentation = annotation.get("segmentation")
        parent_segmentation = parent.get("segmentation")
        if not isinstance(annotation_segmentation, dict) or not isinstance(parent_segmentation, dict):
            return None
        try:
            from pycocotools import mask as mask_utils
        except ImportError:
            return None
        annotation_mask = mask_utils.decode(annotation_segmentation).astype(bool)
        parent_mask = mask_utils.decode(parent_segmentation).astype(bool)
        if annotation_mask.shape != parent_mask.shape:
            return None
        return float((annotation_mask & parent_mask).sum())

    @staticmethod
    def _bbox_overlap(bbox: Any, parent_bbox: Any) -> float:
        if not bbox or not parent_bbox:
            return 0.0
        x, y, width, height = bbox
        px, py, pwidth, pheight = parent_bbox
        overlap_width = max(0.0, min(x + width, px + pwidth) - max(x, px))
        overlap_height = max(0.0, min(y + height, py + pheight) - max(y, py))
        return float(overlap_width * overlap_height)
