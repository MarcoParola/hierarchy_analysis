"""PACO adapter for COCO-style image and annotation files."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping

from .base import BaseHierarchicalDataset, HierarchicalSample, HierarchyNode, ObjectInstance, as_bbox


class PACODataset(BaseHierarchicalDataset):
    """Read PACO annotations into the common multi-object hierarchy format."""

    def __init__(self, root: str | Path, annotations: str | Path, transform: Any = None, load_images: bool = True) -> None:
        super().__init__(transform=transform)
        self.root = Path(root)
        annotation_path = Path(annotations)
        if not annotation_path.is_absolute():
            project_relative_path = Path.cwd() / annotation_path
            annotation_path = (
                project_relative_path
                if project_relative_path.exists()
                else self.root / annotation_path
            )
        with annotation_path.open(encoding="utf-8") as stream:
            payload = json.load(stream)

        self.load_images = load_images
        self.categories = {item["id"]: item for item in payload.get("categories", [])}
        self.images = tuple(payload.get("images", []))
        self.annotations_by_image: dict[Any, list[Mapping[str, Any]]] = defaultdict(list)
        for annotation in payload.get("annotations", []):
            self.annotations_by_image[annotation["image_id"]].append(annotation)

    def __len__(self) -> int:
        return len(self.images)

    def get_sample(self, index: int) -> HierarchicalSample:
        image_info = self.images[index]
        image_path = self.root / image_info["file_name"]
        annotations = self.annotations_by_image.get(image_info["id"], [])
        image = self._load_image(image_path) if self.load_images else None
        metadata = {key: value for key, value in image_info.items() if key != "file_name"}
        return HierarchicalSample(
            image_id=image_info["id"],
            image_path=image_path,
            objects=tuple(self._build_objects(annotations)),
            image=image,
            metadata=metadata,
        )

    def _load_image(self, image_path: Path) -> Any:
        try:
            from PIL import Image
        except ImportError as error:
            raise ImportError("Loading PACO images requires Pillow. Install it in env before use.") from error
        return Image.open(image_path).convert("RGB")

    def _build_objects(self, annotations: list[Mapping[str, Any]]) -> list[ObjectInstance]:
        annotation_ids = {item.get("id") for item in annotations}
        children_by_parent: dict[Any, list[Mapping[str, Any]]] = defaultdict(list)
        roots: list[Mapping[str, Any]] = []
        for annotation in annotations:
            parent_id = self._parent_id(annotation)
            if parent_id in annotation_ids:
                children_by_parent[parent_id].append(annotation)
            else:
                roots.append(annotation)

        objects = []
        for root in roots:
            category = self.categories.get(root.get("category_id"), {})
            objects.append(ObjectInstance(
                instance_id=root.get("id"),
                category_id=root.get("category_id"),
                category_name=category.get("name"),
                bbox=as_bbox(root.get("bbox")),
                segmentation=root.get("segmentation"),
                parts=tuple(self._node(part, children_by_parent) for part in children_by_parent.get(root.get("id"), [])),
                attributes=root,
            ))
        return objects

    def _node(
        self,
        annotation: Mapping[str, Any],
        children_by_parent: Mapping[Any, list[Mapping[str, Any]]],
    ) -> HierarchyNode:
        category = self.categories.get(annotation.get("category_id"), {})
        return HierarchyNode(
            node_id=annotation.get("id"),
            category_id=annotation.get("category_id"),
            category_name=category.get("name"),
            parent_id=self._parent_id(annotation),
            bbox=as_bbox(annotation.get("bbox")),
            segmentation=annotation.get("segmentation"),
            children=tuple(
                self._node(child, children_by_parent)
                for child in children_by_parent.get(annotation.get("id"), [])
            ),
            attributes=annotation,
        )

    @staticmethod
    def _parent_id(annotation: Mapping[str, Any]) -> Any:
        for key in ("parent_id", "parent", "part_of"):
            if annotation.get(key) is not None:
                return annotation[key]
        object_annotation_id = annotation.get("obj_ann_id")
        if object_annotation_id is not None and object_annotation_id != annotation.get("id"):
            return object_annotation_id
        return None