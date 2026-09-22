"""Common representation for hierarchical and part-whole datasets."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

try:
    from torch.utils.data import Dataset as TorchDataset
except ImportError:
    class TorchDataset:
        """Fallback type so annotation parsing works before torch is installed."""

        pass


@dataclass(frozen=True)
class HierarchyNode:
    """One annotated entity in a parent-child hierarchy."""

    node_id: str | int
    category_id: str | int | None = None
    category_name: str | None = None
    parent_id: str | int | None = None
    bbox: tuple[float, float, float, float] | None = None
    segmentation: Any = None
    children: tuple["HierarchyNode", ...] = ()
    attributes: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ObjectInstance:
    """A top-level object and its directly or indirectly nested parts."""

    instance_id: str | int
    category_id: str | int | None = None
    category_name: str | None = None
    bbox: tuple[float, float, float, float] | None = None
    segmentation: Any = None
    parts: tuple[HierarchyNode, ...] = ()
    attributes: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class HierarchicalSample:
    """Dataset-neutral output returned by every hierarchical dataset."""

    image_id: str | int
    image_path: Path
    objects: tuple[ObjectInstance, ...]
    image: Any = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


class BaseHierarchicalDataset(TorchDataset, ABC):
    """Torch Dataset interface with one normalized output type."""

    def __init__(self, transform: Any = None) -> None:
        self.transform = transform

    @abstractmethod
    def __len__(self) -> int:
        """Return the number of images or samples."""

    @abstractmethod
    def get_sample(self, index: int) -> HierarchicalSample:
        """Load one normalized sample before image transforms are applied."""

    def __getitem__(self, index: int) -> HierarchicalSample:
        sample = self.get_sample(index)
        if self.transform is None:
            return sample
        return HierarchicalSample(
            image_id=sample.image_id,
            image_path=sample.image_path,
            objects=sample.objects,
            image=self.transform(sample.image),
            metadata=sample.metadata,
        )

    def __iter__(self):
        for index in range(len(self)):
            yield self[index]


def as_bbox(value: Sequence[float] | None) -> tuple[float, float, float, float] | None:
    """Normalize a COCO-style [x, y, width, height] box when present."""

    if value is None:
        return None
    if len(value) != 4:
        raise ValueError(f"A bounding box must contain four values, got {value!r}")
    return tuple(float(item) for item in value)  # type: ignore[return-value]