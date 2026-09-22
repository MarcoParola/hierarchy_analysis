"""Inspect a hierarchical dataset and save annotated sample images.

Example:
	env/bin/python test/test_dataset.py --root /data/paco --annotations annotations.json
	env/bin/python test/test_dataset.py --dataset spin --root data/spin/images --annotations data/spin/annotations
	env/bin/python test/test_dataset.py --dataset pascal_part --root data/pascal_part/VOCdevkit/VOC2010/JPEGImages --annotations data/pascal_part/Annotations_Part
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
	sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets import HierarchicalSample, HierarchyNode, PACODataset, PascalPartDataset, PartImageNetDataset, SPINDataset


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_spin_category_names_are_normalized_strings(tmp_path: Path) -> None:
    annotation_root = tmp_path / "annotations" / "spin_jsons_for_coco"
    annotation_root.mkdir(parents=True)
    image = {"id": 1, "file_name": "example", "height": 100, "width": 100}

    write_json(
        annotation_root / "spin_test_wholes.json",
        {
            "images": [image],
            "annotations": [
                {"id": 10, "image_id": 1, "category_id": 145, "bbox": [0, 0, 50, 50]},
            ],
            "categories": [{"id": 145, "name": 145}],
        },
    )
    write_json(
        annotation_root / "spin_test_parts.json",
        {
            "images": [image],
            "annotations": [
                {"id": 20, "image_id": 1, "category_id": 2, "bbox": [10, 10, 30, 30]},
                {"id": 21, "image_id": 1, "category_id": 3, "bbox": [60, 60, 10, 10]},
            ],
            "categories": [
                {"id": 2, "name": "Quadruped Foot"},
                {"id": 3, "name": "Quadruped Tail"},
            ],
        },
    )
    write_json(
        annotation_root / "spin_test_subparts.json",
        {
            "images": [image],
            "annotations": [
                {"id": 30, "image_id": 1, "category_id": 21, "bbox": [12, 12, 10, 10]},
                {"id": 31, "image_id": 1, "category_id": 25, "bbox": [62, 62, 5, 5]},
            ],
            "categories": [
                {"id": 21, "name": "Quadruped-Legs-Heel"},
                {"id": 25, "name": "Quadruped-Legs-Wrist/Ankle"},
            ],
        },
    )

    dataset = SPINDataset(tmp_path / "images", tmp_path / "annotations", load_images=False)
    sample = dataset[0]

    assert count_nodes(sample) == (1, 2, 2)
    assert sample.objects[0].category_name == "145"
    assert isinstance(sample.objects[0].category_name, str)
    assert sample.objects[0].parts[0].category_name == "Quadruped Foot"
    assert [part.parent_id for part in sample.objects[0].parts] == [10, 10]
    assert sample.objects[0].parts[1].children[0].parent_id == 21
    assert distinct_object_part_and_subpart_labels(sample) == (
        {"145"},
        {"Quadruped Foot", "Quadruped Tail"},
        {"Quadruped-Legs-Heel", "Quadruped-Legs-Wrist/Ankle"},
    )


def test_pascal_part_dataset_parses_object_and_part_masks(tmp_path: Path, monkeypatch) -> None:
    annotation_root = tmp_path / "Annotations_Part"
    annotation_root.mkdir()
    image_root = tmp_path / "JPEGImages"
    image_root.mkdir()
    annotation_path = annotation_root / "2008_000001.mat"
    annotation_path.touch()

    object_mask = [[False, False, False], [False, True, True], [False, True, False]]
    head_mask = [[False, False, False], [False, True, False], [False, False, False]]
    torso_mask = [[False, False, False], [False, False, True], [False, True, False]]

    def read_mat(path: Path):
        assert path == annotation_path
        return {
            "anno": {
                "imname": "2008_000001",
                "objects": [
                    {
                        "class": "person",
                        "class_ind": 15,
                        "mask": object_mask,
                        "parts": [
                            {"part_name": "head", "mask": head_mask},
                            {"part_name": "torso", "mask": torso_mask},
                        ],
                    }
                ],
            }
        }

    monkeypatch.setattr(PascalPartDataset, "_read_mat", staticmethod(read_mat))

    dataset = PascalPartDataset(image_root, annotation_root, load_images=False)
    sample = dataset[0]

    assert sample.image_id == "2008_000001"
    assert sample.image_path == image_root / "2008_000001.jpg"
    assert count_nodes(sample) == (1, 2, 0)
    assert sample.objects[0].category_name == "person"
    assert sample.objects[0].category_id == 15
    assert sample.objects[0].bbox == (1.0, 1.0, 2.0, 2.0)
    assert [part.category_name for part in sample.objects[0].parts] == ["head", "torso"]
    assert sample.objects[0].parts[0].bbox == (1.0, 1.0, 1.0, 1.0)


def count_nodes(sample: HierarchicalSample) -> tuple[int, int, int]:
    """Return object, direct-part, and subpart counts for one sample."""

    objects = len(sample.objects)
    parts = sum(len(instance.parts) for instance in sample.objects)

    def count_descendants(node: HierarchyNode) -> int:
        return sum(1 + count_descendants(child) for child in node.children)

    subparts = sum(
        count_descendants(part)
        for instance in sample.objects
        for part in instance.parts
    )
    return objects, parts, subparts


def node_label(node: HierarchyNode) -> str:
    return str(node.category_name or node.category_id or node.node_id)


def collect_descendant_labels(node: HierarchyNode) -> set[str]:
    labels = set()
    for child in node.children:
        labels.add(node_label(child))
        labels.update(collect_descendant_labels(child))
    return labels


def distinct_object_part_and_subpart_labels(sample: HierarchicalSample) -> tuple[set[str], set[str], set[str]]:
    objects = set()
    parts = set()
    subparts = set()
    for instance in sample.objects:
        objects.add(node_label(instance))
        for part in instance.parts:
            parts.add(node_label(part))
            subparts.update(collect_descendant_labels(part))
    return objects, parts, subparts


def paco_annotations_path(path: Path, split: str) -> Path:
    if path.is_file():
        return path
    matches = sorted(path.rglob(f"*lvis*_{split}.json")) if path.is_dir() else []
    if len(matches) != 1:
        raise ValueError(f"Expected one PACO-LVIS {split} annotation JSON under {path}; found {len(matches)}")
    return matches[0]


def draw_node(axis, node: HierarchyNode, color: str) -> None:
    from matplotlib.patches import Rectangle

    draw_segmentation(axis, node.segmentation, color)

    if node.bbox is not None:
        x, y, width, height = node.bbox
        axis.add_patch(Rectangle((x, y), width, height, fill=False, edgecolor=color, linewidth=1.5))
        label = node.category_name or str(node.category_id or node.node_id)
        axis.text(x, y, label, color=color, fontsize=7, backgroundcolor="white")
    for child in node.children:
        draw_node(axis, child, "tab:green")


def draw_segmentation(axis, segmentation, color: str) -> None:
    """Overlay COCO polygon or compressed-RLE masks on an image axis."""
    import numpy as np
    from matplotlib.patches import Polygon
    from matplotlib.colors import to_rgba

    if isinstance(segmentation, np.ndarray):
        mask = segmentation.astype(bool)
        overlay = np.zeros((*mask.shape, 4), dtype=float)
        overlay[mask] = to_rgba(color, alpha=0.7)
        axis.imshow(overlay, interpolation="none")
        return
    if isinstance(segmentation, dict):
        mask = decode_rle(segmentation)
        if mask is not None:
            overlay = np.zeros((*mask.shape, 4), dtype=float)
            overlay[mask] = to_rgba(color, alpha=0.7)
            axis.imshow(overlay, interpolation="none")
        return
    if not isinstance(segmentation, list):
        return
    for polygon in segmentation:
        if isinstance(polygon, list) and len(polygon) >= 6:
            points = list(zip(polygon[::2], polygon[1::2]))
            axis.add_patch(Polygon(points, closed=True, facecolor=color, edgecolor=color, alpha=0.7))


def decode_rle(rle):
    """Decode COCO compressed RLE into a height-by-width boolean mask."""
    from pycocotools import mask as mask_utils

    size = rle.get("size")
    counts = rle.get("counts")
    if not isinstance(size, list) or len(size) != 2 or not isinstance(counts, str):
        return None
    decoded = mask_utils.decode({"size": size, "counts": counts.encode("ascii")})
    return decoded.astype(bool)


def save_sample(sample: HierarchicalSample, output_path: Path) -> None:
    import matplotlib.pyplot as plt

    figure = plt.figure(figsize=(14, 6))
    palette = plt.get_cmap("tab20")
    whole_figure, part_figure, subpart_figure = figure.subfigures(1, 3, wspace=0.06)
    whole_axis, part_axis, subpart_axis = (
        whole_figure.subplots(), part_figure.subplots(), subpart_figure.subplots()
    )

    for axis, title in (
        (whole_axis, "Whole/object masks"),
        (part_axis, "Part masks"),
        (subpart_axis, "Subpart masks"),
    ):
        axis.imshow(sample.image)
        axis.set_title(f"{title} - image {sample.image_id}")
        axis.axis("off")

    for object_index, instance in enumerate(sample.objects):
        object_color = palette(object_index % 20)
        draw_segmentation(whole_axis, instance.segmentation, object_color)
        for part_index, part in enumerate(instance.parts):
            part_color = palette((object_index + part_index + 1) % 20)
            draw_segmentation(part_axis, part.segmentation, part_color)
            for child_index, child in enumerate(part.children):
                child_color = palette((object_index + part_index + child_index + 2) % 20)
                draw_segmentation(subpart_axis, child.segmentation, child_color)
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("paco", "pascal_part", "partimagenet", "spin"), default="paco")
    parser.add_argument("--root", type=Path, required=True, help="Image root")
    parser.add_argument("--annotations", type=Path, required=True, help="Annotation directory or JSON file")
    parser.add_argument("--split", default="test", help="Dataset split (PACO requires an annotation directory)")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=10)
    args = parser.parse_args()
    if args.output is None:
        args.output = PROJECT_ROOT / "outputs" / args.dataset

    if args.dataset == "paco":
        args.annotations = paco_annotations_path(args.annotations, args.split)
        dataset = PACODataset(args.root, args.annotations, load_images=False)
    elif args.dataset == "pascal_part":
        dataset = PascalPartDataset(args.root, args.annotations, split=args.split, load_images=False)
    elif args.dataset == "partimagenet":
        dataset = PartImageNetDataset(args.root, split=args.split, load_images=False)
    else:
        dataset = SPINDataset(args.root, args.annotations, split=args.split, load_images=False)
    args.output.mkdir(parents=True, exist_ok=True)
    print(f"Images: {len(dataset)}")
    totals = [0, 0, 0]
    distinct_objects = set()
    distinct_parts = set()
    distinct_subparts = set()
    for index in range(len(dataset)):
        sample = dataset[index]
        counts = count_nodes(sample)
        totals = [left + right for left, right in zip(totals, counts)]
        sample_objects, sample_parts, sample_subparts = distinct_object_part_and_subpart_labels(sample)
        distinct_objects.update(sample_objects)
        distinct_parts.update(sample_parts)
        distinct_subparts.update(sample_subparts)
    print(f"Annotations: objects={totals[0]}, parts={totals[1]}, subparts={totals[2]}")
    print(f"Distinct objects: {len(distinct_objects)}")
    print(f"Distinct parts: {len(distinct_parts)}")
    if distinct_subparts:
        print(f"Distinct subparts: {len(distinct_subparts)}")

    if args.dataset == "paco":
        image_dataset = PACODataset(args.root, args.annotations)
    elif args.dataset == "pascal_part":
        image_dataset = PascalPartDataset(args.root, args.annotations, split=args.split)
    elif args.dataset == "partimagenet":
        image_dataset = PartImageNetDataset(args.root, split=args.split)
    else:
        image_dataset = SPINDataset(args.root, args.annotations, split=args.split)
    for index in range(min(args.limit, len(dataset))):
        sample = image_dataset[index]
        save_sample(sample, args.output / f"{sample.image_id}.png")


if __name__ == "__main__":
    main()
