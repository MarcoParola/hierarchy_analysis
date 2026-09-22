# Dataset Downloads

The download scripts store raw data under `data/`, which is ignored by Git.

## Available datasets

- **PACO**: object and part annotations for PACO-LVIS, plus COCO images. PACO-EGO4D annotations are included; its video frames require separate Ego4D access.
- **PASCAL-Part**: object and body-part segmentation masks for PASCAL VOC 2010 train/val images.
- **PartImageNet**: whole-object and part segmentation masks from the PartImageNet segmentation split.
- **SPIN**: hierarchical whole, part, and subpart annotations. Its images come from the PartImageNet segmentation split.

Download all datasets:

```bash
./download/download.sh all
```

## PACO

Download PACO annotations and the required COCO images:

```bash
./download/download.sh paco
```

Inspect PACO-LVIS and save 10 mask visualizations by default:

```bash
python test/test_dataset.py \
  --dataset paco \
  --root data/paco/images \
  --annotations "$(find data/paco/annotations -type f -iname '*lvis*.json' -print -quit)" \
  --limit 10
```

## PASCAL-Part

Download PASCAL-Part annotations and PASCAL VOC 2010 train/val images:

```bash
./download/download.sh pascal_part
```

Inspect PASCAL-Part train/val annotations and save 10 mask visualizations by default:

```bash
python test/test_dataset.py \
  --dataset pascal_part \
  --root data/pascal_part/VOCdevkit/VOC2010/JPEGImages \
  --annotations data/pascal_part/Annotations_Part \
  --split val \
  --limit 10
```

Use `--split train`, `--split val`, or `--split all` to change which annotation files are inspected. PASCAL-Part publishes train/val annotations; VOC2010 test images do not include released PASCAL-Part ground truth.

## SPIN

Download SPIN annotations and PartImageNet segmentation images:

```bash
./download/download.sh spin
```

Inspect the SPIN test split and save 10 mask visualizations by default:

```bash
python test/test_dataset.py \
  --dataset spin \
  --root data/spin/images \
  --annotations data/spin/annotations \
  --split test \
  --limit 10
```

The `--limit` option defaults to `10`; change it to save a different number of figures. Use `--split train` or `--split val` to inspect another split. Generated figures are saved under `outputs/`.

## PartImageNet

SPIN already downloads the PartImageNet segmentation archive. To extract it as a standalone dataset, the downloader reuses that archive when available:

```bash
./download/download.sh partimagenet
```

Inspect the PartImageNet validation split:

```bash
python test/test_dataset.py \
  --dataset partimagenet \
  --root data/partimagenet \
  --annotations data/partimagenet/annotations \
  --split val \
  --limit 10
```
