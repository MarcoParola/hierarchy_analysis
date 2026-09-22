# Hierarchy Analysis

We investigate how different architectures encode visual hierarchical structure within their embeddings.

Supported datasets:

- PACO
- PASCAL-Part
- SPIN

## Usage

Create and activate the virtual environment, then install the dependencies:

```bash
python3 -m venv env
source env/bin/activate
pip install -r requirements.txt
```

Run the automated tests:

```bash
pytest -q
```

Download all supported datasets:

```bash
./download/download.sh all
```

See [download/README.md](download/README.md) for dataset details and individual commands.

## Feature extraction example

Extract features for the whole image only:

```bash
source env/bin/activate
python scripts/extract_features.py \
  --model dinov2 \
  --dataset partimagenet \
  --dataset-kwargs split=val \
  --save-dir data/features
```

Extract whole + object + part features for a hierarchical dataset:

```bash
source env/bin/activate
python scripts/extract_features.py \
  --model dinov2 \
  --dataset partimagenet \
  --dataset-kwargs split=val \
  --save-dir data/features \
  --hierarchical \
  --mask-mode black
```

You can also pass the full import path directly if you prefer:

```bash
python scripts/extract_features.py --model clip --dataset src.datasets.paco:PACODataset --dataset-kwargs annotations=path/to/annotations.json
```

This saves:

- `data/features/black/dinov2_whole_features.npy`
- `data/features/black/dinov2_object_features.npy`
- `data/features/black/dinov2_part_features.npy`
- and the matching ID files for each layer.

Hierarchical extraction supports `--mask-mode black`, `gray`, `blur`, or `crop`.
Each mode is saved in its own subdirectory under `--save-dir`.
