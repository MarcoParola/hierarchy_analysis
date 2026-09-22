#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="${SPIN_DATA_ROOT:-$PROJECT_ROOT/data/spin}"
ANNOTATION_ROOT="$DATA_ROOT/annotations"
ANNOTATION_ARCHIVE="$ANNOTATION_ROOT/spin_coco.zip"
ANNOTATION_URL="https://joshmyersdean.github.io/spin/spin_coco.zip"
IMAGE_ARCHIVE="$DATA_ROOT/PartImageNet_Seg.zip"
IMAGE_URL="https://huggingface.co/datasets/turkeyju/PartImageNet/resolve/main/PartImageNet_Seg.zip?download=true"

ANNOTATIONS_ONLY=0

if [[ "${1:-}" == "--annotations-only" ]]; then
        ANNOTATIONS_ONLY=1
        shift
fi

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
        cat <<EOF
Usage: ./download/download.sh spin

Downloads the official SPIN COCO annotations and PartImageNet segmentation
images to:
    $DATA_ROOT

Images are normalized under:
    $DATA_ROOT/images/{train,val,test}/

Use --annotations-only to download only the SPIN annotation archive.
EOF
        exit 0
fi

if [[ $# -gt 0 ]]; then
        printf 'Unknown option: %s\n' "$1" >&2
        exit 2
fi

for command in curl unzip rsync; do
    command -v "$command" >/dev/null 2>&1 || {
        printf 'Required command not found: %s\n' "$command" >&2
        exit 3
    }
done

mkdir -p "$ANNOTATION_ROOT"
if [[ ! -f "$ANNOTATION_ARCHIVE" ]]; then
    curl --fail --location --continue-at - --output "$ANNOTATION_ARCHIVE" "$ANNOTATION_URL"
fi
unzip -oq "$ANNOTATION_ARCHIVE" -d "$ANNOTATION_ROOT"

if [[ "$ANNOTATIONS_ONLY" -eq 0 ]]; then
    mkdir -p "$DATA_ROOT/images" "$DATA_ROOT/.spin_image_staging"
    if [[ ! -f "$IMAGE_ARCHIVE" ]]; then
        curl --fail --location --continue-at - --output "$IMAGE_ARCHIVE" "$IMAGE_URL"
    fi
    unzip -oq "$IMAGE_ARCHIVE" -d "$DATA_ROOT/.spin_image_staging"

    for split in train val test; do
        mkdir -p "$DATA_ROOT/images/$split"
        rsync --archive --ignore-existing \
            --exclude='._*' --exclude='.DS_Store' \
            "$DATA_ROOT/.spin_image_staging/PartImageNet/images/$split/" \
            "$DATA_ROOT/images/$split/"
    done
    rm -rf "$DATA_ROOT/.spin_image_staging"
fi

cat <<EOF
SPIN annotations are available under: $ANNOTATION_ROOT
SPIN images are available under: $DATA_ROOT/images
EOF