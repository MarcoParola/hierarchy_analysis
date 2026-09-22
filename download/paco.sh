#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="${PACO_DATA_ROOT:-$PROJECT_ROOT/data/paco}"
ANNOTATION_ROOT="$DATA_ROOT/annotations"
IMAGE_ROOT="$DATA_ROOT/images"
WITH_COCO=1

PACO_LVIS_URL="https://dl.fbaipublicfiles.com/paco/annotations/paco_lvis_v1.zip"
PACO_LVIS_SHA256="02ac4edb22c251e07853e6231d69aec3fad0a180f03de2f8c880650322debc80"
PACO_EGO4D_URL="https://dl.fbaipublicfiles.com/paco/annotations/paco_ego4d_v1.zip"
PACO_EGO4D_SHA256="9a2de524dd64ad8f807f0d1ad2e96de590b9fb222e55192ccfdd7b7b09b89252"

usage() {
    cat <<'EOF'
Usage: ./download/download.sh paco [--with-coco]

Environment:
  PACO_DATA_ROOT  Destination directory (default: ./data/paco)

The default downloads PACO-LVIS and PACO-EGO4D annotations plus the COCO
train2017 and val2017 image archives used by PACO-LVIS. --with-coco is retained
as an explicit, self-documenting option.
PACO-EGO4D frames require an approved Ego4D account and must be downloaded
with the official Ego4D CLI separately.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --with-coco)
            WITH_COCO=1
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            printf 'Unknown option: %s\n\n' "$1" >&2
            usage >&2
            exit 2
            ;;
    esac
    shift
done

for command in curl sha256sum unzip; do
    if ! command -v "$command" >/dev/null 2>&1; then
        printf 'Required command not found: %s\n' "$command" >&2
        exit 3
    fi
done

download_zip() {
    local url="$1"
    local expected_sha256="$2"
    local destination="$3"
    local archive="$destination/$(basename "$url")"

    mkdir -p "$destination"
    if [[ -f "$archive" ]]; then
        printf 'Using existing archive: %s\n' "$archive"
    else
        printf 'Downloading %s\n' "$url"
        curl --fail --location --continue-at - --output "$archive" "$url"
    fi

    printf '%s  %s\n' "$expected_sha256" "$archive" | sha256sum --check --status
    printf 'Extracting %s\n' "$archive"
    unzip -oq "$archive" -d "$destination"
}

download_zip "$PACO_LVIS_URL" "$PACO_LVIS_SHA256" "$ANNOTATION_ROOT"
download_zip "$PACO_EGO4D_URL" "$PACO_EGO4D_SHA256" "$ANNOTATION_ROOT"

if [[ "$WITH_COCO" -eq 1 ]]; then
    COCO_TRAIN_URL="http://images.cocodataset.org/zips/train2017.zip"
    COCO_VAL_URL="http://images.cocodataset.org/zips/val2017.zip"
    printf 'COCO image archives do not have checksums in the PACO instructions.\n' >&2
    mkdir -p "$IMAGE_ROOT"
    curl --fail --location --continue-at - --output "$IMAGE_ROOT/train2017.zip" "$COCO_TRAIN_URL"
    curl --fail --location --continue-at - --output "$IMAGE_ROOT/val2017.zip" "$COCO_VAL_URL"
    unzip -oq "$IMAGE_ROOT/train2017.zip" -d "$IMAGE_ROOT"
    unzip -oq "$IMAGE_ROOT/val2017.zip" -d "$IMAGE_ROOT"
fi

printf '\nPACO data is available under %s\n' "$DATA_ROOT"
printf 'PACO-LVIS images: COCO train2017/val2017 (use --with-coco)\n'
printf 'PACO-EGO4D frames: download separately with the official Ego4D CLI.\n'