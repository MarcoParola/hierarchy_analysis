#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="${PASCAL_PART_DATA_ROOT:-$PROJECT_ROOT/data/pascal_part}"
WITH_VOC=1

ANNOTATION_URL="http://roozbehm.info/pascal-parts/trainval.tar.gz"
ANNOTATION_ARCHIVE="$DATA_ROOT/trainval.tar.gz"
ANNOTATION_MD5="2fa0a19ee9b5e43b2bee520166111120"

VOC_URL="http://host.robots.ox.ac.uk/pascal/VOC/voc2010/VOCtrainval_03-May-2010.tar"
VOC_ARCHIVE="$DATA_ROOT/VOCtrainval_03-May-2010.tar"
VOC_MD5="da459979d0c395079b5c75ee67908abb"

usage() {
    cat <<'EOF'
Usage: ./download/download.sh pascal_part [--annotations-only]

Environment:
  PASCAL_PART_DATA_ROOT  Destination directory (default: ./data/pascal_part)

The default downloads PASCAL-Part train/val annotations and the PASCAL VOC 2010
train/val images. Use --annotations-only if VOC2010 is already available.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --annotations-only)
            WITH_VOC=0
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

for command in curl md5sum tar; do
    if ! command -v "$command" >/dev/null 2>&1; then
        printf 'Required command not found: %s\n' "$command" >&2
        exit 3
    fi
done

download_file() {
    local url="$1"
    local expected_md5="$2"
    local archive="$3"

    mkdir -p "$(dirname "$archive")"
    if [[ -f "$archive" ]]; then
        printf 'Using existing archive: %s\n' "$archive"
    else
        printf 'Downloading %s\n' "$url"
        curl --fail --location --continue-at - --output "$archive" "$url"
    fi

    printf '%s  %s\n' "$expected_md5" "$archive" | md5sum --check --status
}

download_file "$ANNOTATION_URL" "$ANNOTATION_MD5" "$ANNOTATION_ARCHIVE"
tar -xzf "$ANNOTATION_ARCHIVE" -C "$DATA_ROOT"

if [[ "$WITH_VOC" -eq 1 ]]; then
    download_file "$VOC_URL" "$VOC_MD5" "$VOC_ARCHIVE"
    tar -xf "$VOC_ARCHIVE" -C "$DATA_ROOT"
fi

cat <<EOF
PASCAL-Part annotations are available under: $DATA_ROOT/Annotations_Part
PASCAL VOC 2010 images are available under: $DATA_ROOT/VOCdevkit/VOC2010/JPEGImages
EOF
