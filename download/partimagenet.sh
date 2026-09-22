#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="${PARTIMAGENET_DATA_ROOT:-$PROJECT_ROOT/data/partimagenet}"
ARCHIVE="$DATA_ROOT/PartImageNet_Seg.zip"
SOURCE_ARCHIVE="${PARTIMAGENET_SOURCE_ARCHIVE:-$PROJECT_ROOT/data/spin/PartImageNet_Seg.zip}"
URL="https://huggingface.co/datasets/turkeyju/PartImageNet/resolve/main/PartImageNet_Seg.zip?download=true"

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    cat <<EOF
Usage: ./download/download.sh partimagenet

Extracts PartImageNet_Seg under: $DATA_ROOT
If the archive downloaded by the SPIN setup exists, it is reused.
EOF
    exit 0
fi
if [[ $# -gt 0 ]]; then
    printf 'Unknown option: %s\n' "$1" >&2
    exit 2
fi

for command in curl unzip rsync cp; do
    command -v "$command" >/dev/null 2>&1 || {
        printf 'Required command not found: %s\n' "$command" >&2
        exit 3
    }
done

mkdir -p "$DATA_ROOT"
if [[ -f "$SOURCE_ARCHIVE" && "$SOURCE_ARCHIVE" != "$ARCHIVE" ]]; then
    cp "$SOURCE_ARCHIVE" "$ARCHIVE"
elif [[ ! -f "$ARCHIVE" ]]; then
    curl --fail --location --continue-at - --output "$ARCHIVE" "$URL"
fi

mkdir -p "$DATA_ROOT/.staging"
unzip -oq "$ARCHIVE" -d "$DATA_ROOT/.staging"
rsync --archive --exclude='._*' --exclude='.DS_Store' \
    "$DATA_ROOT/.staging/PartImageNet/images/" "$DATA_ROOT/images/"
rsync --archive --exclude='._*' --exclude='.DS_Store' \
    "$DATA_ROOT/.staging/PartImageNet/annotations/" "$DATA_ROOT/annotations/"
rm -rf "$DATA_ROOT/.staging"

printf 'PartImageNet data is available under: %s\n' "$DATA_ROOT"