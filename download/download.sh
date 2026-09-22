#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

usage() {
    cat <<'EOF'
Usage: ./download/download.sh <dataset> [options]

Datasets:
  paco       Download PACO annotations and COCO images.
  pascal_part
             Download PASCAL-Part annotations and VOC2010 images.
  spin       Download SPIN COCO annotations.
  partimagenet
             Download PartImageNet segmentation images and masks.
  all        Run every registered dataset downloader in order.

Options are forwarded to the dataset-specific downloader. For PACO:
  --with-coco    Also download COCO train2017 and val2017 images.
For PASCAL-Part:
  --annotations-only
                 Download annotations without VOC2010 images.
EOF
}

if [[ $# -eq 0 || "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi

dataset="$1"
shift

case "$dataset" in
    paco)
	    if [[ $# -eq 0 ]]; then
	      set -- --with-coco
	    fi
	        exec "$SCRIPT_DIR/paco.sh" "$@"
	        ;;
      pascal_part)
        exec "$SCRIPT_DIR/pascal_part.sh" "$@"
        ;;
	      partimagenet)
	        exec "$SCRIPT_DIR/partimagenet.sh" "$@"
	        ;;
	      spin)
	        exec "$SCRIPT_DIR/spin.sh" "$@"
	        ;;
  all)
    if [[ $# -gt 0 ]]; then
      printf 'Options are not supported with the all target; run a dataset directly.\n' >&2
      exit 2
	    fi
	    "$SCRIPT_DIR/paco.sh"
	    "$SCRIPT_DIR/pascal_part.sh"
	    "$SCRIPT_DIR/spin.sh"
      "$SCRIPT_DIR/partimagenet.sh"
	    ;;
    *)
        printf 'Unknown dataset: %s\n\n' "$dataset" >&2
        usage >&2
        exit 2
        ;;
esac
