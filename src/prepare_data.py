import argparse
import json
from pathlib import Path

from .data import build_manifests


def main():
    parser = argparse.ArgumentParser(description="Audit the full dataset and create deterministic split manifests.")
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/data"))
    parser.add_argument("--verify-images", action="store_true")
    args = parser.parse_args()
    summary = build_manifests(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        verify_images=args.verify_images,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
