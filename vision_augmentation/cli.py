from __future__ import annotations

import argparse
from pathlib import Path

from .export import DatasetExporter
from .generator import SyntheticShapeGenerator
from .pipeline import AugmentationPipeline
from .transforms import (
    GaussianNoise,
    RandomBrightnessContrast,
    RandomCutout,
    RandomHorizontalFlip,
    RandomRotation,
)


def default_pipeline() -> AugmentationPipeline:
    return AugmentationPipeline([
        RandomHorizontalFlip(),
        RandomRotation(),
        RandomBrightnessContrast(),
        GaussianNoise(),
        RandomCutout(),
    ])


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate an augmented segmentation dataset.")
    parser.add_argument("--sources", type=int, default=10)
    parser.add_argument("--variants", type=int, default=4)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("generated_dataset"))
    args = parser.parse_args()
    exporter = DatasetExporter(
        SyntheticShapeGenerator(args.width, args.height), default_pipeline()
    )
    summary = exporter.export(
        args.output,
        source_count=args.sources,
        variants_per_source=args.variants,
        seed=args.seed,
    )
    print(
        f"Exported {summary.source_images} source and {summary.augmented_images} augmented "
        f"samples to {summary.output_directory}"
    )
    print(f"Manifest: {summary.manifest_path}")


if __name__ == "__main__":
    main()
