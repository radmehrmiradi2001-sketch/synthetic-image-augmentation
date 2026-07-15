from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from PIL import Image
import numpy as np

from .generator import SyntheticShapeGenerator
from .pipeline import AugmentationPipeline


@dataclass(frozen=True, slots=True)
class ExportSummary:
    source_images: int
    augmented_images: int
    output_directory: Path
    manifest_path: Path


class DatasetExporter:
    def __init__(self, generator: SyntheticShapeGenerator, pipeline: AugmentationPipeline) -> None:
        self.generator = generator
        self.pipeline = pipeline

    def export(
        self,
        output_directory: Path,
        *,
        source_count: int,
        variants_per_source: int,
        seed: int,
    ) -> ExportSummary:
        if source_count < 1 or variants_per_source < 0:
            raise ValueError("source_count must be positive and variants_per_source non-negative")
        image_directory = output_directory / "images"
        mask_directory = output_directory / "masks"
        image_directory.mkdir(parents=True, exist_ok=True)
        mask_directory.mkdir(parents=True, exist_ok=True)
        rng = np.random.default_rng(seed)
        records: list[dict[str, str | int]] = []

        for source_index in range(source_count):
            source_id = f"sample-{source_index:05d}"
            source = self.generator.generate(rng, source_id)
            records.append(self._save(source, image_directory, mask_directory, "source", 0))
            for variant_index in range(variants_per_source):
                variant = self.pipeline(source, rng)
                records.append(self._save(
                    variant, image_directory, mask_directory, "augmented", variant_index + 1
                ))

        manifest_path = output_directory / "manifest.json"
        manifest_path.write_text(json.dumps({"seed": seed, "samples": records}, indent=2), encoding="utf-8")
        return ExportSummary(source_count, source_count * variants_per_source, output_directory, manifest_path)

    @staticmethod
    def _save(sample, image_directory: Path, mask_directory: Path, kind: str, variant: int) -> dict[str, str | int]:
        stem = f"{sample.sample_id}_{kind}-{variant:02d}"
        image_path = image_directory / f"{stem}.png"
        mask_path = mask_directory / f"{stem}.png"
        Image.fromarray(sample.image, mode="RGB").save(image_path)
        if sample.mask is not None:
            Image.fromarray(sample.mask, mode="L").save(mask_path)
        return {
            "sample_id": sample.sample_id or stem,
            "kind": kind,
            "variant": variant,
            "image": str(image_path.relative_to(image_directory.parent)),
            "mask": str(mask_path.relative_to(mask_directory.parent)),
        }
