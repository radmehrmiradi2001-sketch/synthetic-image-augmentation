import json

from PIL import Image

from vision_augmentation import AugmentationPipeline, DatasetExporter, SyntheticShapeGenerator


def test_export_writes_images_masks_and_manifest(tmp_path) -> None:
    summary = DatasetExporter(
        SyntheticShapeGenerator(48, 48), AugmentationPipeline([])
    ).export(tmp_path, source_count=2, variants_per_source=1, seed=10)
    records = json.loads(summary.manifest_path.read_text())["samples"]
    assert len(records) == 4
    assert summary.source_images == 2
    assert summary.augmented_images == 2
    for record in records:
        image = Image.open(tmp_path / record["image"])
        mask = Image.open(tmp_path / record["mask"])
        assert image.size == mask.size == (48, 48)
