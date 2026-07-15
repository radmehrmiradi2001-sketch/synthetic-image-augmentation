import numpy as np
import pytest

from vision_augmentation import (
    AugmentationPipeline,
    AugmentationSample,
    RandomHorizontalFlip,
    RandomRotation,
    SyntheticShapeGenerator,
)


def test_horizontal_flip_keeps_mask_aligned() -> None:
    image = np.zeros((4, 6, 3), dtype=np.uint8)
    mask = np.zeros((4, 6), dtype=np.uint8)
    image[:, :2] = (255, 0, 0)
    mask[:, :2] = 1
    output = RandomHorizontalFlip(probability=1)(
        AugmentationSample(image, mask), np.random.default_rng(1)
    )
    assert np.all(output.image[:, -2:, 0] == 255)
    assert np.all(output.mask[:, -2:] == 1)


def test_rotation_preserves_discrete_mask_classes() -> None:
    sample = SyntheticShapeGenerator(64, 64, (2, 2)).generate(np.random.default_rng(5))
    output = RandomRotation(20, probability=1)(sample, np.random.default_rng(6))
    assert set(np.unique(output.mask)).issubset(set(np.unique(sample.mask)) | {0})


def test_pipeline_is_reproducible() -> None:
    sample = SyntheticShapeGenerator(64, 64).generate(np.random.default_rng(7))
    pipeline = AugmentationPipeline([RandomHorizontalFlip(0.5), RandomRotation(20, 1)])
    first = pipeline(sample, np.random.default_rng(99))
    second = pipeline(sample, np.random.default_rng(99))
    assert np.array_equal(first.image, second.image)
    assert np.array_equal(first.mask, second.mask)


def test_sample_rejects_misaligned_mask() -> None:
    with pytest.raises(ValueError, match="dimensions must match"):
        AugmentationSample(
            np.zeros((8, 8, 3), dtype=np.uint8), np.zeros((7, 8), dtype=np.uint8)
        ).validate()
