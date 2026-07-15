# Vision Augmentation

A deterministic, annotation-aware toolkit for generating synthetic computer-vision datasets and applying reproducible augmentation pipelines.

Unlike pixel-only augmentation scripts, Vision Augmentation keeps RGB images and semantic-segmentation masks synchronized through geometric transforms. It exports a structured dataset with a machine-readable manifest, making the result suitable for training experiments and CI fixtures.

## Features

- Typed `AugmentationSample` containing an RGB image and optional segmentation mask
- Deterministic pipelines driven by caller-owned NumPy generators
- Synchronized rotation and flipping with correct mask interpolation
- Photometric brightness, contrast, and Gaussian-noise transforms that leave labels unchanged
- Random cutout for occlusion robustness
- Synthetic multi-shape scenes with four segmentation classes
- Organized `images/` and `masks/` output plus JSON manifest
- Headless OpenCV dependency for servers and CI
- Automated alignment, reproducibility, validation, and export tests

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

## Generate a dataset

```bash
vision-augment \
  --sources 20 \
  --variants 5 \
  --width 256 \
  --height 256 \
  --seed 42 \
  --output generated_dataset
```

## Output layout

```text
generated_dataset/
  images/
    sample-00000_source-00.png
    sample-00000_augmented-01.png
  masks/
    sample-00000_source-00.png
    sample-00000_augmented-01.png
  manifest.json
```

Mask classes are `0=background`, `1=circle`, `2=rectangle`, and `3=triangle`.

## Python API

```python
import numpy as np

from vision_augmentation import (
    AugmentationPipeline,
    GaussianNoise,
    RandomHorizontalFlip,
    RandomRotation,
    SyntheticShapeGenerator,
)

rng = np.random.default_rng(42)
sample = SyntheticShapeGenerator(256, 256).generate(rng, "example")

pipeline = AugmentationPipeline([
    RandomHorizontalFlip(probability=0.5),
    RandomRotation(maximum_degrees=20, probability=0.7),
    GaussianNoise(probability=0.3),
])

augmented = pipeline(sample, rng)
```

## Design principles

- Geometric transforms update images and masks together.
- Masks always use nearest-neighbor interpolation to preserve class IDs.
- Photometric transforms affect only images.
- Randomness is explicit and reproducible; global RNG state is never modified.
- Every transform preserves dimensions and validates its output.

## Development

```bash
pytest
ruff check .
```

Future work can extend `AugmentationSample` with bounding boxes and keypoints using the same synchronization contract.
