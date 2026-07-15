from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

ImageArray = NDArray[np.uint8]
MaskArray = NDArray[np.uint8]


@dataclass(frozen=True, slots=True)
class AugmentationSample:
    """An RGB image and its synchronized semantic-segmentation mask."""

    image: ImageArray
    mask: MaskArray | None = None
    sample_id: str | None = None

    def validate(self) -> None:
        if self.image.dtype != np.uint8 or self.image.ndim != 3 or self.image.shape[2] != 3:
            raise ValueError("image must be an RGB uint8 array with shape (height, width, 3)")
        if self.mask is not None:
            if self.mask.dtype != np.uint8 or self.mask.ndim != 2:
                raise ValueError("mask must be a uint8 array with shape (height, width)")
            if self.mask.shape != self.image.shape[:2]:
                raise ValueError("image and mask dimensions must match")

    def with_data(self, image: ImageArray, mask: MaskArray | None = None) -> AugmentationSample:
        return replace(self, image=image, mask=self.mask if mask is None else mask)


class Transform(Protocol):
    def __call__(
        self, sample: AugmentationSample, rng: np.random.Generator
    ) -> AugmentationSample: ...


class AugmentationPipeline:
    """Apply ordered transforms using a caller-owned random generator."""

    def __init__(self, transforms: list[Transform]) -> None:
        self.transforms = tuple(transforms)

    def __call__(
        self, sample: AugmentationSample, rng: np.random.Generator
    ) -> AugmentationSample:
        sample.validate()
        output = AugmentationSample(sample.image.copy(), None if sample.mask is None else sample.mask.copy(), sample.sample_id)
        for transform in self.transforms:
            output = transform(output, rng)
            output.validate()
        return output
