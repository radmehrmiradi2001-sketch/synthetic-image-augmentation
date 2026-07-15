from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .pipeline import AugmentationSample


def _validate_probability(probability: float) -> None:
    if not 0.0 <= probability <= 1.0:
        raise ValueError("probability must be between zero and one")


@dataclass(frozen=True, slots=True)
class RandomHorizontalFlip:
    probability: float = 0.5

    def __post_init__(self) -> None:
        _validate_probability(self.probability)

    def __call__(self, sample: AugmentationSample, rng: np.random.Generator) -> AugmentationSample:
        if rng.random() >= self.probability:
            return sample
        image = np.ascontiguousarray(np.flip(sample.image, axis=1))
        mask = None if sample.mask is None else np.ascontiguousarray(np.flip(sample.mask, axis=1))
        return sample.with_data(image, mask)


@dataclass(frozen=True, slots=True)
class RandomRotation:
    maximum_degrees: float = 25.0
    probability: float = 0.6

    def __post_init__(self) -> None:
        if self.maximum_degrees < 0:
            raise ValueError("maximum_degrees cannot be negative")
        _validate_probability(self.probability)

    def __call__(self, sample: AugmentationSample, rng: np.random.Generator) -> AugmentationSample:
        if rng.random() >= self.probability:
            return sample
        height, width = sample.image.shape[:2]
        angle = float(rng.uniform(-self.maximum_degrees, self.maximum_degrees))
        matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1.0)
        image = cv2.warpAffine(
            sample.image,
            matrix,
            (width, height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101,
        )
        mask = None
        if sample.mask is not None:
            mask = cv2.warpAffine(
                sample.mask,
                matrix,
                (width, height),
                flags=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            )
        return sample.with_data(image, mask)


@dataclass(frozen=True, slots=True)
class RandomBrightnessContrast:
    brightness_limit: float = 0.2
    contrast_limit: float = 0.2
    probability: float = 0.7

    def __post_init__(self) -> None:
        if self.brightness_limit < 0 or self.contrast_limit < 0:
            raise ValueError("brightness and contrast limits cannot be negative")
        _validate_probability(self.probability)

    def __call__(self, sample: AugmentationSample, rng: np.random.Generator) -> AugmentationSample:
        if rng.random() >= self.probability:
            return sample
        contrast = float(rng.uniform(1 - self.contrast_limit, 1 + self.contrast_limit))
        brightness = float(rng.uniform(-self.brightness_limit, self.brightness_limit) * 255)
        image = np.clip(sample.image.astype(np.float32) * contrast + brightness, 0, 255).astype(np.uint8)
        return sample.with_data(image)


@dataclass(frozen=True, slots=True)
class GaussianNoise:
    standard_deviation: tuple[float, float] = (3.0, 18.0)
    probability: float = 0.4

    def __post_init__(self) -> None:
        minimum, maximum = self.standard_deviation
        if minimum < 0 or maximum < minimum:
            raise ValueError("standard_deviation must be an increasing non-negative range")
        _validate_probability(self.probability)

    def __call__(self, sample: AugmentationSample, rng: np.random.Generator) -> AugmentationSample:
        if rng.random() >= self.probability:
            return sample
        sigma = float(rng.uniform(*self.standard_deviation))
        noise = rng.normal(0.0, sigma, sample.image.shape)
        image = np.clip(sample.image.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        return sample.with_data(image)


@dataclass(frozen=True, slots=True)
class RandomCutout:
    maximum_fraction: float = 0.25
    probability: float = 0.35
    fill_value: tuple[int, int, int] = (0, 0, 0)

    def __post_init__(self) -> None:
        if not 0 < self.maximum_fraction <= 1:
            raise ValueError("maximum_fraction must be in (0, 1]")
        _validate_probability(self.probability)

    def __call__(self, sample: AugmentationSample, rng: np.random.Generator) -> AugmentationSample:
        if rng.random() >= self.probability:
            return sample
        height, width = sample.image.shape[:2]
        cutout_width = int(rng.integers(1, max(2, int(width * self.maximum_fraction) + 1)))
        cutout_height = int(rng.integers(1, max(2, int(height * self.maximum_fraction) + 1)))
        x = int(rng.integers(0, width - cutout_width + 1))
        y = int(rng.integers(0, height - cutout_height + 1))
        image = sample.image.copy()
        image[y : y + cutout_height, x : x + cutout_width] = self.fill_value
        return sample.with_data(image)
