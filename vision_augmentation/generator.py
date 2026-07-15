from __future__ import annotations

from enum import IntEnum

import cv2
import numpy as np

from .pipeline import AugmentationSample


class ShapeClass(IntEnum):
    BACKGROUND = 0
    CIRCLE = 1
    RECTANGLE = 2
    TRIANGLE = 3


class SyntheticShapeGenerator:
    """Generate labelled geometric scenes for segmentation experiments."""

    def __init__(self, width: int = 256, height: int = 256, shapes_per_image: tuple[int, int] = (1, 4)) -> None:
        if width < 32 or height < 32:
            raise ValueError("image dimensions must be at least 32 pixels")
        if shapes_per_image[0] < 1 or shapes_per_image[1] < shapes_per_image[0]:
            raise ValueError("shapes_per_image must be an increasing positive range")
        self.width = width
        self.height = height
        self.shapes_per_image = shapes_per_image

    def generate(self, rng: np.random.Generator, sample_id: str | None = None) -> AugmentationSample:
        image = self._background(rng)
        mask = np.zeros((self.height, self.width), dtype=np.uint8)
        count = int(rng.integers(self.shapes_per_image[0], self.shapes_per_image[1] + 1))
        for _ in range(count):
            shape_class = ShapeClass(int(rng.integers(1, len(ShapeClass))))
            self._draw_shape(image, mask, shape_class, rng)
        return AugmentationSample(image=image, mask=mask, sample_id=sample_id)

    def _background(self, rng: np.random.Generator) -> np.ndarray:
        start = rng.integers(0, 70, size=3).astype(np.float32)
        end = rng.integers(80, 180, size=3).astype(np.float32)
        blend = np.linspace(0, 1, self.width, dtype=np.float32)[None, :, None]
        row = start * (1 - blend) + end * blend
        image = np.repeat(row, self.height, axis=0)
        noise = rng.normal(0, 7, image.shape)
        return np.clip(image + noise, 0, 255).astype(np.uint8)

    def _draw_shape(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        shape_class: ShapeClass,
        rng: np.random.Generator,
    ) -> None:
        minimum_dimension = min(self.width, self.height)
        size = int(rng.integers(max(12, minimum_dimension // 10), max(13, minimum_dimension // 3)))
        center_x = int(rng.integers(size, self.width - size))
        center_y = int(rng.integers(size, self.height - size))
        color = tuple(int(value) for value in rng.integers(110, 256, size=3))
        class_value = int(shape_class)
        if shape_class is ShapeClass.CIRCLE:
            cv2.circle(image, (center_x, center_y), size, color, -1)
            cv2.circle(mask, (center_x, center_y), size, class_value, -1)
        elif shape_class is ShapeClass.RECTANGLE:
            first = (center_x - size, center_y - size)
            second = (center_x + size, center_y + size)
            cv2.rectangle(image, first, second, color, -1)
            cv2.rectangle(mask, first, second, class_value, -1)
        else:
            points = np.array([
                [center_x, center_y - size],
                [center_x - size, center_y + size],
                [center_x + size, center_y + size],
            ], dtype=np.int32)
            cv2.fillPoly(image, [points], color)
            cv2.fillPoly(mask, [points], class_value)
