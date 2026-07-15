"""Deterministic image augmentation and synthetic dataset generation."""

from .export import DatasetExporter, ExportSummary
from .generator import ShapeClass, SyntheticShapeGenerator
from .pipeline import AugmentationPipeline, AugmentationSample
from .transforms import (
    GaussianNoise,
    RandomBrightnessContrast,
    RandomCutout,
    RandomHorizontalFlip,
    RandomRotation,
)

__all__ = [
    "AugmentationPipeline",
    "AugmentationSample",
    "DatasetExporter",
    "ExportSummary",
    "GaussianNoise",
    "RandomBrightnessContrast",
    "RandomCutout",
    "RandomHorizontalFlip",
    "RandomRotation",
    "ShapeClass",
    "SyntheticShapeGenerator",
]

__version__ = "1.0.0"
