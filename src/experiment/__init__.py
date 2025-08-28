"""Experiment package for ML research with Pydantic + ExCa

This package provides a complete framework for reproducible machine learning
experiments with strong typing, validation, and caching capabilities.
"""

__version__ = "0.1.0"

from .schemas import ExperimentConfig
from .pipeline import Experiment

__all__ = ["ExperimentConfig", "Experiment"]