"""plumbline — reproducible eval harness for 3D geometric foundation models."""

from plumbline._version import __version__
from plumbline.datasets.base import Dataset, Sample
from plumbline.datasets.registry import register_dataset
from plumbline.models.base import Model, ModelCapabilities, Prediction
from plumbline.models.registry import register_model

__all__ = [
    "Dataset",
    "Model",
    "ModelCapabilities",
    "Prediction",
    "Sample",
    "__version__",
    "register_dataset",
    "register_model",
]
