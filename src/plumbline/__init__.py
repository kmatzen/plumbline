"""plumbline — reproducible eval harness for 3D geometric foundation models."""

from plumbline._version import __version__
from plumbline.datasets.base import Dataset, Sample
from plumbline.models.base import Model, ModelCapabilities, Prediction

__all__ = [
    "Dataset",
    "Model",
    "ModelCapabilities",
    "Prediction",
    "Sample",
    "__version__",
]
