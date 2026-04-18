"""Model adapters: wrap third-party models behind the plumbline ``Model`` ABC."""

from plumbline.models.base import Model, ModelCapabilities, Prediction
from plumbline.models.registry import MODEL_REGISTRY, register_model

__all__ = [
    "MODEL_REGISTRY",
    "Model",
    "ModelCapabilities",
    "Prediction",
    "register_model",
]
