"""
AdaptML - Cut AI inference costs by 50% with adaptive model selection
"""

from .core import (
    AdaptiveInference,
    AdaptiveConfig,
    ModelSize,
    DeviceType,
    InferenceResult,
    DeviceProfiler,
    ModelRegistry,
    create_demo_models,
    quickstart
)

__version__ = "0.1.0"
__author__ = "AdaptML Team"
__email__ = "info2adaptml@gmail.com"
__website__ = "https://adaptml-web-showcase.lovable.app/"

__all__ = [
    "AdaptiveInference",
    "AdaptiveConfig", 
    "ModelSize",
    "DeviceType",
    "InferenceResult",
    "DeviceProfiler",
    "ModelRegistry",
    "create_demo_models",
    "quickstart"
]
