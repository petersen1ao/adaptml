#!/usr/bin/env python3
"""
Copyright (c) 2024 AdaptML Team. All rights reserved.

AdaptML - Community Edition
Cut AI inference costs by 50% with adaptive model selection

This is the Community Edition under MIT License.
Enterprise features available at: https://adaptml-web-showcase.lovable.app/
Contact: info2adaptml@gmail.com

NOTICE: This file contains proprietary algorithms. Reverse engineering
or commercial redistribution without permission is prohibited.
Patents pending on core optimization techniques.
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
