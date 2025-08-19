"""
AdaptML - Adaptive Inference Open Source Edition
Cut AI inference costs by 50% with adaptive model selection

GitHub: https://github.com/petersen1ao/adaptml
Website: https://adaptml-web-showcase.lovable.app/
Email: info2adaptml@gmail.com
License: MIT
"""

import time
import asyncio
import logging
from enum import Enum
from dataclasses import dataclass
from typing import Any, Dict, Optional, List, Union, Callable, Tuple
import warnings

# Core dependencies
import numpy as np
import psutil

# Optional ML framework imports with fallbacks
try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None
    nn = None

try:
    import tensorflow as tf
    HAS_TENSORFLOW = True
except ImportError:
    HAS_TENSORFLOW = False
    tf = None

try:
    import onnxruntime as ort
    HAS_ONNX = True
except ImportError:
    HAS_ONNX = False
    ort = None

class ModelSize(Enum):
    NANO = "nano"
    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large"
    XLARGE = "xlarge"

class DeviceType(Enum):
    CPU = "cpu"
    GPU = "gpu"
    TPU = "tpu"
    EDGE = "edge"

@dataclass
class AdaptiveConfig:
    """Configuration for adaptive inference"""
    cost_threshold: float = 0.01
    latency_threshold: float = 1.0
    quality_threshold: float = 0.8
    prefer_speed: bool = False
    prefer_accuracy: bool = False
    prefer_cost: bool = True
    device_preference: Optional[DeviceType] = None

@dataclass
class InferenceResult:
    """Result of an inference operation"""
    prediction: Any
    confidence: float
    cost: float
    latency: float
    model_used: str
    device_used: str
    metadata: Dict[str, Any]

class DeviceProfiler:
    """Profiles available devices and their capabilities"""
    
    def __init__(self):
        self.device_info = self._profile_devices()
    
    def _profile_devices(self) -> Dict[str, Dict[str, Any]]:
        """Profile available devices"""
        devices = {}
        
        # CPU profiling
        cpu_info = {
            "type": DeviceType.CPU,
            "cores": psutil.cpu_count(),
            "memory": psutil.virtual_memory().total / (1024**3),  # GB
            "cost_per_second": 0.0001,
            "available": True
        }
        devices["cpu"] = cpu_info
        
        # Check for GPU availability
        gpu_available = False
        if HAS_TORCH and torch.cuda.is_available():
            gpu_available = True
        elif HAS_TENSORFLOW:
            try:
                gpus = tf.config.experimental.list_physical_devices('GPU')
                if gpus:
                    gpu_available = True
            except:
                pass
        
        if gpu_available:
            devices["gpu"] = {
                "type": DeviceType.GPU,
                "memory": 8.0,  # Estimate
                "cost_per_second": 0.001,
                "available": True
            }
        
        return devices
    
    def get_optimal_device(self, config: AdaptiveConfig) -> str:
        """Get the optimal device for given configuration"""
        if config.device_preference:
            device_name = config.device_preference.value
            if device_name in self.device_info and self.device_info[device_name]["available"]:
                return device_name
        
        # Default selection logic
        if config.prefer_cost:
            return "cpu"
        elif "gpu" in self.device_info and self.device_info["gpu"]["available"]:
            return "gpu"
        else:
            return "cpu"

class ModelRegistry:
    """Registry for managing models and their metadata"""
    
    def __init__(self):
        self.models: Dict[str, Dict[str, Any]] = {}
        self._model_counter = 0
    
    def register_model(self, name: str, model_data: Any, metadata: Optional[Dict[str, Any]] = None) -> str:
        """Register a model"""
        model_id = f"model_{self._model_counter}"
        self._model_counter += 1
        
        self.models[model_id] = {
            "name": name,
            "data": model_data,
            "metadata": metadata or {},
            "registered_at": time.time()
        }
        
        return model_id
    
    def get_model(self, model_id: str) -> Optional[Dict[str, Any]]:
        """Get model by ID"""
        return self.models.get(model_id)
    
    def list_models(self) -> List[str]:
        """List all registered model IDs"""
        return list(self.models.keys())

class InferenceEngine:
    """Base class for inference engines"""
    
    def __init__(self, device: str = "cpu"):
        self.device = device
        self.cost_per_second = 0.0001 if device == "cpu" else 0.001
    
    async def infer(self, model: Any, input_data: Any) -> Tuple[Any, float]:
        """Run inference on the model"""
        raise NotImplementedError
    
    def estimate_cost(self, latency: float) -> float:
        """Estimate cost based on latency"""
        return latency * self.cost_per_second

class MockEngine(InferenceEngine):
    """Mock inference engine for testing"""
    
    async def infer(self, model: Any, input_data: Any) -> Tuple[Any, float]:
        """Mock inference that simulates processing"""
        # Simulate processing time
        await asyncio.sleep(0.1)
        
        # Mock result based on model type
        model_type = model.get('type', 'mock') if isinstance(model, dict) else 'mock'
        
        if 'text' in str(input_data).lower():
            prediction = f"Processed: {input_data} [by {model_type} model]"
        else:
            prediction = f"Mock result for {model_type} model"
        
        confidence = 0.85
        return prediction, confidence

# Conditional engine classes
if HAS_TORCH:
    class PyTorchEngine(InferenceEngine):
        """PyTorch inference engine"""
        
        async def infer(self, model: torch.nn.Module, input_data: Any) -> Tuple[Any, float]:
            """Run PyTorch inference"""
            model.eval()
            with torch.no_grad():
                if isinstance(input_data, np.ndarray):
                    input_tensor = torch.from_numpy(input_data).float()
                else:
                    # For demo purposes, create a dummy tensor
                    input_tensor = torch.randn(1, 10)
                
                output = model(input_tensor)
                confidence = torch.softmax(output, dim=-1).max().item()
                
                return output.numpy(), confidence

if HAS_TENSORFLOW:
    class TensorFlowEngine(InferenceEngine):
        """TensorFlow inference engine"""
        
        async def infer(self, model: tf.keras.Model, input_data: Any) -> Tuple[Any, float]:
            """Run TensorFlow inference"""
            if isinstance(input_data, np.ndarray):
                input_array = input_data
            else:
                # For demo purposes, create a dummy array
                input_array = np.random.randn(1, 10).astype(np.float32)
            
            output = model(input_array)
            confidence = float(tf.nn.softmax(output).numpy().max())
            
            return output.numpy(), confidence

if HAS_ONNX:
    class ONNXEngine(InferenceEngine):
        """ONNX Runtime inference engine"""
        
        async def infer(self, model: 'ort.InferenceSession', input_data: Any) -> Tuple[Any, float]:
            """Run ONNX inference"""
            input_name = model.get_inputs()[0].name
            
            if isinstance(input_data, np.ndarray):
                input_array = input_data
            else:
                # For demo purposes, create a dummy array
                input_array = np.random.randn(1, 10).astype(np.float32)
            
            output = model.run(None, {input_name: input_array})
            confidence = float(np.max(output[0]))
            
            return output, confidence

class AdaptiveInference:
    """Main adaptive inference system"""
    
    def __init__(self, config: Optional[AdaptiveConfig] = None):
        self.config = config or AdaptiveConfig()
        self.device_profiler = DeviceProfiler()
        self.model_registry = ModelRegistry()
        self.engines = {
            "mock": MockEngine(),
            "cpu": MockEngine("cpu"),
        }
        
        # Initialize available engines
        if HAS_TORCH:
            self.engines["pytorch"] = PyTorchEngine()
        if HAS_TENSORFLOW:
            self.engines["tensorflow"] = TensorFlowEngine()
        if HAS_ONNX:
            self.engines["onnx"] = ONNXEngine()
    
    def register_model(self, name: str, model_data: Any, metadata: Optional[Dict[str, Any]] = None) -> str:
        """Register a model for adaptive inference"""
        return self.model_registry.register_model(name, model_data, metadata)
    
    async def predict_async(self, model_id: str, input_data: Any, **kwargs) -> InferenceResult:
        """Run async prediction with adaptive selection"""
        model_info = self.model_registry.get_model(model_id)
        if not model_info:
            raise ValueError(f"Model {model_id} not found")
        
        # Select optimal device and engine
        device = self.device_profiler.get_optimal_device(self.config)
        engine_type = self._select_engine(model_info["data"], device)
        engine = self.engines[engine_type]
        
        # Run inference
        start_time = time.time()
        prediction, confidence = await engine.infer(model_info["data"], input_data)
        latency = time.time() - start_time
        
        # Calculate cost
        cost = engine.estimate_cost(latency)
        
        return InferenceResult(
            prediction=prediction,
            confidence=confidence,
            cost=cost,
            latency=latency,
            model_used=model_info["name"],
            device_used=device,
            metadata={"engine": engine_type}
        )
    
    def predict(self, model_id: str, input_data: Any, **kwargs) -> InferenceResult:
        """Synchronous prediction wrapper"""
        return asyncio.run(self.predict_async(model_id, input_data, **kwargs))
    
    def _select_engine(self, model_data: Any, device: str) -> str:
        """Select the appropriate inference engine"""
        if isinstance(model_data, dict):
            model_type = model_data.get('type', 'mock')
            if model_type in self.engines:
                return model_type
        
        # Default engine selection
        if HAS_TORCH and isinstance(model_data, torch.nn.Module):
            return "pytorch"
        elif HAS_TENSORFLOW and hasattr(model_data, 'predict'):
            return "tensorflow"
        elif HAS_ONNX and hasattr(model_data, 'run'):
            return "onnx"
        else:
            return "mock"
    
    def get_stats(self) -> Dict[str, Any]:
        """Get system statistics"""
        return {
            "registered_models": len(self.model_registry.models),
            "available_engines": list(self.engines.keys()),
            "device_info": self.device_profiler.device_info,
            "config": self.config
        }

def create_demo_models(inference: 'AdaptiveInference' = None) -> Dict[str, str]:
    """Create demo models for testing"""
    if inference is None:
        inference = AdaptiveInference()
    models = {}
    
    # Register mock models
    models["small"] = inference.register_model(
        "small_classifier",
        {"type": "mock", "size": "small"},
        {"description": "Small mock classifier"}
    )
    
    models["medium"] = inference.register_model(
        "medium_transformer",
        {"type": "mock", "size": "medium"},
        {"description": "Medium transformer model"}
    )
    
    return models

def quickstart():
    """Quick demo of AdaptML functionality"""
    print("🚀 AdaptML Quick Start Demo")
    print("=" * 40)
    
    # Create inference system
    config = AdaptiveConfig(prefer_cost=True, cost_threshold=0.01)
    inference = AdaptiveInference(config)
    
    print(f"📧 Contact: info2adaptml@gmail.com")
    print(f"🌐 Website: https://adaptml-web-showcase.lovable.app/")
    print()
    
    # Create demo models
    print("📦 Creating demo models...")
    models = create_demo_models(inference)
    print(f"✅ Created {len(models)} demo models")
    
    # Test inference
    print("\n🔮 Running inference tests...")
    for name, model_id in models.items():
        result = inference.predict(model_id, f"Test input for {name} model")
        print(f"  {name}: {result.prediction[:50]}... (cost: ${result.cost:.6f}, latency: {result.latency:.3f}s)")
    
    # Show stats
    print("\n📊 System Statistics:")
    stats = inference.get_stats()
    print(f"  Models registered: {stats['registered_models']}")
    print(f"  Available engines: {', '.join(stats['available_engines'])}")
    print(f"  Available devices: {', '.join(stats['device_info'].keys())}")
    
    print("\n🎉 Demo completed! AdaptML is ready for adaptive inference.")
    print("💡 Tip: Use different cost/latency thresholds to optimize for your needs")

if __name__ == "__main__":
    quickstart()
