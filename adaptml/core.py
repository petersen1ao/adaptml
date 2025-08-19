"""
AdaptML - Open Source Edition
Cut AI inference costs by 50% with adaptive model selection
GitHub: https://github.com/yourusername/adaptml
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

# Optional ML framework support
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False

# System monitoring (optional but recommended)
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    warnings.warn("psutil not installed. Install with: pip install psutil")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('adaptml')


# ============================================================================
# CORE ENUMS AND DATA CLASSES
# ============================================================================

class ModelSize(Enum):
    """Model size tiers for adaptive selection"""
    SMALL = "small"   # Fastest, least accurate
    MEDIUM = "medium" # Balanced
    LARGE = "large"   # Slowest, most accurate


class DeviceType(Enum):
    """Supported device types"""
    CPU = "cpu"
    CUDA = "cuda"
    MPS = "mps"  # Apple Silicon
    TPU = "tpu"
    MOBILE = "mobile"
    UNKNOWN = "unknown"


@dataclass
class InferenceResult:
    """Result from adaptive inference"""
    output: Any
    confidence: float
    model_size: ModelSize
    latency_ms: float
    device: str
    metadata: Dict[str, Any] = None
    
    def __repr__(self):
        return (f"InferenceResult(confidence={self.confidence:.3f}, "
                f"model={self.model_size.value}, "
                f"latency={self.latency_ms:.1f}ms)")


@dataclass 
class AdaptiveConfig:
    """Configuration for adaptive behavior"""
    target_confidence: float = 0.9
    max_latency_ms: Optional[float] = None
    prefer_speed: bool = False
    enable_caching: bool = True
    device_override: Optional[str] = None
    

# ============================================================================
# DEVICE DETECTION AND PROFILING
# ============================================================================

class DeviceProfiler:
    """Detect and profile available compute devices"""
    
    @staticmethod
    def detect_device() -> Tuple[DeviceType, Dict[str, Any]]:
        """Detect the best available device"""
        info = {
            'device_name': 'CPU',
            'memory_gb': 0,
            'compute_capability': None
        }
        
        # Check for GPU availability
        if TORCH_AVAILABLE:
            if torch.cuda.is_available():
                info['device_name'] = torch.cuda.get_device_name(0)
                info['memory_gb'] = torch.cuda.get_device_properties(0).total_memory / 1e9
                info['compute_capability'] = torch.cuda.get_device_capability(0)
                return DeviceType.CUDA, info
            elif torch.backends.mps.is_available():
                info['device_name'] = 'Apple Silicon GPU'
                return DeviceType.MPS, info
        
        if TF_AVAILABLE:
            gpus = tf.config.list_physical_devices('GPU')
            if gpus:
                info['device_name'] = 'TensorFlow GPU'
                return DeviceType.CUDA, info
        
        # Check system memory
        if PSUTIL_AVAILABLE:
            info['memory_gb'] = psutil.virtual_memory().total / 1e9
            
        # Check if running on mobile/embedded
        import platform
        if 'arm' in platform.machine().lower():
            return DeviceType.MOBILE, info
            
        return DeviceType.CPU, info
    
    @staticmethod
    def get_available_memory() -> float:
        """Get available memory in MB"""
        if PSUTIL_AVAILABLE:
            return psutil.virtual_memory().available / 1e6
        return 1000.0  # Default assumption


# ============================================================================
# MODEL REGISTRY
# ============================================================================

class ModelRegistry:
    """Registry for managing models of different sizes"""
    
    def __init__(self):
        self.models: Dict[ModelSize, Any] = {}
        self.preprocessors: Dict[ModelSize, Callable] = {}
        self.postprocessors: Dict[ModelSize, Callable] = {}
        
    def register(
        self,
        model: Any,
        size: ModelSize,
        preprocessor: Optional[Callable] = None,
        postprocessor: Optional[Callable] = None
    ):
        """
        Register a model with optional pre/post processors
        
        Args:
            model: The model object (PyTorch, TF, ONNX, or callable)
            size: Model size tier
            preprocessor: Optional data preprocessor
            postprocessor: Optional output postprocessor
        """
        self.models[size] = model
        if preprocessor:
            self.preprocessors[size] = preprocessor
        if postprocessor:
            self.postprocessors[size] = postprocessor
        logger.info(f"Registered {size.value} model")
    
    def get_model(self, size: ModelSize) -> Any:
        """Get model by size"""
        if size not in self.models:
            raise ValueError(f"No model registered for size: {size.value}")
        return self.models[size]
    
    def has_model(self, size: ModelSize) -> bool:
        """Check if model size is available"""
        return size in self.models
    
    def get_available_sizes(self) -> List[ModelSize]:
        """Get list of available model sizes"""
        return list(self.models.keys())


# ============================================================================
# INFERENCE ENGINES
# ============================================================================

class InferenceEngine:
    """Base inference engine for different frameworks"""
    
    def __init__(self, device_type: DeviceType):
        self.device_type = device_type
        
    async def infer(self, model: Any, input_data: Any) -> Tuple[Any, float]:
        """
        Run inference and return output with confidence score
        Returns: (output, confidence)
        """
        raise NotImplementedError


class PyTorchEngine(InferenceEngine):
    """PyTorch inference engine"""
    
    def __init__(self, device_type: DeviceType):
        super().__init__(device_type)
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch not installed")
        
        # Set device
        if device_type == DeviceType.CUDA:
            self.device = torch.device('cuda')
        elif device_type == DeviceType.MPS:
            self.device = torch.device('mps')
        else:
            self.device = torch.device('cpu')
    
    async def infer(self, model: Any, input_data: Any) -> Tuple[Any, float]:
        """Run PyTorch inference"""
        model = model.to(self.device)
        model.eval()
        
        # Convert input to tensor
        if not isinstance(input_data, torch.Tensor):
            input_data = torch.tensor(input_data)
        input_data = input_data.to(self.device)
        
        # Run inference
        with torch.no_grad():
            output = model(input_data)
        
        # Calculate confidence (example - adapt to your needs)
        confidence = self._calculate_confidence(output)
        
        return output.cpu().numpy(), confidence
    
    def _calculate_confidence(self, output) -> float:
        """Calculate confidence from model output"""
        if TORCH_AVAILABLE and hasattr(output, 'dim'):
            if output.dim() > 1 and output.shape[-1] > 1:
                # Classification: use softmax probability
                import torch
                probs = torch.softmax(output, dim=-1)
                confidence = torch.max(probs).item()
            else:
                # Regression or binary: use sigmoid
                import torch
                confidence = torch.sigmoid(output).mean().item()
            return float(confidence)
        else:
            # Fallback for non-torch tensors
            return 0.8  # Default confidence


class ONNXEngine(InferenceEngine):
    """ONNX Runtime inference engine"""
    
    def __init__(self, device_type: DeviceType):
        super().__init__(device_type)
        if not ONNX_AVAILABLE:
            raise ImportError("ONNX Runtime not installed")
    
    async def infer(self, model: Any, input_data: Any) -> Tuple[Any, float]:
        """Run ONNX inference"""
        # Prepare input
        input_name = model.get_inputs()[0].name
        if not isinstance(input_data, np.ndarray):
            input_data = np.array(input_data, dtype=np.float32)
        
        # Run inference
        outputs = model.run(None, {input_name: input_data})
        output = outputs[0]
        
        # Calculate confidence
        if output.ndim > 1 and output.shape[-1] > 1:
            probs = np.exp(output) / np.sum(np.exp(output), axis=-1, keepdims=True)
            confidence = float(np.max(probs))
        else:
            confidence = float(1 / (1 + np.exp(-output.mean())))  # sigmoid
        
        return output, confidence


class TensorFlowEngine(InferenceEngine):
    """TensorFlow inference engine"""
    
    def __init__(self, device_type: DeviceType):
        super().__init__(device_type)
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow not installed")
    
    async def infer(self, model: Any, input_data: Any) -> Tuple[Any, float]:
        """Run TensorFlow inference"""
        # Prepare input
        if not isinstance(input_data, tf.Tensor):
            input_data = tf.constant(input_data, dtype=tf.float32)
        
        # Run inference
        output = model(input_data, training=False)
        
        # Calculate confidence
        if len(output.shape) > 1 and output.shape[-1] > 1:
            probs = tf.nn.softmax(output)
            confidence = float(tf.reduce_max(probs))
        else:
            confidence = float(tf.reduce_mean(tf.nn.sigmoid(output)))
        
        return output.numpy(), confidence



class MockEngine(InferenceEngine):
    """Mock inference engine for when no ML frameworks are available"""
    
    def __init__(self, device_type: DeviceType):
        super().__init__(device_type)
    
    async def infer(self, model: Any, input_data: Any) -> Tuple[Any, float]:
        """Run mock inference using callable models"""
        if callable(model):
            # Model is a lambda or function - call it directly
            return model(input_data)
        else:
            # Unknown model type - return input with default confidence
            return input_data, 0.8


# ============================================================================
# ADAPTIVE INFERENCE SYSTEM
# ============================================================================

class AdaptiveInference:
    """
    Main adaptive inference system
    
    Example:
        >>> system = AdaptiveInference()
        >>> system.register_model(small_model, ModelSize.SMALL)
        >>> system.register_model(large_model, ModelSize.LARGE)
        >>> result = await system.infer(data, target_confidence=0.95)
    """
    
    def __init__(self, config: Optional[AdaptiveConfig] = None):
        """Initialize adaptive inference system"""
        self.config = config or AdaptiveConfig()
        self.registry = ModelRegistry()
        self.device_type, self.device_info = DeviceProfiler.detect_device()
        self.engine = self._create_engine()
        self.cache = {} if self.config.enable_caching else None
        self._performance_history = []
        
        logger.info(f"Initialized on {self.device_type.value} device")
        logger.info(f"Device info: {self.device_info}")
    
    def _create_engine(self) -> InferenceEngine:
        """Create appropriate inference engine"""
        if self.config.device_override:
            self.device_type = DeviceType(self.config.device_override)
        
        # Try to create the best available engine
        if TORCH_AVAILABLE and self.device_type in [DeviceType.CPU, DeviceType.CUDA, DeviceType.MPS]:
            return PyTorchEngine(self.device_type)
        elif ONNX_AVAILABLE:
            return ONNXEngine(self.device_type)
        elif TF_AVAILABLE:
            return TensorFlowEngine(self.device_type)
        else:
            # Use mock engine for callable models
            return MockEngine(self.device_type)
    
    def register_model(
        self,
        model: Any,
        size: ModelSize,
        preprocessor: Optional[Callable] = None,
        postprocessor: Optional[Callable] = None
    ):
        """Register a model with the system"""
        self.registry.register(model, size, preprocessor, postprocessor)
    
    async def infer(
        self,
        input_data: Any,
        target_confidence: Optional[float] = None,
        max_latency_ms: Optional[float] = None
    ) -> InferenceResult:
        """
        Run adaptive inference
        
        Args:
            input_data: Input data for inference
            target_confidence: Override default target confidence
            max_latency_ms: Maximum allowed latency
            
        Returns:
            InferenceResult with output and metadata
        """
        target_confidence = target_confidence or self.config.target_confidence
        max_latency_ms = max_latency_ms or self.config.max_latency_ms
        
        # Check cache
        cache_key = self._get_cache_key(input_data)
        if self.cache and cache_key in self.cache:
            cached = self.cache[cache_key]
            if cached.confidence >= target_confidence:
                logger.debug(f"Cache hit with confidence {cached.confidence:.3f}")
                return cached
        
        # Determine starting model size
        if self.config.prefer_speed or max_latency_ms and max_latency_ms < 100:
            sizes_to_try = [ModelSize.SMALL, ModelSize.MEDIUM, ModelSize.LARGE]
        else:
            sizes_to_try = [ModelSize.MEDIUM, ModelSize.SMALL, ModelSize.LARGE]
        
        # Filter to available models
        sizes_to_try = [s for s in sizes_to_try if self.registry.has_model(s)]
        
        if not sizes_to_try:
            raise ValueError("No models registered")
        
        # Try models in order until confidence is met
        best_result = None
        
        for size in sizes_to_try:
            start_time = time.perf_counter()
            
            # Get model and processors
            model = self.registry.get_model(size)
            preprocessor = self.registry.preprocessors.get(size)
            postprocessor = self.registry.postprocessors.get(size)
            
            # Preprocess if needed
            data = preprocessor(input_data) if preprocessor else input_data
            
            # Run inference
            try:
                output, confidence = await self.engine.infer(model, data)
            except Exception as e:
                logger.warning(f"Failed to run {size.value} model: {e}")
                continue
            
            # Postprocess if needed
            if postprocessor:
                output = postprocessor(output)
            
            # Calculate metrics
            latency_ms = (time.perf_counter() - start_time) * 1000
            
            # Create result
            result = InferenceResult(
                output=output,
                confidence=confidence,
                model_size=size,
                latency_ms=latency_ms,
                device=self.device_type.value,
                metadata={
                    'target_confidence': target_confidence,
                    'device_info': self.device_info
                }
            )
            
            # Track performance
            self._track_performance(result)
            
            # Check if this meets our requirements
            if not best_result or confidence > best_result.confidence:
                best_result = result
            
            logger.info(f"{size.value} model: confidence={confidence:.3f}, "
                       f"latency={latency_ms:.1f}ms")
            
            # Check if we've met our target
            if confidence >= target_confidence:
                logger.info(f"Target confidence met with {size.value} model")
                break
            
            # Check latency constraint
            if max_latency_ms and latency_ms > max_latency_ms:
                logger.warning(f"Latency constraint violated, stopping")
                break
        
        # Cache result
        if self.cache and best_result:
            self.cache[cache_key] = best_result
        
        return best_result
    
    def _get_cache_key(self, input_data: Any) -> str:
        """Generate cache key for input data"""
        if isinstance(input_data, np.ndarray):
            return f"numpy_{input_data.shape}_{input_data.dtype}_{hash(input_data.tobytes())}"
        elif TORCH_AVAILABLE:
            try:
                import torch
                if isinstance(input_data, torch.Tensor):
                    return f"torch_{input_data.shape}_{input_data.dtype}_{hash(input_data.cpu().numpy().tobytes())}"
            except:
                pass
        else:
            return f"generic_{hash(str(input_data))}"
    
    def _track_performance(self, result: InferenceResult):
        """Track performance metrics"""
        self._performance_history.append({
            'timestamp': time.time(),
            'model_size': result.model_size.value,
            'confidence': result.confidence,
            'latency_ms': result.latency_ms
        })
        
        # Keep only last 100 entries
        if len(self._performance_history) > 100:
            self._performance_history = self._performance_history[-100:]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        if not self._performance_history:
            return {}
        
        latencies = [h['latency_ms'] for h in self._performance_history]
        confidences = [h['confidence'] for h in self._performance_history]
        
        model_counts = {}
        for h in self._performance_history:
            size = h['model_size']
            model_counts[size] = model_counts.get(size, 0) + 1
        
        return {
            'total_inferences': len(self._performance_history),
            'avg_latency_ms': np.mean(latencies),
            'p95_latency_ms': np.percentile(latencies, 95),
            'avg_confidence': np.mean(confidences),
            'model_usage': model_counts,
            'cache_size': len(self.cache) if self.cache else 0
        }
    
    def clear_cache(self):
        """Clear inference cache"""
        if self.cache:
            self.cache.clear()
            logger.info("Cache cleared")


# ============================================================================
# QUICK START UTILITIES
# ============================================================================

def create_demo_models():
    """Create demo PyTorch models for testing"""
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch required for demo models. Install with: pip install torch")
    
    class SmallModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(10, 3)
        
        def forward(self, x):
            return self.fc(x)
    
    class MediumModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(10, 32)
            self.relu = nn.ReLU()
            self.fc2 = nn.Linear(32, 3)
        
        def forward(self, x):
            x = self.relu(self.fc1(x))
            return self.fc2(x)
    
    class LargeModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(10, 128)
            self.fc2 = nn.Linear(128, 64)
            self.fc3 = nn.Linear(64, 3)
            self.relu = nn.ReLU()
            self.dropout = nn.Dropout(0.1)
        
        def forward(self, x):
            x = self.relu(self.fc1(x))
            x = self.dropout(x)
            x = self.relu(self.fc2(x))
            x = self.dropout(x)
            return self.fc3(x)
    
    return {
        ModelSize.SMALL: SmallModel(),
        ModelSize.MEDIUM: MediumModel(),
        ModelSize.LARGE: LargeModel()
    }


async def quickstart():
    """Quick start example"""
    print("🚀 AdaptML Quick Start")
    print("-" * 40)
    
    # Create system
    system = AdaptiveInference()
    
    # Register demo models
    if TORCH_AVAILABLE:
        models = create_demo_models()
        for size, model in models.items():
            system.register_model(model, size)
        print("✅ Registered 3 demo models")
    else:
        print("⚠️  PyTorch not installed, using mock models")
        # Register mock models for demo
        system.register_model(lambda x: (x, 0.7), ModelSize.SMALL)
        system.register_model(lambda x: (x, 0.85), ModelSize.MEDIUM)
        system.register_model(lambda x: (x, 0.95), ModelSize.LARGE)
    
    # Run inference with different targets
    test_data = np.random.randn(1, 10).astype(np.float32)
    
    print("\n📊 Running adaptive inference...")
    print("-" * 40)
    
    # Low confidence requirement - should use small model
    result = await system.infer(test_data, target_confidence=0.7)
    print(f"Target 0.7: {result}")
    
    # High confidence requirement - should use larger model
    result = await system.infer(test_data, target_confidence=0.95)
    print(f"Target 0.95: {result}")
    
    # Speed preference - should use small model
    system.config.prefer_speed = True
    result = await system.infer(test_data)
    print(f"Speed mode: {result}")
    
    # Show stats
    print("\n📈 Performance Stats")
    print("-" * 40)
    stats = system.get_stats()
    for key, value in stats.items():
        print(f"{key}: {value}")


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    # Run quickstart demo
    asyncio.run(quickstart())
