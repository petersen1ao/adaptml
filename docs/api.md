# AdaptML API Reference

## Core Classes

### AdaptiveInference

The main class for adaptive ML inference.

```python
from adaptml import AdaptiveInference, AdaptiveConfig

# Initialize with default config
inference = AdaptiveInference()

# Initialize with custom config
config = AdaptiveConfig(cost_threshold=0.01, prefer_cost=True)
inference = AdaptiveInference(config)
```

#### Methods

- `register_model(name, model_data, metadata=None)` - Register a model for inference
- `predict(model_id, input_data, **kwargs)` - Run inference on input data
- `get_stats()` - Get system statistics

### AdaptiveConfig

Configuration class for inference behavior.

```python
config = AdaptiveConfig(
    cost_threshold=0.01,        # Maximum cost per inference
    latency_threshold=1.0,      # Maximum latency in seconds
    quality_threshold=0.8,      # Minimum quality requirement
    prefer_speed=False,         # Optimize for speed
    prefer_accuracy=False,      # Optimize for accuracy
    prefer_cost=True,           # Optimize for cost (default)
    device_preference=None      # Preferred device type
)
```

## Contact Information

- **Email**: info2adaptml@gmail.com
- **Website**: https://adaptml-web-showcase.lovable.app/
- **GitHub**: https://github.com/petersen1ao/adaptml
