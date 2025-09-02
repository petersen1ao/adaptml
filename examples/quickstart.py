"""
AdaptML - Quick Start Example
Demonstrates basic usage of the adaptive inference system
"""

import asyncio
import numpy as np
from adaptml import AdaptiveInference, ModelSize, AdaptiveConfig


async def main():
    """Quick start example showing adaptive inference in action"""
    
    print("AdaptML Quick Start")
    print("=" * 50)
    
    # 1. Create the adaptive inference system
    config = AdaptiveConfig(
        target_confidence=0.9,
        prefer_speed=False,
        enable_caching=True
    )
    system = AdaptiveInference(config)
    print(f"Initialized system on {system.device_type.value}")
    
    # 2. Create some demo models (if PyTorch is available)
    try:
        from adaptml import create_demo_models
        models = create_demo_models()
        
        for size, model in models.items():
            system.register_model(model, size)
        
        print("Registered 3 demo PyTorch models")
    except ImportError:
        print("PyTorch not available, using mock models")
        # Register mock models for demo
        system.register_model(lambda x: (x * 0.1, 0.7), ModelSize.SMALL)
        system.register_model(lambda x: (x * 0.5, 0.85), ModelSize.MEDIUM)
        system.register_model(lambda x: (x * 1.0, 0.95), ModelSize.LARGE)
        print("Registered 3 mock models")
    
    # 3. Generate test data
    test_data = np.random.randn(1, 10).astype(np.float32)
    print(f"Generated test data: {test_data.shape}")
    
    print("\n" + "=" * 50)
    print("Running AdaptML Examples")
    print("=" * 50)
    
    # 4. Example 1: Low confidence requirement (should use small model)
    print("\n🎯 Example 1: Low confidence requirement (0.7)")
    result = await system.infer(test_data, target_confidence=0.7)
    print(f"   Used: {result.model_size.value} model")
    print(f"   Confidence: {result.confidence:.1%}")
    print(f"   Latency: {result.latency_ms:.1f}ms")
    
    # 5. Example 2: High confidence requirement (should use larger model)
    print("\n🎯 Example 2: High confidence requirement (0.95)")
    result = await system.infer(test_data, target_confidence=0.95)
    print(f"   Used: {result.model_size.value} model")
    print(f"   Confidence: {result.confidence:.1%}")
    print(f"   Latency: {result.latency_ms:.1f}ms")
    
    # 6. Example 3: Speed preference (should use small model)
    print("\n🎯 Example 3: Speed preference (max 50ms)")
    result = await system.infer(test_data, max_latency_ms=50)
    print(f"   Used: {result.model_size.value} model")
    print(f"   Confidence: {result.confidence:.1%}")
    print(f"   Latency: {result.latency_ms:.1f}ms")
    
    # 7. Example 4: Batch processing (multiple inferences)
    print("\n🎯 Example 4: Batch processing (5 inferences)")
    for i in range(5):
        data = np.random.randn(1, 10).astype(np.float32)
        result = await system.infer(data)
        print(f"   #{i+1}: {result.model_size.value} model, {result.confidence:.1%} confidence")
    
    # 8. Show performance statistics
    print("\n" + "=" * 50)
    print("📈 Performance Statistics")
    print("=" * 50)
    
    stats = system.get_stats()
    if stats:
        print(f"Total inferences: {stats['total_inferences']}")
        print(f"Average latency: {stats['avg_latency_ms']:.1f}ms")
        print(f"P95 latency: {stats['p95_latency_ms']:.1f}ms")
        print(f"Average confidence: {stats['avg_confidence']:.1%}")
        print(f"Model usage: {stats['model_usage']}")
        print(f"Cache size: {stats['cache_size']} items")
    else:
        print("No statistics available yet")
    
    # 9. Cost savings calculation (example)
    print("\n" + "=" * 50)
    print("💰 Estimated Cost Savings")
    print("=" * 50)
    
    if stats and stats['model_usage']:
        # Simulate cost calculation
        model_costs = {'small': 0.001, 'medium': 0.005, 'large': 0.02}  # $ per inference
        
        actual_cost = sum(
            stats['model_usage'].get(model, 0) * cost 
            for model, cost in model_costs.items()
        )
        
        # Cost if always using large model
        traditional_cost = stats['total_inferences'] * model_costs['large']
        
        savings = traditional_cost - actual_cost
        savings_percent = (savings / traditional_cost) * 100 if traditional_cost > 0 else 0
        
        print(f"Traditional cost (large model): ${traditional_cost:.4f}")
        print(f"Adaptive cost: ${actual_cost:.4f}")
        print(f"💰 Savings: ${savings:.4f} ({savings_percent:.1f}%)")
    
    print("\n🎉 Quick start complete!")
    print("Next steps:")
    print("  - Try with your own models")
    print("  - Experiment with different confidence targets")
    print("  - Check out the examples/ directory for more use cases")


if __name__ == "__main__":
    asyncio.run(main())
