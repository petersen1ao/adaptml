#!/usr/bin/env python3
"""
AdaptML Quickstart - Fixed Version
Enterprise-ready demonstration with proper API usage.

Contact: info2adaptml@gmail.com
Website: https://adaptml-web-showcase.lovable.app/
"""

import numpy as np
import time
from adaptml import AdaptiveInference, AdaptiveConfig, ModelSize, DeviceType

def create_simple_mock_model(name, cost, accuracy):
    """Create a simple mock model for demonstration"""
    def mock_prediction(input_data):
        # Simulate processing time based on cost
        time.sleep(cost * 10)  # Scale down for demo
        
        # Generate mock prediction
        confidence = accuracy + np.random.normal(0, 0.02)
        confidence = max(0.6, min(0.99, confidence))
        
        prediction = "positive" if np.random.random() > 0.5 else "negative"
        
        return prediction, confidence
    
    return mock_prediction

def main():
    print("AdaptML Quickstart Demo")
    print("=" * 50)
    
    # Create the adaptive inference system
    config = AdaptiveConfig(
        cost_threshold=0.01,
        prefer_cost=True
    )
    system = AdaptiveInference(config)
    print("System initialized successfully")
    
    # Register demo models
    print("\nRegistering demo models...")
    
    # Create mock models with different characteristics
    fast_model = create_simple_mock_model("fast-classifier", 0.001, 0.80)
    balanced_model = create_simple_mock_model("balanced-classifier", 0.005, 0.90)
    accurate_model = create_simple_mock_model("accurate-classifier", 0.01, 0.95)
    
    # Register models
    fast_id = system.register_model("fast-classifier", fast_model, {"cost": 0.001, "accuracy": 0.80})
    balanced_id = system.register_model("balanced-classifier", balanced_model, {"cost": 0.005, "accuracy": 0.90})
    accurate_id = system.register_model("accurate-classifier", accurate_model, {"cost": 0.01, "accuracy": 0.95})
    
    print("Registered 3 demo models:")
    print(f"  Fast Model: {fast_id}")
    print(f"  Balanced Model: {balanced_id}")
    print(f"  Accurate Model: {accurate_id}")
    
    # Generate test data
    test_data = np.random.randn(10).astype(np.float32)
    print(f"\nGenerated test data: shape {test_data.shape}")
    
    print("\n" + "=" * 50)
    print("Running Inference Examples")
    print("=" * 50)
    
    # Example 1: Cost-optimized inference
    print("\n1. Cost-optimized inference:")
    result1 = system.predict(fast_id, test_data)
    print(f"   Prediction: {result1.prediction}")
    print(f"   Confidence: {result1.confidence:.2%}")
    print(f"   Cost: ${result1.cost:.4f}")
    print(f"   Latency: {result1.latency:.3f}s")
    print(f"   Model: {result1.model_used}")
    
    # Example 2: Balanced inference
    print("\n2. Balanced inference:")
    result2 = system.predict(balanced_id, test_data)
    print(f"   Prediction: {result2.prediction}")
    print(f"   Confidence: {result2.confidence:.2%}")
    print(f"   Cost: ${result2.cost:.4f}")
    print(f"   Latency: {result2.latency:.3f}s")
    print(f"   Model: {result2.model_used}")
    
    # Example 3: High-accuracy inference
    print("\n3. High-accuracy inference:")
    result3 = system.predict(accurate_id, test_data)
    print(f"   Prediction: {result3.prediction}")
    print(f"   Confidence: {result3.confidence:.2%}")
    print(f"   Cost: ${result3.cost:.4f}")
    print(f"   Latency: {result3.latency:.3f}s")
    print(f"   Model: {result3.model_used}")
    
    # Batch processing example
    print("\n4. Batch processing:")
    batch_data = [
        np.random.randn(10).astype(np.float32) for _ in range(5)
    ]
    
    total_cost = 0
    total_time = 0
    
    for i, data in enumerate(batch_data):
        start_time = time.time()
        result = system.predict(fast_id, data)  # Use fast model for batch
        processing_time = time.time() - start_time
        
        total_cost += result.cost
        total_time += processing_time
        
        print(f"   Batch {i+1}: {result.prediction} (${result.cost:.4f}, {processing_time:.3f}s)")
    
    print(f"\n   Batch Summary:")
    print(f"   Total cost: ${total_cost:.4f}")
    print(f"   Total time: {total_time:.3f}s")
    print(f"   Average cost: ${total_cost/len(batch_data):.4f}")
    print(f"   Average time: {total_time/len(batch_data):.3f}s")
    
    # System statistics
    stats = system.get_stats()
    print(f"\nSystem Statistics:")
    print(f"   Models registered: {stats['registered_models']}")
    print(f"   Total predictions: {stats.get('total_predictions', 'N/A')}")
    
    print(f"\nContact Information:")
    print(f"   Email: info2adaptml@gmail.com")
    print(f"   Website: https://adaptml-web-showcase.lovable.app/")
    
    print("\nQuickstart completed successfully!")

if __name__ == "__main__":
    main()
