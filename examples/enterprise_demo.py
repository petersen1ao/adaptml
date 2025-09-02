#!/usr/bin/env python3
"""
AdaptML - Enterprise Demo
Demonstrates cost-effective adaptive inference for enterprise applications.

Contact: info2adaptml@gmail.com
Website: https://adaptml-web-showcase.lovable.app/
"""

import numpy as np
import time
from adaptml import AdaptiveInference, AdaptiveConfig, ModelSize, DeviceType


def main():
    """Enterprise demonstration of AdaptML capabilities"""
    
    print("=" * 60)
    print("AdaptML Enterprise Demo - Cost Optimization Platform")
    print("=" * 60)
    
    # Enterprise configuration for cost optimization
    config = AdaptiveConfig(
        cost_threshold=0.005,  # $0.005 per inference
        prefer_cost=True,
        enable_caching=True,
        device_preference=DeviceType.AUTO
    )
    
    # Initialize the adaptive inference system
    system = AdaptiveInference(config)
    
    print(f"SYSTEM INITIALIZED:")
    print(f"  Device: {system.device_type.value}")
    print(f"  Cost threshold: ${config.cost_threshold}")
    print(f"  Caching enabled: {config.enable_caching}")
    
    # Register enterprise models with different cost/performance profiles
    print(f"\nREGISTERING ENTERPRISE MODELS:")
    
    # Small model - fast and cheap
    small_model = create_mock_model("lightweight-classifier", cost=0.001, accuracy=0.85)
    small_id = system.register_model(small_model, ModelSize.SMALL)
    print(f"  Small Model: $0.001/inference, 85% accuracy")
    
    # Medium model - balanced
    medium_model = create_mock_model("balanced-classifier", cost=0.003, accuracy=0.92)
    medium_id = system.register_model(medium_model, ModelSize.MEDIUM)
    print(f"  Medium Model: $0.003/inference, 92% accuracy")
    
    # Large model - high accuracy but expensive
    large_model = create_mock_model("premium-classifier", cost=0.008, accuracy=0.98)
    large_id = system.register_model(large_model, ModelSize.LARGE)
    print(f"  Large Model: $0.008/inference, 98% accuracy")
    
    # Enterprise test scenarios
    test_scenarios = [
        ("Simple customer query", "low_complexity"),
        ("Technical support analysis", "medium_complexity"),
        ("Complex fraud detection", "high_complexity"),
        ("Routine data classification", "low_complexity"),
        ("Multi-factor risk assessment", "high_complexity")
    ]
    
    print(f"\nRUNNING ENTERPRISE TEST SCENARIOS:")
    print(f"{'Scenario':<25} {'Model Used':<15} {'Cost':<8} {'Latency':<10} {'Accuracy'}")
    print("-" * 70)
    
    total_cost = 0
    total_time = 0
    
    for scenario, complexity in test_scenarios:
        # Create test data based on complexity
        test_data = generate_test_data(complexity)
        
        # Run adaptive inference
        start_time = time.time()
        result = system.predict(test_data)
        inference_time = time.time() - start_time
        
        total_cost += result.cost
        total_time += inference_time
        
        # Display results
        model_name = result.model_used.split('-')[0].capitalize()
        print(f"{scenario:<25} {model_name:<15} ${result.cost:<7.4f} {inference_time:<9.3f}s {result.confidence:.1%}")
    
    # Summary statistics
    print("-" * 70)
    print(f"\nENTERPRISE PERFORMANCE SUMMARY:")
    print(f"  Total scenarios processed: {len(test_scenarios)}")
    print(f"  Total cost: ${total_cost:.4f}")
    print(f"  Average cost per request: ${total_cost/len(test_scenarios):.4f}")
    print(f"  Total processing time: {total_time:.3f}s")
    print(f"  Average latency: {total_time/len(test_scenarios):.3f}s")
    
    # Cost savings analysis
    baseline_cost = len(test_scenarios) * 0.008  # If all used large model
    savings = baseline_cost - total_cost
    savings_percent = (savings / baseline_cost) * 100
    
    print(f"\nCOST OPTIMIZATION RESULTS:")
    print(f"  Baseline cost (large model only): ${baseline_cost:.4f}")
    print(f"  AdaptML optimized cost: ${total_cost:.4f}")
    print(f"  Cost savings: ${savings:.4f} ({savings_percent:.1f}%)")
    
    # System statistics
    stats = system.get_stats()
    print(f"\nSYSTEM STATISTICS:")
    print(f"  Models registered: {stats['registered_models']}")
    print(f"  Cache hits: {stats.get('cache_hits', 0)}")
    print(f"  Adaptive selections: {stats.get('adaptive_selections', len(test_scenarios))}")
    
    print(f"\nCONTACT INFORMATION:")
    print(f"  Email: info2adaptml@gmail.com")
    print(f"  Website: https://adaptml-web-showcase.lovable.app/")
    print(f"  Enterprise Support: Available for production deployments")
    
    print("=" * 60)


def create_mock_model(name, cost, accuracy):
    """Create a mock model for demonstration purposes"""
    def model_function(data):
        # Simulate model processing time based on cost
        time.sleep(cost * 100)  # Higher cost = more processing time
        
        # Generate mock prediction with specified accuracy
        confidence = accuracy + np.random.normal(0, 0.02)  # Small variance
        confidence = max(0.5, min(0.99, confidence))  # Keep in reasonable range
        
        prediction = "positive" if np.random.random() > 0.5 else "negative"
        
        return {
            "prediction": prediction,
            "confidence": confidence,
            "model_name": name,
            "cost": cost
        }
    
    return model_function


def generate_test_data(complexity):
    """Generate test data based on complexity level"""
    if complexity == "low_complexity":
        return np.random.randn(10).astype(np.float32)
    elif complexity == "medium_complexity":
        return np.random.randn(50).astype(np.float32)
    else:  # high_complexity
        return np.random.randn(100).astype(np.float32)


if __name__ == "__main__":
    main()
