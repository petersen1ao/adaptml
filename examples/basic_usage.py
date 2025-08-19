#!/usr/bin/env python3
"""
Basic AdaptML Usage Example

This example shows how to set up AdaptML for cost-effective inference.
Contact: info2adaptml@gmail.com
Website: https://adaptml-web-showcase.lovable.app/
"""

from adaptml import AdaptiveInference, AdaptiveConfig

def main():
    print("🚀 AdaptML Basic Usage Example")
    print("=" * 40)
    
    # Configure for cost optimization
    config = AdaptiveConfig(
        cost_threshold=0.01,
        prefer_cost=True
    )
    
    # Initialize inference system
    inference = AdaptiveInference(config)
    
    # Register models (using mock models for demo)
    small_model_id = inference.register_model(
        name="efficient-classifier",
        model_data={"type": "mock", "size": "small"},
        metadata={"cost_per_inference": 0.001}
    )
    
    large_model_id = inference.register_model(
        name="accurate-classifier",
        model_data={"type": "mock", "size": "large"}, 
        metadata={"cost_per_inference": 0.01}
    )
    
    # Run inference
    test_inputs = [
        "Simple classification task",
        "Complex multi-step analysis",
        "Quick sentiment analysis"
    ]
    
    total_cost = 0
    
    for i, input_text in enumerate(test_inputs, 1):
        result = inference.predict(small_model_id, input_text)
        total_cost += result.cost
        
        print(f"\n{i}. Input: '{input_text}'")
        print(f"   Prediction: {result.prediction}")
        print(f"   Model: {result.model_used}")
        print(f"   Cost: ${result.cost:.6f}")
        print(f"   Latency: {result.latency:.3f}s")
    
    print(f"\n💰 Total cost: ${total_cost:.6f}")
    print(f"📊 Average cost per request: ${total_cost/len(test_inputs):.6f}")
    
    # Show system stats
    stats = inference.get_stats()
    print(f"\n📈 System Statistics:")
    print(f"   Models registered: {stats['registered_models']}")
    print(f"   Available engines: {', '.join(stats['available_engines'])}")
    
    print(f"\n📞 Contact us:")
    print(f"   📧 Email: info2adaptml@gmail.com")
    print(f"   🌐 Website: https://adaptml-web-showcase.lovable.app/")

if __name__ == "__main__":
    main()
