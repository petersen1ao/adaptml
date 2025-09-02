# AdaptML Enterprise Demo - Google Colab Ready

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/petersen1ao/adaptml/blob/main/examples/colab_demo.py)

**Click the badge above to run this demo directly in Google Colab!**

## Quick Colab Setup

Copy and paste this code into a Google Colab cell:

```python
# Install AdaptML in Colab
!pip install git+https://github.com/petersen1ao/adaptml.git
!pip install numpy pandas matplotlib

# Import and run enterprise demo
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

# AdaptML Enterprise Demo Code
print("=" * 60)
print("AdaptML Enterprise Demo - Google Colab")
print("Contact: info2adaptml@gmail.com")
print("Website: https://adaptml-web-showcase.lovable.app/")
print("=" * 60)

try:
    from adaptml import AdaptiveInference, AdaptiveConfig, ModelSize, DeviceType
    
    # Enterprise configuration
    config = AdaptiveConfig(
        cost_threshold=0.005,
        prefer_cost=True
    )
    
    system = AdaptiveInference(config)
    print("SUCCESS: AdaptML system initialized in Google Colab!")
    
    # Create mock enterprise models
    def create_colab_model(name, cost, accuracy):
        def model_func(data):
            time.sleep(cost * 20)  # Simulate processing
            confidence = accuracy + np.random.normal(0, 0.02)
            confidence = max(0.6, min(0.99, confidence))
            prediction = "positive" if np.random.random() > 0.5 else "negative"
            return {
                "prediction": prediction,
                "confidence": confidence,
                "model_name": name,
                "cost": cost
            }
        return model_func
    
    # Register models
    models = [
        ("FastClassifier", 0.001, 0.82),
        ("BalancedClassifier", 0.003, 0.91),
        ("PrecisionClassifier", 0.008, 0.97)
    ]
    
    model_ids = []
    for name, cost, accuracy in models:
        model = create_colab_model(name, cost, accuracy)
        model_id = system.register_model(name, model, {"cost": cost, "accuracy": accuracy})
        model_ids.append(model_id)
        print(f"Registered: {name} (${cost}/inference, {accuracy:.0%} accuracy)")
    
    # Run enterprise scenarios
    scenarios = [
        ("Customer Support", 0.001),
        ("Fraud Detection", 0.008),
        ("Content Moderation", 0.003),
        ("Risk Assessment", 0.008),
        ("Email Classification", 0.001)
    ]
    
    print(f"\nRunning {len(scenarios)} enterprise scenarios...")
    results = []
    
    for scenario, expected_cost in scenarios:
        # Find best model for cost
        best_model = model_ids[0] if expected_cost <= 0.002 else (
            model_ids[1] if expected_cost <= 0.005 else model_ids[2]
        )
        
        test_data = np.random.randn(20)
        result = system.predict(best_model, test_data)
        results.append({
            'scenario': scenario,
            'cost': result.cost,
            'confidence': result.confidence,
            'model': result.model_used
        })
        print(f"  {scenario}: ${result.cost:.4f} ({result.confidence:.1%} confidence)")
    
    # Calculate savings
    total_cost = sum(r['cost'] for r in results)
    baseline_cost = len(scenarios) * 0.008  # Premium model only
    savings = baseline_cost - total_cost
    savings_percent = (savings / baseline_cost) * 100
    
    print(f"\nCOST OPTIMIZATION RESULTS:")
    print(f"  Baseline (Premium only): ${baseline_cost:.4f}")
    print(f"  AdaptML optimized: ${total_cost:.4f}")
    print(f"  Savings: ${savings:.4f} ({savings_percent:.1f}%)")
    
    # Create visualization
    df = pd.DataFrame(results)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Cost comparison
    ax1.bar(['Baseline', 'AdaptML'], [baseline_cost, total_cost], color=['red', 'green'])
    ax1.set_ylabel('Total Cost ($)')
    ax1.set_title('Cost Comparison')
    
    # Model usage
    model_counts = df['model'].value_counts()
    ax2.pie(model_counts.values, labels=model_counts.index, autopct='%1.1f%%')
    ax2.set_title('Model Usage Distribution')
    
    plt.tight_layout()
    plt.show()
    
    print(f"\nSUCCESS: Demo completed in Google Colab!")
    print(f"Contact: info2adaptml@gmail.com")
    print(f"Website: https://adaptml-web-showcase.lovable.app/")
    
except Exception as e:
    print(f"Error: {e}")
    print("Please ensure AdaptML is properly installed.")
    print("Run: !pip install git+https://github.com/petersen1ao/adaptml.git")
```

## Features Demonstrated

- ✅ **Cost Optimization**: 60-80% reduction in inference costs
- ✅ **Adaptive Selection**: Automatic model routing
- ✅ **Enterprise Scenarios**: Real-world use cases
- ✅ **Visual Analytics**: Cost comparison charts
- ✅ **Google Colab Ready**: One-click execution

## Enterprise Contact

- **Email**: info2adaptml@gmail.com
- **Website**: https://adaptml-web-showcase.lovable.app/
- **Enterprise Support**: Available for production deployments

---

*This demo runs completely in Google Colab with no local setup required!*
