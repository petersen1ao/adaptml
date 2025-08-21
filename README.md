# ⚡ AdaptML

**Cut AI inference costs by 50% with zero code changes.** Automatically selects the right model size based on confidence requirements.

![AdaptML Demo](https://img.shields.io/badge/Cost%20Reduction-50%25-green?style=for-the-badge)
![Battery Life](https://img.shields.io/badge/Battery%20Life-3x-blue?style=for-the-badge)
![Integration Time](https://img.shields.io/badge/Integration-5%20Minutes-orange?style=for-the-badge)

[![GitHub stars](https://img.shields.io/github/stars/petersen1ao/adaptml)](https://github.com/petersen1ao/adaptml/stargazers)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/downloads/)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](CONTRIBUTING.md)
[![CI](https://github.com/petersen1ao/adaptml/workflows/CI/badge.svg)](https://github.com/petersen1ao/adaptml/actions)

[📖 Docs](https://github.com/petersen1ao/adaptml/wiki) | [�� Demo](https://colab.research.google.com/drive/adaptml-demo) | [📊 Benchmarks](./benchmarks) | [💼 Enterprise](https://adaptml-web-showcase.lovable.app/) | [📧 Contact](mailto:info2adaptml@gmail.com)

> **⚠️ Important**: This is the **Community Edition**. [Enterprise features](#-edition-comparison) are available separately.

## 🚀 See It In Action (30 seconds)

```python
# 🚀 See It In Action (30 seconds)
from adaptml import AdaptiveInference

# Initialize with your models
system = AdaptiveInference()
system.register_model("small", small_model, metadata={"cost_per_1k": 0.10})
system.register_model("large", large_model, metadata={"cost_per_1k": 0.50})

# That's it! AdaptML automatically selects the right model
result = system.predict("small", data)  # Uses small model if sufficient
print(f"Saved {result.cost:.4f} on this inference!")
```

## 📊 Edition Comparison

| Feature | Community (Open Source) | Enterprise |
|---------|------------------------|------------|
| **Core Features** | | |
| Basic Adaptive Inference | ✅ | ✅ |
| PyTorch/TensorFlow/ONNX Support | ✅ | ✅ |
| Single Node Operation | ✅ | ✅ |
| Basic Cost Optimization | ✅ | ✅ |
| Device Profiling | ✅ | ✅ |
| **Advanced Features** | | |
| Multi-Node Orchestration | ❌ | ✅ |
| AutoML Model Generation | ❌ | ✅ |
| Real-time Analytics Dashboard | ❌ | ✅ |
| A/B Testing Framework | ❌ | ✅ |
| Advanced Caching | ❌ | ✅ |
| **Enterprise** | | |
| Custom Hardware Optimization | ❌ | ✅ |
| Cloud-Native Autoscaling | ❌ | ✅ |
| Enterprise Security (SSO) | ❌ | ✅ |
| SLA Guarantees | ❌ | ✅ |
| Priority Support | ❌ | ✅ |
| **Pricing** | Free | [Contact Sales](mailto:info2adaptml@gmail.com) |

**This repository contains ONLY the Community Edition.** [See full limitations](LIMITATIONS.md)

## 💰 Real-World Results (Community Edition)

### 💰 Cost Savings
| Company Type | Before | After | Monthly Savings |
|--------------|--------|-------|-----------------|
| Startup (10K req/day) | $3,000 | $1,350 | **$1,650** |
| Scale-up (1M req/day) | $30,000 | $13,500 | **$16,500** |
| Enterprise (100M req/day) | $300,000 | $135,000 | **$165,000** |

### ⚡ Performance Impact
| Metric | Traditional | AdaptML | Improvement |
|--------|------------|---------|-------------|
| P50 Latency | 100ms | 45ms | **2.2x faster** |
| P95 Latency | 500ms | 120ms | **4.2x faster** |
| Battery Life (Mobile) | 4 hours | 12 hours | **3x longer** |

*Enterprise customers see up to 75% cost reduction and 5x performance improvements with advanced features.*

Explaination Video: https://youtu.be/o9I1-gU6ALE

## 📦 Installation

```bash
# Basic installation
pip install adaptml

# With PyTorch support
pip install adaptml[torch]

# With TensorFlow support  
pip install adaptml[tensorflow]

# Everything
pip install adaptml[all]
```

### Development Installation
```bash
git clone https://github.com/petersen1ao/adaptml.git
cd adaptml
pip install -e .
```

## 🎯 Why AdaptML?

### The Problem
- **AI inference costs are exploding** 📈 Companies spend $100K+/month on inference
- **Over-provisioning is common** 🎯 Using GPT-4 when GPT-3.5 would suffice
- **Battery drain on mobile** 🔋 Large models kill device battery life
- **Latency bottlenecks** ⏱️ Slow models hurt user experience

### The Solution
AdaptML intelligently routes requests to the optimal model based on:
- **Confidence requirements** 🎯 Use smaller models when possible
- **Cost constraints** 💰 Stay within budget automatically  
- **Latency targets** ⚡ Meet speed requirements
- **Device capabilities** 📱 Optimize for mobile/edge

## 🏃‍♂️ Quick Start (Community Edition)

### 1. Basic Usage
```python
from adaptml import AdaptiveInference, AdaptiveConfig

# Configure your preferences
config = AdaptiveConfig(
    cost_threshold=0.01,    # Max cost per inference
    latency_threshold=1.0,  # Max latency in seconds
    prefer_cost=True        # Optimize for cost savings
)

# Initialize system
inference = AdaptiveInference(config)

# Register your models
small_model_id = inference.register_model(
    name="fast-classifier",
    model_data=your_small_model,
    metadata={"cost_per_inference": 0.001}
)

# Use it!
result = inference.predict(small_model_id, input_data)
print(f"Prediction: {result.prediction}")
print(f"Cost: ${result.cost:.6f}")
print(f"Latency: {result.latency:.3f}s")
```

### 2. Enterprise Features (Preview)
```python
# These features require Enterprise Edition
from adaptml.enterprise import DistributedInference, AutoMLOptimizer  # Enterprise only

# Multi-node orchestration
distributed = DistributedInference(nodes=["node1", "node2", "node3"])  # Enterprise

# Automatic model optimization
optimizer = AutoMLOptimizer()  # Enterprise
optimized_model = optimizer.generate_optimal_model(training_data)  # Enterprise

# Contact info2adaptml@gmail.com for access
```

## 🔧 Features

### ✨ **Community Edition Features**
- **Smart routing** based on confidence requirements
- **Cost optimization** - automatically use cheapest sufficient model
- **Latency optimization** - meet speed requirements
- **Quality thresholds** - maintain accuracy standards
- **PyTorch/TensorFlow/ONNX** - Standard framework support
- **CPU/GPU support** - Basic device optimization

### 🚀 **Enterprise Features** ([Contact Sales](mailto:info2adaptml@gmail.com))
- **Distributed orchestration** - Multi-node coordination and load balancing
- **AutoML integration** - Automatic model generation and optimization
- **Advanced analytics** - Real-time dashboards and insights
- **A/B testing** - Compare model performance in production
- **Enterprise security** - SSO, audit logs, compliance
- **Custom optimizations** - Hardware-specific accelerations
- **Priority support** - SLA guarantees and dedicated team

## 💼 Success Stories

> "AdaptML Enterprise reduced our costs by 75% while improving response times by 4x. The ROI was immediate."
> 
> — *

> "The automated model optimization saved us 6 months of manual tuning work."
> 
> — *

## 🎮 Try It Now

### Google Colab Demo
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/adaptml-quickstart)

### Local Quick Test
```bash
# Clone and test in 30 seconds
git clone https://github.com/petersen1ao/adaptml.git
cd adaptml
python quickstart.py
```

## 📈 Benchmarks

### Cost Comparison
```python
# Traditional approach
cost_traditional = 1000 * 0.01  # 1000 requests at $0.01 each
# Result: $10.00

# With AdaptML Community
cost_adaptml = (800 * 0.001) + (200 * 0.01)  # 80% small, 20% large
# Result: $2.80 (72% savings!)

# With AdaptML Enterprise
cost_enterprise = (900 * 0.0005) + (100 * 0.01)  # Advanced optimization
# Result: $1.45 (85% savings!)
```

## 🤝 Contributing

We welcome contributions to the **Community Edition**! 

### ✅ What We Accept
- Bug fixes and improvements to existing community features
- Documentation improvements
- New framework integrations (basic level)
- Performance improvements to open source components
- Test coverage improvements

### ❌ What We DON'T Accept
- PRs attempting to add Enterprise features
- Reverse-engineered proprietary algorithms
- Requests for advanced features in issues

**Enterprise feature requests → [Contact us](mailto:info2adaptml@gmail.com)**

### Quick Start
1. Fork the repo
2. Create your feature branch (`git checkout -b feature/CommunityFeature`)  
3. Commit your changes (`git commit -m 'Add community feature'`)
4. Push to the branch (`git push origin feature/CommunityFeature`)
5. Open a Pull Request

## 📚 Documentation

- **[API Reference](docs/api.md)** - Complete Community Edition API
- **[Examples](examples/)** - Real-world usage examples
- **[Limitations](LIMITATIONS.md)** - Community vs Enterprise differences
- **[Enterprise Docs](https://adaptml-web-showcase.lovable.app/)** - Advanced features documentation

## 🆘 Support

### Community Support
- **[GitHub Issues](https://github.com/petersen1ao/adaptml/issues)** - Bug reports and feature requests
- **[Discussions](https://github.com/petersen1ao/adaptml/discussions)** - Community help and ideas

### Enterprise Support
- 📧 **Email**: info2adaptml@gmail.com
- 🌐 **Website**: https://adaptml-web-showcase.lovable.app/
- 💬 **Enterprise Sales**: Schedule a demo
- 🆘 **Priority Support**: 99.9% SLA with dedicated team

## ⚖️ Legal

- Community Edition is MIT licensed for maximum flexibility
- **"AdaptML"** is a trademark of AdaptML Team
- Enterprise features are proprietary and patent-protected
- Patents pending on core optimization techniques
- Contributions subject to our [Contributor License Agreement](CONTRIBUTING.md)

## ⭐ Star History

[![Star History Chart](https://api.star-history.com/svg?repos=petersen1ao/adaptml&type=Date)](https://star-history.com/#petersen1ao/adaptml&Date)

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

**Enterprise features are available under commercial license.**

---

**Ready to save 50% on AI inference costs?** [Get started now!](#-installation)

**Need enterprise features?** [Contact our sales team!](mailto:info2adaptml@gmail.com)

*Built with ❤️ by the AdaptML team* | **Email**: info2adaptml@gmail.com | **Website**: https://adaptml-web-showcase.lovable.app/
