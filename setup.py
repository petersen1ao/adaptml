"""
AdaptML - Cut AI inference costs by 50% with adaptive model selection + Unified QLoRA Security
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Core requirements
core_requirements = [
    "numpy>=1.21.0",
    "psutil>=5.8.0",
]

# Unified QLoRA System requirements
unified_qlora_requirements = [
    "torch>=2.0.0",
    "transformers>=4.35.0", 
    "peft>=0.6.0",
    "bitsandbytes>=0.41.0",
    "accelerate>=0.24.0",
    "datasets>=2.14.0",
    "cryptography>=41.0.0",
    "requests>=2.31.0",
]

# Optional dependencies for different ML frameworks
extras_require = {
    "core": core_requirements,
    "unified-qlora": core_requirements + unified_qlora_requirements,
    "torch": ["torch>=2.0.0", "torchvision>=0.15.0"],
    "tensorflow": ["tensorflow>=2.13.0"],
    "onnx": ["onnxruntime>=1.16.0"],
    "security": ["cryptography>=41.0.0", "requests>=2.31.0"],
    "all": core_requirements + unified_qlora_requirements + ["tensorflow>=2.13.0", "onnxruntime>=1.16.0"],
    "dev": ["pytest>=7.4.0", "black>=23.9.0", "isort>=5.12.0", "mypy>=1.5.0", "pre-commit>=3.4.0"],
}

setup(
    name="adaptml",
    version="0.2.0",  # Updated version for Unified QLoRA integration
    author="AdaptML Team",
    author_email="info2adaptml@gmail.com",
    description="Cut AI inference costs by 50% with adaptive model selection + Unified QLoRA Security",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/petersen1ao/adaptml",
    project_urls={
        "Bug Tracker": "https://github.com/petersen1ao/adaptml/issues",
        "Documentation": "https://adaptml.readthedocs.io",
        "Source Code": "https://github.com/petersen1ao/adaptml",
        "Website": "https://adaptml-web-showcase.lovable.app/",
        "Support": "mailto:info2adaptml@gmail.com",
    },
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=core_requirements,
    extras_require=extras_require,
    include_package_data=True,
    zip_safe=False,
    entry_points={
        "console_scripts": [
            "adaptml=adaptml.cli:main",
        ],
    },
    keywords=[
        "machine learning",
        "artificial intelligence",
        "adaptive inference",
        "cost optimization",
        "model selection",
        "pytorch",
        "tensorflow",
        "onnx"
    ],
)
