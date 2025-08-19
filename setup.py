"""
AdaptML - Cut AI inference costs by 50% with adaptive model selection
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

# Optional dependencies for different ML frameworks
extras_require = {
    "torch": ["torch>=1.9.0", "torchvision>=0.10.0"],
    "tensorflow": ["tensorflow>=2.6.0"],
    "onnx": ["onnxruntime>=1.8.0"],
    "all": ["torch>=1.9.0", "torchvision>=0.10.0", "tensorflow>=2.6.0", "onnxruntime>=1.8.0"],
    "dev": ["pytest>=6.0", "black>=21.0", "flake8>=3.9", "mypy>=0.910", "pre-commit>=2.15"],
}

setup(
    name="adaptml",
    version="0.1.0",
    author="AdaptML Team",
    author_email="info2adaptml@gmail.com",
    description="Cut AI inference costs by 50% with adaptive model selection",
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
    python_requires=">=3.7",
    install_requires=requirements,
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
