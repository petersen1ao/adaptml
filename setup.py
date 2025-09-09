#!/usr/bin/env python3
"""
AdaptML Setup Configuration
==========================

Production-ready setup configuration for the AdaptML AI Optimization Platform.
"""

from setuptools import setup, find_packages
import os
import sys

# Ensure we're using Python 3.8+
if sys.version_info < (3, 8):
    print("ERROR: AdaptML requires Python 3.8 or higher")
    sys.exit(1)

# Read version from package
def get_version():
    """Get version from package __init__.py"""
    version_file = os.path.join(os.path.dirname(__file__), 'adaptml', '__init__.py')
    if os.path.exists(version_file):
        with open(version_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.startswith('__version__'):
                    return line.split('"')[1]
    return "2.0.1"

# Read README
def get_long_description():
    """Get long description from README file"""
    readme_file = os.path.join(os.path.dirname(__file__), 'README.md')
    if os.path.exists(readme_file):
        with open(readme_file, 'r', encoding='utf-8') as f:
            return f.read()
    return "AdaptML - Advanced AI Optimization Platform"

# Package requirements
INSTALL_REQUIRES = [
    # Core dependencies
    "asyncio-mqtt>=0.11.0",
    "aiofiles>=23.2.1",
    
    # Data processing
    "numpy>=1.21.0",
    "pandas>=1.3.0",
    
    # Async and concurrency
    "asyncio>=3.4.3",
    "concurrent-futures>=3.1.1",
    
    # Utilities
    "python-dateutil>=2.8.2",
    "pytz>=2023.3",
    "requests>=2.28.0",
    
    # Logging and monitoring
    "structlog>=23.1.0",
    
    # Configuration
    "pyyaml>=6.0",
    "python-dotenv>=1.0.0",
]

# Development dependencies
EXTRAS_REQUIRE = {
    'dev': [
        'pytest>=7.4.0',
        'pytest-asyncio>=0.21.1',
        'pytest-cov>=4.1.0',
        'black>=23.7.0',
        'isort>=5.12.0',
        'flake8>=6.0.0',
        'mypy>=1.5.0',
    ],
    'docs': [
        'sphinx>=7.1.0',
        'sphinx-rtd-theme>=1.3.0',
        'sphinxcontrib-asyncio>=0.3.0',
    ],
    'testing': [
        'pytest>=7.4.0',
        'pytest-asyncio>=0.21.1',
        'pytest-benchmark>=4.0.0',
        'memory-profiler>=0.61.0',
    ]
}

# All extra dependencies
EXTRAS_REQUIRE['all'] = list(set(sum(EXTRAS_REQUIRE.values(), [])))

setup(
    # Package metadata
    name="adaptml",
    version=get_version(),
    author="AdaptML Team",
    author_email="info2adaptml@gmail.com",
    description="Advanced AI Optimization Platform with 2.4x-3.0x performance improvements",
    long_description=get_long_description(),
    long_description_content_type="text/markdown",
    
    # URLs
    url="https://github.com/petersen1ao/adaptml",
    project_urls={
        "Bug Reports": "https://github.com/petersen1ao/adaptml/issues",
        "Source": "https://github.com/petersen1ao/adaptml",
        "Documentation": "https://adaptml-web-showcase.lovable.app/",
        "Changelog": "https://github.com/petersen1ao/adaptml/blob/main/CHANGELOG.md",
    },
    
    # Package discovery
    packages=find_packages(exclude=['tests*', 'docs*', 'examples*']),
    package_dir={'adaptml': 'adaptml'},
    
    # Include additional files
    include_package_data=True,
    package_data={
        'adaptml': [
            '*.py',
            'config/*.yaml',
            'config/*.json',
        ],
    },
    
    # Dependencies
    python_requires=">=3.8",
    install_requires=INSTALL_REQUIRES,
    extras_require=EXTRAS_REQUIRE,
    
    # Entry points
    entry_points={
        'console_scripts': [
            'adaptml-demo=adaptml.adaptml_production_system:main',
            'adaptml=adaptml:main',
        ],
    },
    
    # Classification
    classifiers=[
        # Development Status
        "Development Status :: 5 - Production/Stable",
        
        # Intended Audience
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Information Technology",
        
        # License
        "License :: Other/Proprietary License",
        
        # Programming Language
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3 :: Only",
        
        # Topic
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: System :: Systems Administration",
        "Topic :: Internet :: WWW/HTTP :: Dynamic Content",
        "Topic :: Software Development :: Libraries :: Application Frameworks",
        
        # Operating System
        "Operating System :: OS Independent",
        
        # Natural Language
        "Natural Language :: English",
    ],
    
    # Keywords
    keywords=[
        "ai", "optimization", "llm", "qlora", "performance", 
        "preprocessing", "meta-router", "transformer", "quantization",
        "adaptive-learning", "machine-learning", "deep-learning",
        "neural-networks", "artificial-intelligence"
    ],
    
    # Project status
    zip_safe=False,
    
    # Minimum version requirements
    setup_requires=[
        "setuptools>=45",
        "wheel>=0.37.0",
    ],
)

# Print installation summary
if __name__ == "__main__":
    print("🚀 AdaptML Setup Configuration")
    print("=" * 40)
    print(f"📦 Package: adaptml v{get_version()}")
    print(f"🐍 Python: {sys.version}")
    print(f"📧 Contact: info2adaptml@gmail.com")
    print(f"🌐 Repository: https://github.com/petersen1ao/adaptml")
    print()
    print("✅ Ready for installation with:")
    print("   pip install -e .")
    print("   pip install -e .[dev]")
    print("   pip install -e .[all]")
