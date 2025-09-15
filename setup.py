"""
LLM Token Analytics Library
===========================
A comprehensive Python library for analyzing LLM token usage patterns,
retrieving provider data, and running robust pricing simulations.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="llm-token-analytics",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Robust simulation and analytics for LLM token pricing",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/llm-token-analytics",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "scipy>=1.7.0",
        "scikit-learn>=1.0.0",
        "statsmodels>=0.13.0",
        "tqdm>=4.62.0",
        "requests>=2.26.0",
        "python-dotenv>=0.19.0",
        "openai>=1.0.0",
        "anthropic>=0.18.0",
        "google-cloud-aiplatform>=1.25.0",
        "click>=8.0.0",
        "rich>=10.0.0",
        "plotly>=5.3.0",
        "pyyaml>=6.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.2.0",
            "pytest-cov>=3.0.0",
            "black>=21.0",
            "pylint>=2.11.0",
            "mypy>=0.910",
        ],
        "viz": [
            "matplotlib>=3.4.0",
            "seaborn>=0.11.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "llm-analytics=llm_token_analytics.cli:main",
        ],
    },
    include_package_data=True,
    package_data={
        "llm_token_analytics": ["data/*.json", "configs/*.yaml"],
    },
)
