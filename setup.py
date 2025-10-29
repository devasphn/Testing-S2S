#!/usr/bin/env python3
"""
Testing-S2S: Custom Speech-to-Speech AI Model
Combining GLM-4-Voice + Moshi Streaming + GPT-4o Architecture
"""

from setuptools import setup, find_packages
import os

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="testing-s2s",
    version="0.1.0",
    author="Deva Kumar",
    author_email="devasphn@gmail.com",
    description="Custom Speech-to-Speech AI Model with Ultra-Low Latency",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/devasphn/Testing-S2S",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Multimedia :: Sound/Audio :: Speech",
    ],
    python_requires=">=3.10",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0",
            "black>=22.0",
            "flake8>=4.0",
            "mypy>=0.950",
        ],
        "gpu": [
            "triton>=2.0.0",
            "flash-attn>=2.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "s2s-server=src.server:main",
            "s2s-client=src.client:main",
            "s2s-train=src.train:main",
        ],
    },
)