from setuptools import setup, find_packages

setup(
    name="fiable",
    version="0.1.0",
    description="CLI for downloading, quantizing, evaluating, and visualizing LLM compression",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Fiable Team",
    author_email="",
    url="https://github.com/Fiable/fiable",
    packages=find_packages(include=["fiable", "fiable.*"]),
    python_requires=">=3.8",
    install_requires=[
        "typer>=0.12.0",
        "rich>=13.7.0",
        "huggingface_hub>=0.23.0",
        "matplotlib>=3.8.0",
        "seaborn>=0.13.0",
        "pandas>=2.0.0",
        "lm-eval>=0.4.0",
        "datasets>=2.14.0",
        "evaluate>=0.4.0",
        "transformers>=4.40.0",
        "accelerate>=0.26.0",
        "llama-cpp-python>=0.2.0",
        "nvidia-ml-py>=12.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=23.0.0",
            "isort>=5.12.0",
            "mypy>=1.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "fiable=fiable.cli.commands:run",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)
