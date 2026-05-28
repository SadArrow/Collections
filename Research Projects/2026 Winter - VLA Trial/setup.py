from setuptools import setup, find_packages

setup(
    name="vla_model",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.24.0",
        "timm>=0.9.0",
        "torchvision>=0.15.0",
        "Pillow>=9.0.0",
        "transformers>=4.36.0",
        "pydantic>=2.0.0",
        "typing-extensions>=4.0.0",
    ],
    extras_require={
        "dev": ["pytest", "black", "ruff"],
    },
)
