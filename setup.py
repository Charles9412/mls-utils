from setuptools import setup, find_packages

setup(
    name="mls_utils",
    version="0.1.0",
    description="Helper functions for MLS scoring and segmentation",
    author="Carlos Eugenio Miranda Rocha",
    packages=find_packages(),
    install_requires=[
        "pandas",
        "numpy",
        "scikit-learn",
        "matplotlib",
        "seaborn",
        "gradio"
    ],
)
