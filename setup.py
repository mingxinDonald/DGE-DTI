from setuptools import setup, find_packages

setup(
    name="dge-dti",
    version="0.1.0",
    description="Drug-Target Interaction prediction using Drug Graph Encoder",
    author="DGE-DTI Contributors",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "scikit-learn>=1.0.0",
        "torch>=1.12.0",
        "rdkit>=2022.3.1",
        "PyYAML>=6.0",
        "tqdm>=4.62.0",
        "scipy>=1.7.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "matplotlib>=3.5.0",
            "seaborn>=0.11.0",
        ]
    },
)
