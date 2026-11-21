from setuptools import setup, find_packages

setup(
    name="analytics_steer",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "duckdb>=0.9.0",
        "pydantic>=2.0.0",
        "python-dateutil>=2.8.0",
        "typing-extensions>=4.0.0"
    ],
) 