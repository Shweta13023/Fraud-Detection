from setuptools import setup, find_packages

setup(
    name="fraud_detection_system",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        'numpy>=1.21.0',
        'pandas>=1.3.0',
        'scikit-learn>=0.24.2',
        'matplotlib>=3.4.0',
        'imbalanced-learn>=0.8.0',
    ],
)
