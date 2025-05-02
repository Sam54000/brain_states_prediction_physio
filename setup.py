from setuptools import setup, find_packages

setup(
    name="brain_states_prediction_physio",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "pandas",
        "numpy>=2.0.0",
        "matplotlib",
        "scipy",
        "bids_explorer",
        "neurokit2",
        "PyWavelets",
        "opencv-python",
        "typing-extensions",
        "dataclasses",
    ],
    python_requires=">=3.8",  # Based on typing usage and modern Python features
) 