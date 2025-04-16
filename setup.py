from setuptools import setup, find_packages

setup(
    name="brain_states_prediction_physio",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "pandas",
        "numpy",
        "matplotlib",
        "scipy",
        "bids_explorer",
        "neurokit2",
    ],
) 