#!/usr/bin/env python
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
from src.utils import test_artifact_detectors
import os
from pathlib import Path

if __name__ == "__main__":
    # Create output directory if it doesn't exist
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    
    # Run the artifact detector test
    save_path = output_dir / "artifact_detection_test.png"
    rsp_results, ecg_results = test_artifact_detectors(save_path=save_path)
    
    comparison_path = str(save_path).replace('.png', '_comparison.png')
    
    print("Artifact detection test completed!")
    print(f"Test visualization saved to: {save_path}")
    print(f"Comparison visualization saved to: {comparison_path}")
    
    # Report detection accuracy
    print("\nRSP Artifact Detection:")
    print(f"- High frequency artifact placed at: {rsp_results['known_artifacts']['high_freq']} seconds")
    print(f"- Flat segment artifact placed at: {rsp_results['known_artifacts']['flat']} seconds")
    
    print("\nECG Artifact Detection:")
    print(f"- High frequency artifact placed at: {ecg_results['known_artifacts']['high_freq']} seconds")
    print(f"- Flat segment artifact placed at: {ecg_results['known_artifacts']['flat']} seconds")
    
    print("\nTwo plots have been generated:")
    print("1. artifact_detection_test.png: Shows the signals and the detection results")
    print("2. artifact_detection_test_comparison.png: Shows clean vs artifacted signals")
    
    # Display the plot
    # Note: In non-interactive environments, plt.show() may not work as expected
    try:
        plt.show()
    except Exception as e:
        print(f"Could not show plots interactively: {e}") 