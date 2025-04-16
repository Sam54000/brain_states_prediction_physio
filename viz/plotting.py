import matplotlib.pyplot as plt
import numpy as np

def plot_ppg_with_artifacts(ppg_signal, mask, fs):
    time_axis = np.arange(len(ppg_signal)) / fs

    fig, ax = plt.subplots(figsize=(15, 5))

    ax.plot(
        time_axis, 
        ppg_signal,
        color='blue', 
        label='PPG Signal'
        )
    ax.scatter(
        time_axis[~mask.astype(bool)], 
        np.ones_like(ppg_signal[~mask.astype(bool)])*0.04,
        color='red', 
        marker="o",
        label='Bad Segment'
        )
    ax.set_title('PPG Signal with Detected Motion Artifacts')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('PPG Amplitude')
    ax.set_xlim(300,400)


    ax.legend()
    plt.tight_layout()
    plt.show()