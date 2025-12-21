import numpy as np
from scipy.signal import find_peaks
import matplotlib.pyplot as plt  # Optional, for visualization

# Replace this with your actual signal data
# For example, you might have a signal called 'signal_data'.
signal_data = np.array([1, 2, 3, 2, 1, 2, 3, 4, 3, 2, 1])



# Find peaks
peaks, _ = find_peaks(signal_data)

# Invert the signal to find valleys
inverted_signal = -signal_data
valleys, _ = find_peaks(inverted_signal)

# Optional: Plot the signal and highlight the peaks and valleys
plt.plot(signal_data, label='Signal')
plt.plot(peaks, signal_data[peaks], 'x', label='Peaks', color='r')
plt.plot(valleys, signal_data[valleys], 'o', label='Valleys', color='g')
plt.legend()
plt.show()
