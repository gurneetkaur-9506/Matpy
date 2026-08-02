import numpy as np
import matplotlib.pyplot as plt

# MATLAB: % comment -> Python: # comment (MATLAB comment replaced by Python comment)

# MATLAB: clear; close all; clc; -> Python: re-initialize figure/state (no-op here)

# MATLAB: fs = 1000; -> Python: scalar assignment
fs = 1000

# MATLAB: t = 0:1/fs:1-1/fs; -> Python: np.arange for a MATLAB colon expression
t = np.arange(0, 1, 1 / fs)

# MATLAB: f1 = 50; f2 = 120; -> Python: multiple scalar assignments
f1 = 50
f2 = 120

# MATLAB: x = sin(2*pi*f1*t) + 0.5*sin(2*pi*f2*t); -> Python: numpy element-wise trig
x = np.sin(2 * np.pi * f1 * t) + 0.5 * np.sin(2 * np.pi * f2 * t)

# MATLAB: Y = fft(x); -> Python: numpy.fft.fft
Y = np.fft.fft(x)

# MATLAB: P2 = abs(Y); -> Python: numpy absolute value
P2 = np.abs(Y)

# MATLAB: P1 = P2(1:length(P2)/2+1); -> Python: 1-based slice converted to 0-based
P1 = P2[: len(P2) // 2 + 1]

# MATLAB: P1(2:end-1) = 2*P1(2:end-1); -> Python: slice assignment (skip DC, skip Nyquist)
P1[1:-1] = 2 * P1[1:-1]

# MATLAB: f = fs*(0:(length(P1)-1))/length(P2); -> Python: numpy frequency vector
f = fs * np.arange(0, len(P1)) / len(P2)

# MATLAB: figure; -> Python: create a new matplotlib figure
plt.figure()

# MATLAB: plot(f, P1); -> Python: matplotlib plot
plt.plot(f, P1)

# MATLAB: title('Single-Sided Amplitude Spectrum'); -> Python: matplotlib title
plt.title('Single-Sided Amplitude Spectrum')

# MATLAB: xlabel('Frequency (Hz)'); -> Python: matplotlib xlabel
plt.xlabel('Frequency (Hz)')

# MATLAB: ylabel('Magnitude'); -> Python: matplotlib ylabel
plt.ylabel('Magnitude')

# MATLAB: (implicit display) -> Python: show the figure
plt.show()
