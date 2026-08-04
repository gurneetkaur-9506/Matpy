import numpy as np

fs = 1000.0
f1 = 50.0
f2 = 120.0

t = np.arange(0.0, 1.0, 1.0 / fs)
x = np.sin(2 * np.pi * f1 * t) + 0.5 * np.sin(2 * np.pi * f2 * t)

X = np.fft.fft(x)
magnitude = np.abs(X)

half = len(magnitude) // 2 + 1
P1 = magnitude[:half]
P1[1:-1] *= 2.0

freqs = np.arange(0, len(P1)) * fs / len(magnitude)

print("samples:", len(x))
print("first 8 spectrum values:", P1[:8])
print("corresponding frequencies:", freqs[:8])
