import numpy as np

fs = 1000
t = np.arange(0, 1 - 1 / fs, 1 / fs)
f1 = 50
f2 = 120
x = np.sin(2*pi*f1*t) + 0.5 * np.sin(2*pi*f2*t)
Y = np.fft.fft(x)
P2 = np.abs(Y)
P1 = P2[0:len(P2)/2+1]
P1[1:-1] = 2 * P1[1:-1]
f = fs * (np.arange(0, len(P1))) / len(P2)
plt.figure()
plt.plot(f, P1)
plt.title('Single-Sided Amplitude Spectrum')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')
