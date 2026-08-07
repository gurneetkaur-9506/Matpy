import numpy as np

c = 3e8
fc = 10e9
fs = 20e6
pulseWidth = 10e - 6
PRI = 1e - 3
targetRange = 5000
targetRCS = 1
SNR = 20
timeDelay = 2 * targetRange / c
# UNRESOLVED: samplesDelay = round(timeDelay * fs)
t = np.arange(0, pulseWidth, 1 / fs)
# UNRESOLVED: txPulse = chirp(t,0,pulseWidth,5e6)
rxLength = samplesDelay + len(txPulse) + 100
rxSignal = np.zeros((1, rxLength))
rxSignal[samplesDelay+1:samplesDelay+len(txPulse)] = targetRCS * txPulse
# UNRESOLVED: rxSignal = awgn(rxSignal,SNR,'measured')
# UNRESOLVED: matchedFilter = fliplr(conj(txPulse))
# UNRESOLVED: output = conv(rxSignal,matchedFilter,'same')
# UNRESOLVED: [peakValue,peakIndex] = max(abs(output))
estimatedDelay = peakIndex - len(txPulse) / 2
estimatedRange = np.linalg.solve((2 * fs).T, estimatedDelay * c.T).T
fprintf['Actual Range:%.2f m\n', targetRange]
fprintf['Estimated Range:%.2f m\n', estimatedRange]
plt.figure()
subplot[2, 0, 0]
plt.plot(txPulse)
plt.title('Transmitted Pulse')
subplot[2, 0, 1]
plt.plot(rxSignal)
plt.title('Received Signal')
subplot[2, 0, 2]
plt.plot(np.abs(output))
# UNRESOLVED: hold on
plt.plot(peakIndex, peakValue, 'r*')
plt.title('Matched Filter Output')
plt.xlabel('Samples')
plt.ylabel('Amplitude')
