%% High-Level Radar Simulation

clc;
clear;
close all;

%% Radar Parameters
c = 3e8;                 % Speed of light (m/s)
fc = 10e9;               % Carrier frequency (10 GHz)
fs = 20e6;               % Sampling frequency
pulseWidth = 10e-6;      % Pulse width
PRI = 1e-3;              % Pulse Repetition Interval

%% Target Parameters
targetRange = 5000;      % meters
targetRCS = 1;           % Radar Cross Section
SNR = 20;                % dB

%% Derived Parameters
timeDelay = 2 * targetRange / c;
samplesDelay = round(timeDelay * fs);

%% Generate Transmitted Pulse
t = 0:1/fs:pulseWidth;
txPulse = chirp(t,0,pulseWidth,5e6);

%% Received Signal
rxLength = samplesDelay + length(txPulse) + 100;
rxSignal = zeros(1,rxLength);

rxSignal(samplesDelay+1:samplesDelay+length(txPulse)) = targetRCS * txPulse;

%% Add Noise
rxSignal = awgn(rxSignal,SNR,'measured');

%% Matched Filter
matchedFilter = fliplr(conj(txPulse));

output = conv(rxSignal,matchedFilter,'same');

%% Peak Detection
[peakValue,peakIndex] = max(abs(output));

estimatedDelay = peakIndex - length(txPulse)/2;
estimatedRange = estimatedDelay * c /(2*fs);

%% Display Results
fprintf('Actual Range      : %.2f m\n',targetRange);
fprintf('Estimated Range   : %.2f m\n',estimatedRange);

%% Plot

figure;

subplot(3,1,1)
plot(txPulse)
title('Transmitted Pulse')

subplot(3,1,2)
plot(rxSignal)
title('Received Signal')

subplot(3,1,3)
plot(abs(output))
hold on
plot(peakIndex,peakValue,'r*')
title('Matched Filter Output')
xlabel('Samples')
ylabel('Amplitude')