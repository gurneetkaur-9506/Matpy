% Single-Sided Amplitude Spectrum of a signal
clear; close all; clc;

fs = 1000;
t = 0:1/fs:1-1/fs;

f1 = 50;
f2 = 120;

x = sin(2*pi*f1*t) + 0.5*sin(2*pi*f2*t);

Y = fft(x);
P2 = abs(Y);
P1 = P2(1:length(P2)/2+1);
P1(2:end-1) = 2*P1(2:end-1);

f = fs*(0:(length(P1)-1))/length(P2);

figure;
plot(f, P1);
title('Single-Sided Amplitude Spectrum');
xlabel('Frequency (Hz)');
ylabel('Magnitude');
