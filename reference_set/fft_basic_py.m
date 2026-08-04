% Python: import numpy as np -> MATLAB: no import; numpy names become built-ins
% (fft, abs, sin, pi are native MATLAB functions/constants)

% Python: fs = 1000.0 -> MATLAB: scalar assignment
fs = 1000.0;

% Python: f1 = 50.0; f2 = 120.0 -> MATLAB: multiple scalar assignments
f1 = 50.0;
f2 = 120.0;

% Python: t = np.arange(0.0, 1.0, 1.0/fs) -> MATLAB: colon operator
% numpy arange(start, stop, step) -> start:step:stop-step
t = 0.0:1.0/fs:1.0-1.0/fs;

% Python: x = np.sin(2*np.pi*f1*t) + 0.5*np.sin(2*np.pi*f2*t)
%      -> MATLAB: sin/pi are built-ins, element-wise on vector t
x = sin(2*pi*f1*t) + 0.5*sin(2*pi*f2*t);

% Python: X = np.fft.fft(x) -> MATLAB: fft(x)
X = fft(x);

% Python: magnitude = np.abs(X) -> MATLAB: abs(X)
magnitude = abs(X);

% Python: half = len(magnitude) // 2 + 1 -> MATLAB: floor division
% Python '//' is floor division -> floor(length(...)/2)
half = floor(length(magnitude)/2) + 1;

% Python: P1 = magnitude[:half] -> MATLAB: 0-based slice -> 1-based indices
P1 = magnitude(1:half);

% Python: P1[1:-1] *= 2.0 -> MATLAB: slice assignment skipping first and last
% 0-based [1:-1] -> 1-based 2:end-1; '*=' in-place scaling
P1(2:end-1) = 2.0 * P1(2:end-1);

% Python: freqs = np.arange(0, len(P1)) * fs / len(magnitude)
%      -> MATLAB: (0:(length(P1)-1)) vector scaled by fs/length(magnitude)
freqs = (0:(length(P1)-1)) * fs / length(magnitude);

% Python: print("samples:", len(x)) -> MATLAB: disp with formatted text
disp(['samples: ', num2str(length(x))]);

% Python: print("first 8 spectrum values:", P1[:8]) -> MATLAB: disp a slice
% 0-based P1[:8] -> 1-based P1(1:8)
disp('first 8 spectrum values:');
disp(P1(1:8));

% Python: print("corresponding frequencies:", freqs[:8]) -> MATLAB: disp a slice
disp('corresponding frequencies:');
disp(freqs(1:8));
