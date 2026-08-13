% Shape inference: scalar, vector and matrix variables.
function out = shape_inference(fs, numPoints)
    f1 = 50;
    f2 = 120;
    t = 0:1/fs:1-1/fs;

    x = sin(2*pi*f1*t) + 0.5*sin(2*pi*f2*t);

    Y = fft(x);
    P2 = abs(Y);
    P1 = P2(1:length(P2)/2+1);

    A = [1 2 3; 4 5 6];
    v = [1 2 3];
    s = sum(x);
    gain = 0.5;
    y = gain * x;

    scale = length(P2);
    f = fs*(0:(length(P1)-1))/scale;

    acc = zeros(1, numPoints);
    for n = 1:numPoints
        acc(n) = n * 2;
    end

    out = sum(y) + s + sum(acc);
end
