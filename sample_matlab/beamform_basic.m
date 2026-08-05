function af = beamform_basic(N, d, lambda, theta, theta0)
    k = 2 * pi / lambda;
    phase = k * d * (sin(theta) - sin(theta0));
    af = zeros(size(theta));
    for n = 1:N
        af = af + exp(1i * (n - 1) * phase);
    end
end
