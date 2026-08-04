import numpy as np


def beamform_basic(N, d, lamb, theta, theta0):
    k = 2 * np.pi / lamb
    phase = k * d * (np.sin(theta) - np.sin(theta0))
    n = np.arange(N)
    af = np.exp(1j * n[:, np.newaxis] * phase).sum(axis=0)
    return af


theta = np.linspace(0, np.pi, 91)
af = beamform_basic(N=8, d=0.5, lamb=1.0, theta=theta, theta0=0.0)

print("array factor shape:", af.shape)
print("first 3 values:", af[:3])
