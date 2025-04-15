import numpy as np

def genFourierWeights(nx, ny, dz, mag, nu, fmax, overlap, theta, display):
    # Cutoff frequency
    fCut = np.sqrt(theta / (np.pi * dz * mag**2 * nu))

    # Convert overlap from percentage to bandwidth
    df = overlap / 100 * fmax / 4

    # Check validity of overlap
    if (fCut - df < 0) or (fCut + df > fmax / 2 - fmax / ny):
        raise ValueError('Overlap is too big')

    # Grid
    x, y = np.meshgrid(np.linspace(-fmax / 2, fmax / 2 - fmax / nx, nx),
                       np.linspace(-fmax / 2, fmax / 2 - fmax / ny, ny))
    r = np.sqrt(x**2 + y**2)

    # Lowpass mask
    w1 = 0.5 * (1 + np.cos(np.pi / (2 * df) * (r - (fCut - df))))
    w1[r < fCut - df] = 1
    w1[r > fCut + df] = 0

    # Highpass mask
    w2 = 1 - w1

    # Display
    # if display:
    #     import matplotlib.pyplot as plt
    #     plt.figure()
    #     plt.subplot(2, 2, 1), plt.imshow(w1, cmap='gray'), plt.axis('equal'), plt.title('Lowpass')
    #     plt.subplot(2, 2, 2), plt.imshow(w2, cmap='gray'), plt.axis('equal'), plt.title('Highpass')
    #     plt.subplot(2, 2, [3, 4]), plt.plot(w1[ny // 2, ny // 2:], linewidth=1.5)

    w1 = np.fft.fftshift(w1)
    w2 = np.fft.fftshift(w2)

    return w1, w2
