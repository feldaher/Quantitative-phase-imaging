import numpy as np
from scipy.fftpack import fftn, fft2, ifft2
from scipy.ndimage import shift


def psf2otf(psf, shape):
    """
    Convert a PSF (Point Spread Function) to an OTF (Optical Transfer Function).

    Parameters:
    psf: numpy array
        The point spread function.
    shape: tuple
        The desired output shape of the OTF.

    Returns:
    numpy array
        The optical transfer function.
    """
    # Pad the PSF to the desired shape
    psf = np.pad(psf, [(0, shape[i] - psf.shape[i]) for i in range(psf.ndim)], 'constant')

    # Roll the PSF to center it
    for i in range(psf.ndim):
        psf = np.roll(psf, -int(psf.shape[i] / 2), i)

    # Return the Fourier transform of the PSF
    return fftn(psf)


def reconPhaseTieTvWeighted(b1, b2, H1, H2, W1, W2, lambda_, x0=None, T=100, mu=0.1,verbose=True, regval=1, relerr=1e-4, oracle=None):
    """
    Reconstruct the phase using the Total Variation (TV) regularization.

    Parameters:
    b1, b2: numpy arrays
        The input images.
    H1, H2: numpy arrays
        The transfer functions.
    W1, W2: numpy arrays
        The weighting functions.
    lambda_: float
        The regularization parameter.
    x0: numpy array, optional
        The initial guess for the solution.
    T: int, optional
        The maximum number of iterations.
    verbose: bool, optional
        If True, print progress messages.
    fig: bool, optional
        If True, plot figures.
    mu: float, optional
        The parameter for the augmented Lagrangian.
    regval: float, optional
        The regularization value for the kernels.
    relerr: float, optional
        The relative error for the stopping criterion.
    oracle: numpy array, optional
        The ground truth image for SNR computation.

    Returns:
    xopt: numpy array
        The reconstructed image.
    outs: dict
        A dictionary containing the functional values, SNR values, and the exit message.
    """
    # Get the shape of the input images
    nr, nc = b1.shape

    # Regularize the kernels
    H1[0, 0] = regval
    H2[0, 0] = regval

    # Compute the OTFs for the finite difference operators
    otf1 = np.abs(psf2otf(np.array([1, -1]), [nr, nc])) ** 2
    otf2 = np.abs(psf2otf(np.array([[1], [-1]]), [nr, nc])) ** 2

    # Compute the sum of the OTFs
    LTL = otf1 + otf2

    # Compute the conjugates of the transfer functions
    H1T = np.conj(H1)
    H2T = np.conj(H2)

    # Compute HT*y
    HTy = np.real(ifft2((H1T * W1 * fft2(b1)) + (H2T * W2 * fft2(b2))))

    # Initialize the solution
    if x0 == 'L2':
        x0 = np.real(ifft2(fft2(HTy) / (H1T * W1 * H1 + H2T * W2 * H2 + 2 * lambda_ * LTL)))
    elif x0 == 'zero':
        x0 = np.zeros((nr, nc))
    else:
        x0 = b1

    # Initialize the Lagrange multipliers
    alpha1 = np.zeros([nr, nc])
    alpha2 = np.zeros([nr, nc])

    # Initialize the counter for the residual error
    counter = 0

    # Initialize the output dictionary
    outs = {'fval': np.zeros(T, dtype=np.complex128), 'snr': np.zeros(T, dtype=np.float64),
            'exit': 'max number of iterations is reached'}

    # Initialize the solution
    x_init = x0

    # Iterate
    for t in range(T):
        # Compute the forward finite differences
        dhx, dvx = forwFinDiff(x_init)

        # Update the auxiliary variables
        z1 = dhx + alpha1 / mu
        z2 = dvx + alpha2 / mu
        tval = lambda_ / mu
        u1, u2 = updateAuxVar(z1, z2, tval)

        # Update the primary variable
        b = HTy + mu * adjForwFinDiff(u1 - alpha1 / mu, u2 - alpha2 / mu)
        x_new = updatePriVar(b, mu, H1, H2, H1T, H2T, W1, W2, LTL)
 #
        # Update the Lagrange multipliers
        dhx, dvx = forwFinDiff(x_new)
        alpha1 = alpha1 + mu * (dhx - u1)
        alpha2 = alpha2 + mu * (dvx - u2)

        # Compute the functional value
        outs['fval'][t] = computeFunctional(x_new, b1, b2, H1, H2, W1, W2, lambda_)

        # Compute the SNR if the oracle is provided
        if oracle is not None:
            outs['snr'][t] = computeSnr(oracle, x_new, nr, nc)

        # Compute the relative error
        res = np.linalg.norm(x_new - x_init, 'fro') / np.linalg.norm(x_init, 'fro')

        # Check the stopping criterion
        if res <= relerr:
            counter += 1
            if counter > 5:
                outs['exit'] = 'residual error is met'
                break

        # Update the solution
        x_init = x_new

    # Remove the unused entries in the functional and SNR arrays
    outs['fval'] = outs['fval'][outs['fval'] != 0]
    outs['snr'] = outs['snr'][outs['snr'] != 0]

    # Return the final solution and the output dictionary
    return x_new, {'fval': outs['fval'], 'snr': outs['snr'], 'exit': outs['exit']}


def forwFinDiff(f):
    """
    Compute the forward finite differences of an image.

    Parameters:
    f: numpy array
        The input image.

    Returns:
    dhf, dvf: numpy arrays
        The horizontal and vertical finite differences.
    """

    dhf = -f + np.roll(f, shift=-1, axis=1)
    dvf = -f + np.roll(f, shift=-1, axis=0)
    return dhf, dvf

def adjForwFinDiff(dhf, dvf):
    """
    Compute the adjoint of the forward finite differences.

    Parameters:
    dhf, dvf: numpy arrays
        The horizontal and vertical finite differences.

    Returns:
    f: numpy array
        The adjoint finite differences.
    """

    f = -dhf + np.roll(dhf, shift=1, axis=1) + -dvf + np.roll(dvf, shift=1, axis=0)
    return f

def computeFunctional(x, y1, y2, H1, H2, W1, W2, lambda_):
    """
    Compute the functional value.

    Parameters:
    x: numpy array
        The current solution.
    y1, y2: numpy arrays
        The input images.
    H1, H2: numpy arrays
        The transfer functions.
    W1, W2: numpy arrays
        The weighting functions.
    lambda_: float
        The regularization parameter.

    Returns:
    fval: float
        The functional value.
    """
    xs = fft2(x)
    y1s = fft2(y1)
    y2s = fft2(y2)
    H1x = (xs*H1) - y1s
    H2x = (xs*H2) - y2s

    data1 = 0.5 * np.sum(np.real(ifft2(H1x)) * np.real(ifft2(W1 * H1x)))
    data2 = 0.5 * np.sum(np.real(ifft2(H2x)) * np.real(ifft2(W2 * H2x)))


    d1x, d2x = forwFinDiff(xs)
    reg = lambda_ * np.sum(np.sqrt(np.real(d1x) ** 2 + np.real(d2x) ** 2))

    fval = data1 + data2 + reg
    return fval


def computeSnr(oracle, recon, nr, nc):
    """
    Compute the Signal-to-Noise Ratio (SNR).

    Parameters:
    oracle: numpy array
        The ground truth image.
    recon: numpy array
        The reconstructed image.
    nr, nc: int
        The number of rows and columns in the images.

    Returns:
    snr: float
        The SNR.
    """
    P = oracle.flatten()
    x = recon.flatten()
    N = nr * nc

    sumP = np.sum(P)
    sumI = np.sum(x)
    sumIP = np.sum(P * x)
    sumP2 = np.sum(P ** 2)
    sumI2 = np.sum(x ** 2)
    A = np.array([[sumI2, sumI], [sumI, N ** 2]])
    c = np.linalg.solve(A, [sumIP, sumP])
    x = c[0] * x + c[1]
    err = np.sum((P - x) ** 2)
    snr = 10 * np.log10(sumP2 / err)
    return snr


def updateAuxVar(x1, x2, thresh):
    """
    Update the auxiliary variables.

    Parameters:
    x1, x2: numpy arrays
        The current auxiliary variables.
    thresh: float
        The threshold.

    Returns:
    u1, u2: numpy arrays
        The updated auxiliary variables.
    """
    mag = np.sqrt(x1 ** 2 + x2 ** 2)
    mag[mag == 0] = 1
    mag = np.maximum(mag - thresh, 0) / mag
    u1 = x1 * mag
    u2 = x2 * mag
    return u1, u2


def updatePriVar(rhs, mu, H1, H2, H1T, H2T, W1, W2, LTL):
    """
    Update the primary variable.

    Parameters:
    rhs: numpy array
        The right-hand side of the equation.
    mu: float
        The parameter for the augmented Lagrangian.
    H1, H2: numpy arrays
        The transfer functions.
    H1T, H2T: numpy arrays
        The conjugates of the transfer functions.
    W1, W2: numpy arrays
        The weighting functions.
    LTL: numpy array
        The sum of the OTFs.

    Returns:
    xmin: numpy array
        The updated primary variable.
    """
    numerator=fft2(rhs)
    denominator = H1T * W1 * H1 + H2T * W2 * H2 + mu * LTL
    xmin = ifft2(numerator / denominator)
    return np.real(xmin)