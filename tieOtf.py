import numpy as np

def tie_otf(nx, ny, fmax, nu, dz):
    """
    Computes the OTF used to solve TIE (Laplacian)
    Input:
    - nx: horizontal size (even number)
    - ny: vertical size (even number)
    - fmax: maximal frequency of the grid
    - nu: wavelength (meters)
    - dz: distance from the object (defocus distance) (meters)
    """
    # Spatial frequency grid
    fx, fy = np.meshgrid(np.linspace(-fmax/2, fmax/2 - fmax/nx, nx),
                         np.linspace(-fmax/2, fmax/2 - fmax/ny, ny))
    f2 = fx**2 + fy**2

    # Operator
    OTF = 4 * np.pi * nu * dz * np.fft.fftshift(f2)

    return OTF
