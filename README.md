# Quantitative Phase Imaging (QPI)

A Python implementation for phase reconstruction using the Transport-of-Intensity Equation (TIE) approach with Total Variation (TV) regularization.

## Overview

This project implements algorithms to estimate phase shifts on every pixel of brightfield microscopy images using the transport-of-intensity equation. It is particularly useful for biological imaging applications, allowing for quantitative analysis of cellular structures without staining.

Based on the methods described in: [Variations of intracellular density during the cell cycle arise from tip-growth regulation in fission yeast](https://elifesciences.org/articles/64901)

## Features

- Robust phase reconstruction from defocused brightfield images
- Weighted TIE (Transport-of-Intensity Equation) implementation
- Total Variation (TV) regularization for noise suppression
- Batch processing for time-series experiments

## Requirements

- Python 3.6+
- NumPy
- SciPy
- Matplotlib
- OpenCV (cv2)
- scikit-image

## Installation

Clone the repository:

```bash
git clone https://gitlab.com/your-username/quantitative-phase-imaging.git
cd quantitative-phase-imaging
```

Install the required dependencies:

```bash
pip install numpy scipy matplotlib opencv-python scikit-image
```

## Usage

### Basic Usage

The main script processes a z-stack of brightfield images to reconstruct the phase:

```bash
python main.py
```

By default, the script looks for TIF/TIFF files in the specified directory path. Modify the `path` variable in `main.py` to point to your image directory:

```python
path = "/path/to/your/images"
```

### Image Requirements

- Images should be organized as z-stacks (multiple defocused images of the same field of view)
- Default configuration expects 5 images per z-stack (2 above focus, 1 in focus, 2 below focus)
- Images should be in TIFF format

### Customization

Key parameters that can be modified in `main.py`:

```python
# Physical parameters
data = {
    'nu': 512e-9,  # wavelength in meters
    'mag': 60,     # magnification
    'px': 108.3e-9,  # pixel size in meters
    'NA': 1.2      # numerical aperture
}

# Reconstruction parameters
LAMBDA = [2.5e-3]  # regularization parameter
z_step = 1         # corresponding to 500 nm between images in z-stack
data['dzNear'] = 0.5e-6  # distance between images (500 nm)
data['dzFar'] = 1.5e-6   # distance to most out of focus image (1500 nm)
```

## Algorithm Description

The reconstruction process consists of the following steps:

1. Load a z-stack of images (different focal planes)
2. Calculate axial intensity derivatives
3. Generate Fourier domain filters using `genFourierWeights.py`
4. Apply the appropriate optical transfer functions (OTF) using `tieOtf.py`
5. Reconstruct the phase using Total Variation regularization via `reconPhaseTieTvWeighted.py`
6. Save the resulting phase image

## File Structure

- `main.py`: Main script for batch processing
- `genFourierWeights.py`: Generates appropriate frequency domain filters
- `tieOtf.py`: Creates optical transfer functions for TIE
- `reconPhaseTieTvWeighted.py`: Core algorithm for TV regularized phase reconstruction
- `reconPhaseTieTvWeightedBatch.py`: Wrapper for batch processing

## Output

The reconstructed phase images are saved in a 'results_python' directory with filenames following the pattern:

```
Tie_XXX_MIN_MAX.tif
```

Where:
- `XXX` is the timepoint number
- `MIN` is the minimum phase value
- `MAX` is the maximum phase value
