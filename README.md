# Assignments for Image Processing

## Assignment 1
- **Prework**: Install Jupyter Notebook in Miniconda, set up `ece352_env` with pip libraries, create first notebook, and explore cells/saving/loading via online manual.
- **ToDo**:
  - Process "scarlett.jpg" (or any image) with transform \( s = T(z) = \log_2(z) \), plot \( s \) vs. \( z \), normalize with \( s_{scale} = \frac{L}{\log_2(1+L)} \) (\( L = 255 \)), and replot.
  - Apply sigmoid function, plot \( s \) vs. \( z \).
  - Show original and processed images as subplots for both cases.

## Assignment 2
- **ToDo**:
  - Create `image_conv` function for convolution of image \( f \) and filter \( w \).
  - Use 5x5 \( f \) with random integers [0,9] and 3x3 filter \( [[0.125, 0.25, 0], [0.125, 0.5, 0], [0, 0, 0]] \).
  - Create `padding` function to zero-pad \( f \) from NxN to (N+2)x(N+2).
  - Convolve padded \( f \) with flipped \( w \) (both dimensions).
  - Convolve "gradient_noise.png" with 3x3 averaging filter (all values \( \frac{1}{9} \)) using above functions.

## Assignment 3
- **ToDo**:
  - Load color images: "mountain.jpg", "pencils.png", "pencils2.png".
  - Create `euclidean_distance` function for two inputs (p, q), returning float distance.
  - Create `color_histogram` function for color images, using per-channel histograms, concatenating results, and normalizing by sum (inputs: image, bins).
  - Create `min_max_normalization` function to scale array \( f \) to [0,1].
  - Create `luminance` function for 3-channel image, computing \( 0.299R + 0.587G + 0.114B \), normalizing with `min_max_normalization`, and scaling to [0,255].
  - Tasks:
    - Compute `color_histogram` (bins=16) and `euclidean_distance` for pairs (1-2, 1-3, 2-3).
    - Repeat with luminance images, compute histograms and distances (bins=16), compare results in code comments.
    - Visualize three color images and their luminance versions.
    
# Project Overview

## Image Editing and Meme Creation Tool
- **Description**: A Python-based tool for editing images and generating memes. Features interactive filter application (e.g., blur, sharpen, grayscale) and meme overlays on detected facial features (face, eyes, mouth, etc.) using OpenCV’s Haar cascades. Supports saving, loading, and managing images/memes with a simple command-line interface.
- **Purpose**: Designed for quick image manipulation and humorous meme creation with customizable filters and text.
- **Tech**: Built with Python, OpenCV, NumPy, Matplotlib, and PIL.
