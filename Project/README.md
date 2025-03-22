# Image Processing and Meme Generator

## Overview
- **Description**: Python script for loading, editing, and saving images with filters and meme overlays using OpenCV, NumPy, and Matplotlib. Supports face/eye/mouth detection with Haar cascades and various image transformations.

## Features
- **Menus**: 
  - Filter menu: Smooth, Bright, Sharpen, Grayscale, Bilateral, Color mix, Green, Binary, Add text, Load meme, Save, Erase filters, Exit.
  - Meme menu: Face, Mouth, Eyes, Head, Torso, Top left/right background operations.
- **Image Handling**: Load (jpg, jpeg, tiff, png), plot, save processed images, remove from folders.
- **Filters**: Apply Gaussian blur, brightness/contrast, sharpening, grayscale, bilateral, color mixing, green tint, binary thresholding.
- **Meme Overlay**: Detects facial features (face, mouth, eyes, torso, head) via Haar cascades, overlays memes with scaling and transparency.
- **Text**: Adds customizable text (color, position) from `.txt` files.
- **Utilities**: RGB/BGR detection, format checking, transparency conversion.

## Usage
- **Run**: Execute script, choose options via input:
  - `l`: Load image from `images/`.
  - `p`: List available images.
  - `r`/`rm`: Remove image from `images/` or `processed_memes/`.
  - `m`: Load/display processed meme.
  - `e`: Exit.
- **Processing**: After loading, apply filters or memes interactively, save to `processed_memes/`.

## Key Functions
- **`plot_img`**: Displays image with RGB conversion.
- **`face_detect`/`mouth_detect`/`eyes_detect`/`torso_detect`/`head_detect`**: Detect features, overlay memes.
- **`background_top_left`/`background_top_right`**: Place memes in corners.
- **`transparent`**: Converts to 4-channel (RGBA).
- **`check_format`**: Validates image formats.
- **`add_text`**: Overlays text with customizable color/size.

## Dependencies
- NumPy, OpenCV (`cv2`), Matplotlib, PIL, SciPy, IPython, Tkinter.

## Notes
- Images stored in `images/`, memes in `meme_assets/`, processed outputs in `processed_memes/`, text in `texts/`.
- Haar cascade files in `Haar/` for detection.
- Author: Georgios Kapakos (ID: 03165).
