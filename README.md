# KAZE Feature Descriptor & Perceptual Image Hashing

[![DOI](https://img.shields.io/badge/DOI-10.XXXX%2FXXXXXX-blue)](https://doi.org/10.XXXX/XXXXXX)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains the implementation of KAZE feature descriptor and perceptual image hashing techniques. The codebase includes various feature extraction methods, shot detection, and analysis of normalized features.

> **📝 Citation Notice:** This code repository is directly related to the manuscript submitted to _The Visual Computer_. If you use this code or data in your research, please cite:
>
> _[Authors]. "KAZE Feature Descriptor & Perceptual Image Hashing for Video Transition Detection." The Visual Computer (under review). DOI: [to be assigned]_

**Repository Links:**

- **GitHub:** https://github.com/hritxx/kaze-feature-descriptor-perceptual-image-hashing
- **Code Archive (Zenodo):** [![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.XXXXXX-blue)](https://doi.org/10.5281/zenodo.XXXXXX) _(to be assigned)_
- **Dataset (Zenodo):** [![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.XXXXXX-blue)](https://doi.org/10.5281/zenodo.XXXXXX) _(to be assigned)_

## Table of Contents

- [Introduction](#introduction)
- [Setup](#setup)
  - [Dependencies and Requirements](#dependencies-and-requirements)
  - [Installation](#installation)
- [Key Algorithms and Implementation](#key-algorithms-and-implementation)
  - [KAZE Feature Descriptor](#kaze-feature-descriptor)
  - [Perceptual Hashing](#perceptual-hashing)
  - [Transition Detection Methods](#transition-detection-methods)
- [Usage](#usage)
  - [Extract Frames](#extract-frames)
  - [Detect Shots](#detect-shots)
  - [Feature Extraction](#feature-extraction)
  - [Hamming Distance Calculation](#hamming-distance-calculation)
- [Briefing of the Overall Task](#briefing-of-the-overall-task)
  - [Preprocessing](#preprocessing)
  - [Transition Detection](#transition-detection)
- [Performance Comparison](#performance-comparison)
- [Analysis](#analysis)
- [Visualization](#visualization)
- [Reproducibility](#reproducibility)
- [Contributing](#contributing)
- [License](#license)

## Introduction

This project demonstrates the use of KAZE feature descriptors and perceptual image hashing for image analysis. The repository includes scripts for extracting frames from videos, detecting shot boundaries, and calculating feature descriptors and Hamming distances.

## Setup

### Dependencies and Requirements

This project requires the following dependencies:

**Core Libraries:**

- Python 3.7+
- OpenCV (cv2) >= 4.5.0 - For computer vision operations and KAZE feature extraction
- NumPy >= 1.19.0 - For numerical computations
- scikit-image >= 0.18.0 - For SSIM and image processing
- Matplotlib >= 3.3.0 - For visualization
- SciPy >= 1.5.0 - For signal processing (Savitzky-Golay filter)

**Optional Libraries:**

- Pandas - For data analysis
- Seaborn - For enhanced visualizations

All dependencies are listed in `requirements.txt` for easy installation.

### Installation

1. Clone the repository:

   ```sh
   git clone https://github.com/hritxx/kaze-feature-descriptor-perceptual-image-hashing.git
   cd kaze-feature-descriptor-perceptual-image-hashing
   ```

2. Install the required dependencies:

   ```sh
   pip install -r requirements.txt
   ```

3. Download the required videos:
   You can download the videos from the following Google Drive links:

   - [BG_37770.mpg](https://drive.google.com/file/d/13hRRWs3uGK5nxqJb3DMRwq3eOWYA_HwM/view)
   - [D5.mpg](https://drive.google.com/file/d/1G_l-wxMPEN4x-UywXzv4zsdtvlhzNgh4/view)

   After downloading, place the videos in the root directory of the project.

   ## Microscopic / Biological Videos

   As an extension of this research into medical and biological image analysis, we include links to two related Google Drive folders containing microscopic observation videos and their extracted frames. These datasets are useful for testing the robustness of the KAZE feature descriptor and perceptual hashing approach on microscopy and medically-relevant video data (different scales, noise characteristics, and motion patterns).

   - Microscopy / biological videos (original video files):

     https://drive.google.com/drive/folders/15lFETsBZx6CWNDH0v0zBKNBvg5aCp_AU?usp=sharing

   - Extracted frames from the microscopy videos (already extracted frames):

     https://drive.google.com/drive/folders/128lx1sNXqr2LgW37znA2AKW5jLcL70ha?usp=sharing

   Usage notes:

   - Recommended folder names locally:

     - `microscopy_videos/` for the raw video files
     - `microscopy_frames/` for the extracted frames

   - Suggested workflow:

     1. Download the videos into `microscopy_videos/` or place the provided frames into `microscopy_frames/`.
     2. Use `Frames&Shots-Extractor.py` to (re-)extract frames if you only have videos and want control over frame rate or resizing.
     3. Run `kazeHash&Hamming.py` and `comparison.py` against `microscopy_frames/` to evaluate feature descriptors and hashing on the dataset.

   - Notes specific to microscopy data:
     - Frames may have high frame-to-frame similarity; consider subsampling (e.g., every Nth frame) before hashing to reduce redundancy.
     - Lighting and focus changes are common; tune thresholds for SSIM/hamming accordingly or apply simple preprocessing (CLAHE, denoising) before feature extraction.

   If you'd like, I can add a small helper script to download these Drive folders (or to read from them if you provide a mounted path) and place contents into the recommended local folders.

## Key Algorithms and Implementation

### KAZE Feature Descriptor

KAZE is a 2D feature detection and description algorithm that operates in a nonlinear scale space. Unlike traditional methods (SIFT, SURF) that use Gaussian blurring, KAZE uses nonlinear diffusion filtering to preserve object boundaries while reducing noise.

**Implementation Details:**

- **Keypoint Detection:** Uses nonlinear scale space with Additive Operator Splitting (AOS) schemes
- **Descriptor Computation:** Modified Local Difference Binary (M-LDB) descriptor
- **Parameters:**
  - `threshold`: Detection threshold (default: 0.001)
  - `nOctaves`: Number of octave levels (default: 4)
  - `nOctaveLayers`: Number of sublevels per octave (default: 4)
  - `extended`: Use 128-dimensional descriptor vs 64-dimensional (default: False)

**Key Features:**

- Superior boundary localization compared to Gaussian-based methods
- Rotation and scale invariant
- Robust to noise while preserving edges

### Perceptual Hashing

Perceptual hashing generates compact binary representations of images that are robust to minor variations but sensitive to significant content changes.

**Implementation Steps:**

1. **Image Preprocessing:**
   - Resize to standard dimensions (e.g., 256×256) using bilinear interpolation
   - Convert to grayscale
2. **Feature Extraction:**
   - Extract KAZE keypoints and descriptors
   - Compute spatial distribution of features
3. **Hash Generation:**

   - Encode descriptor information into binary hash
   - Normalize hash length for consistent comparison

4. **Similarity Measurement:**
   - Compute Hamming distance between hash values
   - Lower distance = higher similarity

### Transition Detection Methods

This repository implements multiple transition detection approaches:

#### 1. Basic Shot Boundary Detection (`kazeHash&Hamming.py`)

**Method:** Structural Similarity Index (SSIM)

- Compares consecutive frames using SSIM metric
- Threshold-based classification (default: 0.3)
- Identifies abrupt vs gradual transitions

**Algorithm:**

```
for each consecutive frame pair (i, i+1):
    1. Convert to grayscale
    2. Resize to 256×256
    3. Compute SSIM score
    4. difference = 1 - SSIM
    5. if difference > threshold:
        classify as abrupt (difference > 0.5) or gradual
```

#### 2. Advanced Multi-Metric Detection (`kazeHash&Hamming.py`)

**Method:** Combines SSIM, Histogram, and Edge differences

**Metrics:**

- **SSIM:** Structural similarity
- **Histogram Difference:** Chi-square distance between color histograms
- **Edge Difference:** Canny edge detection comparison

**Algorithm:**

```
for each frame pair:
    1. Compute SSIM score
    2. Calculate histogram difference (chi-square)
    3. Detect edges with Canny
    4. Compute edge difference
    5. combined_diff = mean([ssim, hist_diff, edge_diff])
    6. Apply threshold for transition detection
```

#### 3. Hybrid KAZE-Based Detection (`hyd2.py`)

**Method:** Multi-modal approach with adaptive weighting

**Components:**

1. **KAZE Feature Matching:**

   - Detect and match KAZE features between frames
   - Apply Lowe's ratio test for robust matching
   - Compute similarity based on good matches

2. **Histogram Analysis:**

   - Correlation and chi-square histogram comparison
   - Normalized to 0-1 range

3. **Pixel Difference:**
   - Mean squared error (MSE) between frames
   - Edge-based structural difference

**Adaptive Weighting:**

- If few KAZE matches (< 10): prioritize pixel/histogram (20/40/40)
- Otherwise: prioritize KAZE features (50/25/25)

**Algorithm:**

```
for each frame pair:
    1. Extract KAZE features and match
    2. Compute histogram correlation and chi-square
    3. Calculate pixel MSE and edge difference
    4. Apply adaptive weights based on match count
    5. combined_score = weighted_sum(kaze, hist, pixel)
    6. Boost score if multiple methods agree
    7. Smooth signal with Savitzky-Golay filter
    8. Detect transitions using adaptive thresholds
```

**Transition Classification:**

- **Abrupt:** Single frame with high difference (> 0.6)
- **Gradual:** Sustained elevated difference over multiple frames (5-30 frames)

### Extract Frames

To extract frames from a video, use the "Frames&Shots-Extractor.py" script:

```sh
python Frames&Shots-Extractor.py
```

This will extract frames from the specified video and save them in the output directory.

### Detect Shots

To detect shot boundaries and save representative frames, use the "Frames&Shots-Extractor.py" script:

```sh
python Frames&Shots-Extractor.py
```

This will detect shot boundaries based on histogram differences and save representative frames in the output directory.

### Feature Extraction

To extract features using different methods, use the "comparison.py" script. The `FeatureExtractor` class provides various feature extraction methods such as KAZE, SIFT, SVD, LBP Histogram, and Fuzzy Entropy.

### Hamming Distance Calculation

To calculate Hamming distances between KAZE hashes of frames and detect shot boundaries, use the updated "kazeHash&Hamming.py" script:

```sh
python kazeHash&Hamming.py
```

This script:

1. **Basic Shot Boundary Detection**:

   - Detects shot boundaries by calculating the **Structural Similarity Index (SSIM)** between consecutive frames.
   - Determines the type of transition (**abrupt** or **gradual**) based on a similarity threshold (default: `0.3`).

2. **Advanced Shot Boundary Detection**:

   - Combines multiple similarity measures:
     - **SSIM**: Measures structural similarity.
     - **Histogram Difference**: Computes the chi-square distance between color histograms of frames.
     - **Edge Difference**: Uses Canny edge detection to compute edge differences.
   - Averages these metrics to detect transitions more robustly.

3. **Output**:
   - Prints details of detected shot boundaries, including:
     - The type of transition (abrupt/gradual).
     - The difference score.
     - The filenames of consecutive frames involved.

## Briefing of the Overall Task

KAZE feature point descriptor is used to detect key points from the extracted video frames and similarity measure is computed. Hash value is generated from these extracted frames. Hash values of successive frames are compared to detect abrupt transitions. Normalized Hamming distance is calculated and compared for two successive hashes to detect abrupt transition.

### Preprocessing

Initially, input color frames are extracted from the video. Frames are resized and converted into standard normalized images by `x × x` bilinear interpolation. The purpose of resizing the input frames is to generate hash values of the same length for frames of different dimensions. Then RGB images are transformed into grayscale images to enable key points for the feature detection process. Hashes generated from individual gray images are classified to detect abrupt transitions based on the Hamming distance between two successive images.

### Transition Detection

Hamming distances are compared with the value of the threshold to determine if two consecutive frames are perceptually similar or dissimilar. Two consecutive frames which are perceptually dissimilar denote abrupt transition.

## Performance Comparison

Few state-of-the-art techniques like **SIFT (Scale-Invariant Feature Transform)**, **SVD (Singular Value Decomposition)**, **LBP Histogram**, and **Fuzzy-based algorithms** are compared with the proposed system. For an unbiased comparison between the performance of the algorithms, a subset of detected abrupt transition frames from the TRECVid 2001 dataset and TRECVid 2007 dataset are used.

## Analysis

To analyze normalized features, use the "comparison.py" script. The `NormalizedFeatureAnalyzer` class provides methods to analyze and normalize feature distances.

## Visualization

To visualize the results of normalized feature analysis, use the "comparison.py" script. The
`visualize_normalized_results` and `visualize_normalized_results_curve` methods provide comprehensive visualizations of the analysis.

## Reproducibility

This repository is designed to facilitate full reproducibility of the experimental results presented in our manuscript submitted to _The Visual Computer_.

### Permanent Links

All code and data are hosted with permanent identifiers to ensure long-term accessibility:

- **GitHub Repository:** https://github.com/hritxx/kaze-feature-descriptor-perceptual-image-hashing
- **Code Archive (Zenodo):** DOI will be assigned upon acceptance
- **Dataset (Zenodo):** DOI will be assigned upon acceptance

### Replicating Experiments

To replicate the experiments from the paper:

1. **Setup Environment:**

   ```sh
   git clone https://github.com/hritxx/kaze-feature-descriptor-perceptual-image-hashing.git
   cd kaze-feature-descriptor-perceptual-image-hashing
   pip install -r requirements.txt
   ```

2. **Download Test Videos:**

   - Download the TRECVid dataset videos from the links provided in the Setup section
   - Download microscopy videos (optional) for biological dataset experiments

3. **Extract Frames:**

   ```sh
   python Frames&Shots-Extractor.py
   ```

4. **Run Transition Detection:**

   ```sh
   # Basic SSIM-based detection
   python kazeHash&Hamming.py

   # Hybrid KAZE-based detection
   python hyd2.py
   ```

5. **Compare Methods:**
   ```sh
   python comparison.py
   ```

### Expected Results

The code will produce:

- Frame extraction outputs in designated directories
- Shot boundary detection results with transition types (abrupt/gradual)
- Visualization plots showing difference signals and detected transitions
- Performance comparison metrics between KAZE, SIFT, SVD, LBP, and Fuzzy-based methods

### Parameter Tuning

Key parameters that can be adjusted for different datasets:

- **KAZE threshold:** Lower values (0.0001-0.001) detect more features
- **Transition thresholds:** Adjust based on video characteristics
  - `abrupt_threshold_percentile`: 85-95 (default: 90)
  - `abrupt_min_diff`: 0.5-0.8 (default: 0.6)
- **SSIM threshold:** 0.2-0.4 for different sensitivity levels

### Citation

If you use this code or data in your research, please cite:

```bibtex
@article{kaze-perceptual-hashing,
  title={KAZE Feature Descriptor \& Perceptual Image Hashing for Video Transition Detection},
  author={[Authors]},
  journal={The Visual Computer},
  year={2025},
  note={under review},
  doi={[to be assigned]}
}
```

For questions about reproducibility or issues with the code, please open an issue on GitHub or contact the corresponding author.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

## License

This project is licensed under the MIT License. See the LICENSE file for details.
