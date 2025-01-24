# KAZE Feature Descriptor & Perceptual Image Hashing

This repository contains the implementation of KAZE feature descriptor and perceptual image hashing techniques. The codebase includes various feature extraction methods, shot detection, and analysis of normalized features.

## Table of Contents

- [Introduction](#introduction)
- [Setup](#setup)
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
- [Contributing](#contributing)
- [License](#license)

## Introduction

This project demonstrates the use of KAZE feature descriptors and perceptual image hashing for image analysis. The repository includes scripts for extracting frames from videos, detecting shot boundaries, and calculating feature descriptors and Hamming distances.

## Setup

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

## Usage

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

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

## License

This project is licensed under the MIT License. See the LICENSE file for details.
