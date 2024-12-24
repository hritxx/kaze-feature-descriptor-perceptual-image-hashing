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

To extract features using different methods, use the "comparison.py" script. The FeatureExtractor
class provides various feature extraction methods such as KAZE, SIFT, SVD, LBP Histogram, and Fuzzy Entropy.

### Hamming Distance Calculation

To calculate Hamming distances between KAZE hashes of frames, use the "kazeHash&Hamming.py" script:

```sh
python kazeHash&Hamming.py
```

This will generate KAZE hashes for each frame and calculate Hamming distances between consecutive frames.

## Analysis

To analyze normalized features, use the "comparison.py" script. The NormalizedFeatureAnalyzer class provides methods to analyze and normalize feature distances.

## Visualization

To visualize the results of normalized feature analysis, use the "comparison.py" script. The
visualize_normalized_results and visualize_normalized_results_curve methods provide comprehensive visualizations of the analysis.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

## License

This project is licensed under the MIT License. See the LICENSE file for details.

```

This README provides an overview of the project, setup instructions, usage examples, and information on contributing and licensing.
This README provides an overview of the project, setup instructions, usage examples, and information on contributing and licensing.
```
