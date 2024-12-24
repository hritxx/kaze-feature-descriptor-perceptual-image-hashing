import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import local_binary_pattern
from scipy.spatial.distance import pdist, squareform
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from scipy.stats import skew, kurtosis


class FeatureExtractor:
    def kaze_extractor(frame, hash_size=256,num_scales=3):
        """
            Supercharged KAZE feature extraction with multi-statistical aggregation,
            multi-scale analysis, spatial keypoint metrics, advanced dimensionality reduction,
            and robust normalization.
            """
        # Convert to grayscale if needed
        if len(frame.shape) == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        kaze = cv2.KAZE_create()

        # Multi-scale feature aggregation
        aggregated_features = []
        for scale in range(num_scales):
            # Downsample the image for multi-scale analysis
            scaled_frame = cv2.resize(frame, (frame.shape[1] // (scale + 1), frame.shape[0] // (scale + 1)))
            keypoints, descriptors = kaze.detectAndCompute(scaled_frame, None)

            if descriptors is not None and len(descriptors) > 0:
                # Step 1: Descriptor Statistics
                mean_desc = np.mean(descriptors, axis=0)
                std_desc = np.std(descriptors, axis=0)
                min_desc = np.min(descriptors, axis=0)
                max_desc = np.max(descriptors, axis=0)
                skew_desc = skew(descriptors, axis=0)
                kurt_desc = kurtosis(descriptors, axis=0)

                # Combine statistics
                desc_features = np.concatenate([mean_desc, std_desc, min_desc, max_desc, skew_desc, kurt_desc])
                aggregated_features.append(desc_features)

        if not aggregated_features:
            # Return zero vector if no descriptors are found
            return np.zeros(hash_size)

        # Step 2: Aggregate Across Scales
        combined_features = np.mean(aggregated_features, axis=0)

        # Step 3: Keypoint Spatial Features
        if keypoints:
            keypoint_coords = np.array([[kp.pt[0], kp.pt[1], kp.size, kp.angle] for kp in keypoints])
            spatial_mean = np.mean(keypoint_coords, axis=0)
            spatial_std = np.std(keypoint_coords, axis=0)
            spatial_features = np.concatenate([spatial_mean, spatial_std])
            combined_features = np.concatenate([combined_features, spatial_features])

        # Step 4: Dimensionality Reduction (PCA or Autoencoder)
        if len(combined_features) > hash_size:
            pca = PCA(n_components=hash_size)
            try:
                reduced_features = pca.fit_transform(combined_features.reshape(1, -1)).flatten()
            except ValueError:
                # PCA fallback if insufficient samples
                reduced_features = combined_features[:hash_size]
        else:
            # Truncate or pad to hash_size
            reduced_features = (np.pad(combined_features, (0, hash_size - len(combined_features)), mode='constant')
                                if len(combined_features) < hash_size else combined_features[:hash_size])

        # Step 5: Normalize Features
        scaler = MinMaxScaler()
        normalized_features = scaler.fit_transform(reduced_features.reshape(-1, 1)).flatten()

        return normalized_features

    @staticmethod
    def sift_extractor(frame, hash_size=128):
        """
        SIFT feature extraction with normalization
        """
        # Convert to grayscale if needed
        if len(frame.shape) == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        sift = cv2.SIFT_create()
        keypoints, descriptors = sift.detectAndCompute(frame, None)

        if descriptors is None or len(descriptors) == 0:
            return np.zeros(hash_size)

        # Aggregate and truncate/pad
        mean_descriptor = np.mean(descriptors, axis=0)

        # Truncate or pad to hash_size
        if len(mean_descriptor) > hash_size:
            normalized_desc = mean_descriptor[:hash_size]
        else:
            normalized_desc = np.pad(mean_descriptor, (0, hash_size - len(mean_descriptor)), mode='constant')

        scaler = StandardScaler()
        normalized_desc = scaler.fit_transform(normalized_desc.reshape(-1, 1)).flatten()

        return normalized_desc

    @staticmethod
    def svd_extractor(frame, hash_size=10):
        """
        SVD-based feature extraction with normalization
        """
        # Convert to grayscale if needed
        if len(frame.shape) == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Resize to a consistent size to ensure SVD works
        frame_resized = cv2.resize(frame, (64, 64))

        # Perform SVD
        u, s, vh = np.linalg.svd(frame_resized.astype(float))

        # Extract top singular values and first singular vector components
        top_values = s[:5]
        top_vector = u[:, 0][:5]

        # Combine and normalize
        features = np.concatenate([top_values, top_vector])

        # Truncate or pad to hash_size
        if len(features) > hash_size:
            normalized_features = features[:hash_size]
        else:
            normalized_features = np.pad(features, (0, hash_size - len(features)), mode='constant')

        scaler = MinMaxScaler()
        normalized_features = scaler.fit_transform(normalized_features.reshape(-1, 1)).flatten()

        return normalized_features

    @staticmethod
    def lbp_histogram_extractor(frame, hash_size=32):
        """
        Local Binary Pattern histogram with normalization
        """
        # Convert to grayscale if needed
        if len(frame.shape) == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        radius = 1
        n_points = 8 * radius
        lbp = local_binary_pattern(frame, n_points, radius, method='uniform')
        hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, n_points + 3), range=(0, n_points + 2))
        normalized_hist = hist.astype(float) / hist.sum()

        # Truncate or pad to hash_size
        if len(normalized_hist) > hash_size:
            normalized_hist = normalized_hist[:hash_size]
        else:
            normalized_hist = np.pad(normalized_hist, (0, hash_size - len(normalized_hist)), mode='constant')

        # Additional normalization
        scaler = StandardScaler()
        normalized_hist = scaler.fit_transform(normalized_hist.reshape(-1, 1)).flatten()

        return normalized_hist

    @staticmethod
    def fuzzy_extractor(frame, hash_size=10):
        """
        Fuzzy entropy-based feature extraction with normalization
        """
        # Convert to grayscale if needed
        if len(frame.shape) == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Normalize frame
        frame_norm = frame.astype(float) / 255.0

        # Calculate various fuzzy entropy metrics
        fuzzy_entropy = -np.sum(frame_norm * np.log2(frame_norm + 1e-10))
        variance = np.var(frame_norm)
        mean = np.mean(frame_norm)
        median = np.median(frame_norm)
        std = np.std(frame_norm)

        # Combine and normalize
        features = np.array([fuzzy_entropy, variance, mean, median, std])

        # Truncate or pad to hash_size
        if len(features) > hash_size:
            normalized_features = features[:hash_size]
        else:
            normalized_features = np.pad(features, (0, hash_size - len(features)), mode='constant')

        scaler = MinMaxScaler()
        normalized_features = scaler.fit_transform(normalized_features.reshape(-1, 1)).flatten()

        return normalized_features

    @staticmethod
    def hamming_extractor(frame, hash_size=64):
        """
        Perceptual hash with Hamming distance preparation
        """
        # Convert to grayscale if needed
        if len(frame.shape) == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Resize to consistent size
        resized = cv2.resize(frame, (8, 8))

        # Compute mean pixel value
        mean = np.mean(resized)

        # Create binary hash
        binary_hash = (resized > mean).astype(int)

        return binary_hash.flatten()


class NormalizedFeatureAnalyzer:
    def __init__(self, frames_folder):
        """
        Initialize the analyzer with frames from the specified folder
        """
        self.frames = self.load_frames(frames_folder)

        # Define feature extraction methods
        self.extraction_methods = {
            'KAZE': FeatureExtractor.kaze_extractor,
            'SIFT': FeatureExtractor.sift_extractor,
            'SVD': FeatureExtractor.svd_extractor,
            'LBP Histogram': FeatureExtractor.lbp_histogram_extractor,
            'Fuzzy Entropy': FeatureExtractor.fuzzy_extractor,
        }

    def load_frames(self, folder_path):
        """
        Load frames from the specified folder
        """
        frames = []
        frame_names = []
        for filename in sorted(os.listdir(folder_path)):
            if filename.endswith(('.jpg', '.png', '.jpeg')):
                frame_path = os.path.join(folder_path, filename)
                frame = cv2.imread(frame_path)
                if frame is not None:
                    frames.append(frame)
                    frame_names.append(filename)

        print(f"Loaded {len(frames)} frames: {frame_names}")
        return frames

    def normalize_distances(self, distances):
        """
        Normalize distances to [0, 1] range
        """
        min_dist = np.min(distances)
        max_dist = np.max(distances)

        # Avoid division by zero
        if min_dist == max_dist:
            return np.zeros_like(distances)

        return (distances - min_dist) / (max_dist - min_dist)

    def analyze_normalized_features(self):
        """
        Comprehensive normalized feature analysis
        """
        # Warn about small dataset
        if len(self.frames) < 10:
            print("\n!!! WARNING: Small dataset detected. Results may not be statistically significant !!!")

        # Feature analysis results
        feature_analysis = {}

        # Analyze each extraction method
        for method_name, extractor in self.extraction_methods.items():
            # Extract normalized features for all frames
            features = [extractor(frame) for frame in self.frames]

            # Calculate normalized distances
            # Use multiple distance metrics for comprehensive analysis
            distance_metrics = {
                'Manhattan': self.normalize_distances(pdist(features, metric='cityblock')),
                'Cosine': self.normalize_distances(pdist(features, metric='cosine')),
                'Euclidean': self.normalize_distances(pdist(features, metric='euclidean')),


            }

            # Compile detailed metrics for each distance metric
            method_metrics = {}
            for metric_name, distances in distance_metrics.items():
                method_metrics[metric_name] = {
                    'avg_distance': np.mean(distances),
                    'distance_std': np.std(distances),
                    'max_distance': np.max(distances),
                    'min_distance': np.min(distances)
                }

            # Store method analysis
            feature_analysis[method_name] = method_metrics

        return feature_analysis

    def visualize_normalized_results(self, feature_analysis):
        """
        Create comprehensive visualization for normalized features
        """
        plt.figure(figsize=(20, 15))

        # Prepare data for plotting
        methods = list(feature_analysis.keys())
        distance_metrics = list(list(feature_analysis.values())[0].keys())

        # Plot for each distance metric
        for i, metric in enumerate(distance_metrics, 1):
            plt.subplot(2, 2, i)

            # Prepare data
            avg_distances = [method[metric]['avg_distance'] for method in feature_analysis.values()]
            distance_stds = [method[metric]['distance_std'] for method in feature_analysis.values()]

            # Create bar plot
            x = np.arange(len(methods))
            width = 0.35

            plt.bar(x - width / 2, avg_distances, width, label='Avg Distance', color='blue', alpha=0.7)
            # plt.bar(x + width / 2, distance_stds, width, label='Distance Std', color='red', alpha=0.7)

            plt.title(f'{metric} Distance Analysis (Normalized)')
            plt.xlabel('Feature Extraction Method')
            plt.ylabel('Normalized Distance')
            plt.xticks(x, methods, rotation=45)
            plt.ylim(0, 1)  # Set y-axis to [0, 1]
            plt.legend()

        plt.tight_layout()
        plt.show()

        # Detailed print of normalized metrics
        print("\nNormalized Feature Extraction Metrics:")
        for method, metrics in feature_analysis.items():
            print(f"\n{method}:")
            for metric_name, stat in metrics.items():
                print(f"  {metric_name} Distances:")
                for key, value in stat.items():
                    print(f"    {key}: {value:.4f}")

    def visualize_normalized_results_curve(self,feature_analysis):
        """
        Create line plots for normalized distances for each feature extractor.
        """
        plt.figure(figsize=(20, 10))

        # Prepare data for plotting
        methods = list(feature_analysis.keys())
        distance_metrics = list(list(feature_analysis.values())[0].keys())

        # Iterate through each distance metric
        for i, metric in enumerate(distance_metrics, 1):
            plt.subplot(2, 2, i)

            # Prepare data
            avg_distances = [method[metric]['avg_distance'] for method in feature_analysis.values()]
            distance_stds = [method[metric]['distance_std'] for method in feature_analysis.values()]

            # Plot average distances
            plt.plot(methods, avg_distances, marker='o', label='Avg Distance', color='blue', linestyle='-', linewidth=2)

            # Plot distance standard deviations
            plt.plot(methods, distance_stds, marker='s', label='Distance Std', color='red', linestyle='--', linewidth=2)

            plt.title(f'{metric} Distance Analysis (Normalized)', fontsize=14)
            plt.xlabel('Feature Extraction Method', fontsize=12)
            plt.ylabel('Normalized Distance', fontsize=12)
            plt.xticks(rotation=45, fontsize=10)
            plt.yticks(fontsize=10)
            plt.grid(alpha=0.5)
            plt.legend(fontsize=12)
            plt.tight_layout()

        plt.suptitle('Normalized Distance Analysis for Feature Extraction Methods', fontsize=16, y=1.02)
        plt.show()

        # Detailed print of normalized metrics
        print("\nNormalized Feature Extraction Metrics:")
        for method, metrics in feature_analysis.items():
            print(f"\n{method}:")
            for metric_name, stat in metrics.items():
                print(f"  {metric_name} Distances:")
                for key, value in stat.items():
                    print(f"    {key}: {value:.4f}")

def main():
    # Specify the folder containing frames
    frames_folder = 'frames-2'  # Update this path

    # Create analyzer
    analyzer = NormalizedFeatureAnalyzer(frames_folder)

    # Analyze normalized features
    feature_analysis = analyzer.analyze_normalized_features()

    # Visualize results
    analyzer.visualize_normalized_results(feature_analysis)
    analyzer.visualize_normalized_results_curve(feature_analysis)


if __name__ == '__main__':
    main()