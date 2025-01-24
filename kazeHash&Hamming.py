import os
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim


class ShotBoundaryDetector:
    def __init__(self, frames_folder, threshold=0.3):
        self.frames_folder = frames_folder
        self.threshold = threshold
        self.frame_paths = self._get_sorted_frame_paths()

    def _get_sorted_frame_paths(self):
        return sorted([
            os.path.join(self.frames_folder, f)
            for f in os.listdir(self.frames_folder)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ])

    def detect_shot_boundaries(self):
        shot_boundaries = []

        for i in range(len(self.frame_paths) - 1):
            # Read consecutive frames
            frame1 = cv2.imread(self.frame_paths[i], cv2.IMREAD_GRAYSCALE)
            frame2 = cv2.imread(self.frame_paths[i + 1], cv2.IMREAD_GRAYSCALE)

            # Resize frames to ensure consistent comparison
            frame1 = cv2.resize(frame1, (256, 256))
            frame2 = cv2.resize(frame2, (256, 256))

            # Compute structural similarity index
            similarity = ssim(frame1, frame2)
            difference = 1 - similarity

            # Classify transition type
            if difference > self.threshold:
                transition_type = "Abrupt" if difference > 0.5 else "Gradual"
                shot_boundaries.append({
                    'frame1': os.path.basename(self.frame_paths[i]),
                    'frame2': os.path.basename(self.frame_paths[i + 1]),
                    'difference': difference,
                    'type': transition_type
                })

        return shot_boundaries

    def advanced_shot_boundary_detection(self):
        # Advanced detection with multiple similarity measures
        shot_boundaries = []

        for i in range(len(self.frame_paths) - 1):
            frame1 = cv2.imread(self.frame_paths[i])
            frame2 = cv2.imread(self.frame_paths[i + 1])

            # Multiple feature extraction techniques
            ssim_score = self._compute_ssim(frame1, frame2)
            histogram_diff = self._compute_histogram_difference(frame1, frame2)
            edge_difference = self._compute_edge_difference(frame1, frame2)

            # Combined difference metric
            combined_diff = np.mean([ssim_score, histogram_diff, edge_difference])

            if combined_diff > self.threshold:
                transition_type = "Abrupt" if combined_diff > 0.5 else "Gradual"
                shot_boundaries.append({
                    'frame1': os.path.basename(self.frame_paths[i]),
                    'frame2': os.path.basename(self.frame_paths[i + 1]),
                    'difference': combined_diff,
                    'type': transition_type
                })

        return shot_boundaries

    def _compute_ssim(self, frame1, frame2):
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        return ssim(gray1, gray2)

    def _compute_histogram_difference(self, frame1, frame2):
        hist1 = cv2.calcHist([frame1], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        hist2 = cv2.calcHist([frame2], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        return cv2.compareHist(hist1, hist2, cv2.HISTCMP_CHISQR)

    def _compute_edge_difference(self, frame1, frame2):
        edges1 = cv2.Canny(frame1, 100, 200)
        edges2 = cv2.Canny(frame2, 100, 200)
        return np.mean(np.abs(edges1.astype(int) - edges2.astype(int))) / 255.0


def main():
    frames_folder = "frames-1"
    detector = ShotBoundaryDetector(frames_folder)

    # Basic shot boundary detection
    basic_boundaries = detector.detect_shot_boundaries()
    print("Basic Shot Boundaries:")
    for boundary in basic_boundaries:
        print(
            f"{boundary['type']} Transition: {boundary['frame1']} -> {boundary['frame2']} (Difference: {boundary['difference']:.4f})")

    # Advanced shot boundary detection
    advanced_boundaries = detector.advanced_shot_boundary_detection()
    print("\nAdvanced Shot Boundaries:")
    for boundary in advanced_boundaries:
        print(
            f"{boundary['type']} Transition: {boundary['frame1']} -> {boundary['frame2']} (Difference: {boundary['difference']:.4f})")


if __name__ == "__main__":
    main()