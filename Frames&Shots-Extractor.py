import cv2
import os
import numpy as np


def extract_all_frames(video_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)

    count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imwrite(os.path.join(output_dir, f"frame_{count:05d}.jpg"), frame)
        count += 1
    cap.release()
    return count


def calculate_histogram(image):
    hist = cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()

def detect_shots(frames_dir, total_frames, threshold=0.5):
    prev_hist = None
    shot_boundaries = []
    for i in range(total_frames):
        frame_path = os.path.join(frames_dir, f"frame_{i:05d}.jpg")
        frame = cv2.imread(frame_path)
        hist = calculate_histogram(frame)
        if prev_hist is not None:
            diff = cv2.compareHist(prev_hist, hist, cv2.HISTCMP_CORREL)
            if diff < threshold:
                shot_boundaries.append(i)
        prev_hist = hist
    return shot_boundaries




def select_representative_frames(shot_boundaries, total_frames):
    representative_frames = []
    prev_boundary = 0
    for boundary in shot_boundaries:
        mid_frame = (prev_boundary + boundary) // 2
        representative_frames.append(mid_frame)
        prev_boundary = boundary
    if prev_boundary < total_frames:
        representative_frames.append((prev_boundary + total_frames) // 2)
    return representative_frames



def save_shot_images(frames_dir, representative_frames, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for idx, frame_idx in enumerate(representative_frames):
        frame_path = os.path.join(frames_dir, f"frame_{frame_idx:05d}.jpg")
        frame = cv2.imread(frame_path)
        if frame is not None:
            output_path = os.path.join(output_dir, f"shot_{idx:05d}.jpg")
            cv2.imwrite(output_path, frame)


video_path = 'BG_37770.mpg'
output_dir = 'BG_37770_frames'
total_frames = extract_all_frames(video_path, output_dir)
print(total_frames)
shot_boundaries = detect_shots(output_dir, total_frames)
representative_frames = select_representative_frames(shot_boundaries, total_frames)
output_dir_shots = 'shots'
save_shot_images(output_dir, representative_frames, output_dir_shots)
