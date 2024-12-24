import os
import cv2
import hashlib


def image_to_kaze_hash(image_path):
    # Read the image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Image at path {image_path} could not be loaded.")

    # Initialize the KAZE feature detector
    kaze = cv2.KAZE_create()

    # Detect keypoints and descriptors
    keypoints, descriptors = kaze.detectAndCompute(image, None)
    if descriptors is None:
        raise ValueError(f"No descriptors found in the image.")

    # Flatten the descriptor array
    descriptor_flat = descriptors.flatten()

    # Convert the descriptor to a byte string
    descriptor_bytes = descriptor_flat.tobytes()

    # Generate the hash using SHA-256
    hash_object = hashlib.sha256(descriptor_bytes)
    hash_hex = hash_object.hexdigest()

    return hash_hex


def hamming_distance(hash1, hash2):
    if len(hash1) != len(hash2):
        raise ValueError("Hash lengths are not equal.")

    distance = sum(c1 != c2 for c1, c2 in zip(hash1, hash2))
    return distance


def calculate_hamming_distances(frames_folder):
    # Get a sorted list of image paths from the folder
    image_paths = sorted(
        [os.path.join(frames_folder, f) for f in os.listdir(frames_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]
    )

    # Generate KAZE hashes for each image
    hashes = [image_to_kaze_hash(image_path) for image_path in image_paths]

    # Calculate Hamming distances between consecutive hashes
    distances = []
    for i in range(len(hashes) - 1):
        distance = hamming_distance(hashes[i], hashes[i + 1])
        distances.append((image_paths[i], image_paths[i + 1], distance))

    return distances


# Example Usage
frames_folder = "path_to_frames_folder"  # Replace with your actual frames folder path
hamming_distances = calculate_hamming_distances(frames_folder)

# Print the results
for frame1, frame2, distance in hamming_distances:
    print(f"Hamming distance between {os.path.basename(frame1)} and {os.path.basename(frame2)}: {distance}")