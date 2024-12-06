import os
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from typing import Tuple

import cv2
import imagehash
import pytesseract
from cv2.typing import MatLike
from PIL import Image
from tqdm import tqdm


def extract_key_frames(
    video_path: str, interval: int = 1, text_density_threshold: int = 25
):
    """Extract key frames with completed text content using concurrent futures."""
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_interval = int(fps * interval)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    frames_to_process = []
    frame_indices = []

    # Collect frames at specified intervals
    with tqdm(total=total_frames // frame_interval, desc="Loading frames") as pbar:
        frame_idx = 0
        while frame_idx < total_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, curr_frame = cap.read()
            if not ret:
                break

            frames_to_process.append(curr_frame)
            frame_indices.append(frame_idx)
            frame_idx += frame_interval
            pbar.update(1)

    cap.release()

    # Process frames in parallel to calculate text densities
    def process_frame(frame):
        return _calculate_text_density(frame)

    text_densities = []
    with ThreadPoolExecutor() as executor:
        with tqdm(total=len(frames_to_process), desc="Processing frames") as pbar:
            for result in executor.map(process_frame, frames_to_process):
                text_densities.append(result)
                pbar.update(1)

    # Filter frames based on text density threshold
    selected_frames = [
        (idx, frame)
        for idx, frame, density in zip(frame_indices, frames_to_process, text_densities)
        if density >= text_density_threshold
    ]

    unique_frames = _select_highest_density_frames(
        selected_frames,
        [text_densities[frame_indices.index(f[0])] for f in selected_frames],
    )

    return unique_frames


def save_frames(frames: list[Tuple[int, MatLike]], output_dir: str):
    """Save frames as images"""

    os.makedirs(output_dir, exist_ok=True)
    for idx, frame in tqdm(frames, desc="Saving frames.."):
        filename = f"{output_dir}/frame_{idx}.jpg"
        cv2.imwrite(filename, frame)


def _calculate_text_density(frame: MatLike) -> int:
    """
    Return the number of characters in a frame by using OCR
    """

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    binary = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
    )

    # Clean up noise with morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    binary_cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    # OCR on the cleaned image
    text = pytesseract.image_to_string(Image.fromarray(binary_cleaned))
    return sum(c.isalnum() for c in text)


def _select_highest_density_frames(
    frames: list[Tuple[int, MatLike]],
    densities: list[int],
    similarity_threshold: int = 25,
):
    """
    Group similar frames using perceptual hashing and select the one
    with the highest text density in each group.
    """

    grouped_frames = defaultdict(list)
    hashes = []  # Store perceptual hashes to group similar frames

    for (idx, frame), density in zip(frames, densities):
        img = Image.fromarray(frame)
        h = imagehash.average_hash(img)

        # Group similar frames based on perceptual hash
        for i, existing_hash in enumerate(hashes):
            print(abs(h - existing_hash))
            if abs(h - existing_hash) < similarity_threshold:  # Similar frame
                grouped_frames[i].append((idx, frame, density))
                break
        else:  # No similar frame found, create a new group
            group_idx = len(hashes)
            hashes.append(h)
            grouped_frames[group_idx].append((idx, frame, density))

    # Select the frame with the highest text density from each group
    unique_frames = []
    for group in grouped_frames.values():
        best_frame = max(group, key=lambda x: x[2])  # Select frame with max density
        unique_frames.append((best_frame[0], best_frame[1]))

    return unique_frames
