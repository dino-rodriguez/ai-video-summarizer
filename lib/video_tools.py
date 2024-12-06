import os
from typing import Tuple

import cv2
import imagehash
import pytesseract
from cv2.typing import MatLike
from PIL import Image
from tqdm import tqdm


def extract_key_frames(video_path: str, interval: int = 1):
    """Extract information dense key frames from video"""

    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Look at 1 frame per `interval` (in seconds)
    frame_interval = int(fps * interval)

    frames = []
    ret, prev_frame = cap.read()
    frame_idx = 0

    while cap.isOpened():
        ret, curr_frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_interval == 0:
            if prev_frame is None or _is_frame_significantly_changed(
                prev_frame, curr_frame
            ):
                prev_frame = curr_frame

                if _detect_text_or_diagrams(curr_frame):
                    frames.append((frame_idx, curr_frame))

        frame_idx += 1

    cap.release()

    unique_frames = _remove_duplicates(frames)

    return unique_frames


def save_frames(frames: list[Tuple[int, MatLike]], output_dir: str):
    """Save frames as images"""

    os.makedirs(output_dir, exist_ok=True)
    for idx, frame in tqdm(frames, desc="Saving frames.."):
        filename = f"{output_dir}/frame_{idx}.jpg"
        cv2.imwrite(filename, frame)


def _detect_text_or_diagrams(
    frame: MatLike, text_threshold: int = 20, edge_threshold: int = 50
):
    """Return True if text or diagrams detected
    Uses OCR for text and contour detection for diagrams"""

    # Text Detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
    text = pytesseract.image_to_string(Image.fromarray(binary))

    # Diagram Detection
    edges = cv2.Canny(gray, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    return len(text.split()) > text_threshold or len(contours) > edge_threshold


def _is_frame_significantly_changed(
    prev_frame: MatLike, curr_frame: MatLike, threshold: int = 5000
):
    """Compute the difference between frames and return True if it exceeds the difference threshold"""

    gray_prev = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    gray_curr = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(gray_prev, gray_curr)
    _, diff_thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
    change_score = cv2.countNonZero(diff_thresh)

    return change_score > threshold


def _remove_duplicates(frames: list[Tuple[int, MatLike]]):
    """Use perceptual hashing to remove duplicate frames"""

    unique_frames = []
    hashes = set()
    for idx, frame in frames:
        img = Image.fromarray(frame)
        h = imagehash.average_hash(img)
        if h not in hashes:
            hashes.add(h)
            unique_frames.append((idx, frame))
    return unique_frames
