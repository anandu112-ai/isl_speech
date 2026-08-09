"""
Video Preprocessing and Feature Extraction Processor for INCLUDE-50 Pipeline.
Handles uniform 32-frame temporal sampling, MediaPipe feature extraction,
and saving compressed NumPy .npz feature files.
"""

from pathlib import Path
from typing import Tuple
import cv2
import numpy as np

from src.features.extractor import LandmarkExtractor


def sample_frame_indices(total_frames: int, num_frames: int = 32) -> np.ndarray:
    """
    Uniformly samples num_frames indices across total_frames.
    Interpolates indices safely if total_frames < num_frames.
    """
    if total_frames <= 0:
        return np.zeros(num_frames, dtype=int)

    if total_frames >= num_frames:
        return np.linspace(0, total_frames - 1, num_frames, dtype=int)
    else:
        # Interpolate frame indices for short videos
        indices = np.linspace(0, total_frames - 1, num_frames)
        return np.round(indices).astype(int)


def process_video_file(
    video_path: Path,
    output_npz_path: Path,
    extractor: LandmarkExtractor,
    num_frames: int = 32,
) -> Tuple[bool, str, Tuple[int, int]]:
    """
    Reads a .MOV video, samples 32 frames, extracts landmarks, and saves as .npz.
    Returns (success, message, (num_frames, feature_dim)).
    """
    vpath = Path(video_path)
    out_path = Path(output_npz_path)

    if not vpath.exists():
        return False, f"Video file not found: {vpath}", (0, 0)

    cap = cv2.VideoCapture(str(vpath))
    if not cap.isOpened():
        return False, f"OpenCV failed to open video: {vpath}", (0, 0)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        # Read frames sequentially to count
        frameCount = 0
        while True:
            ret, _ = cap.read()
            if not ret:
                break
            frameCount += 1
        total_frames = frameCount
        cap.release()
        cap = cv2.VideoCapture(str(vpath))

    if total_frames == 0:
        cap.release()
        return False, f"Video has 0 frames: {vpath}", (0, 0)

    target_indices = set(sample_frame_indices(total_frames, num_frames=num_frames))
    ordered_indices = sample_frame_indices(total_frames, num_frames=num_frames)

    frame_map = {}
    current_frame = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if current_frame in target_indices:
            frame_map[current_frame] = extractor.extract_frame_features(frame)
        current_frame += 1

    cap.release()

    if not frame_map:
        return False, f"No frames could be read from video: {vpath}", (0, 0)

    # Assemble feature matrix of shape (num_frames, 225)
    feature_list = []
    fallback_feature = next(iter(frame_map.values()))

    for idx in ordered_indices:
        feat = frame_map.get(idx, fallback_feature)
        feature_list.append(feat)

    feature_matrix = np.array(feature_list, dtype=np.float32)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, features=feature_matrix)

    return True, "Success", feature_matrix.shape
