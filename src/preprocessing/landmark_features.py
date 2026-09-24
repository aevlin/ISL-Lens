from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path("/tmp") / "isl_lens_matplotlib"))

import cv2
import mediapipe as mp
import numpy as np


FRAME_FEATURES = 2 * 21 * 3 + 12
SUMMARY_FEATURES = FRAME_FEATURES * 6


def normalize_hand(landmarks: list[float]) -> np.ndarray:
    points = np.asarray(landmarks, dtype=np.float32).reshape(21, 3)
    wrist = points[0].copy()
    points[:, :3] -= wrist[:3]
    scale = float(np.max(np.linalg.norm(points[:, :2], axis=1)))
    if scale < 1e-6:
        scale = 1.0
    points[:, :3] /= scale
    return points.reshape(-1)


def frame_vector(results) -> np.ndarray:
    vector = np.zeros(FRAME_FEATURES, dtype=np.float32)
    if not results.multi_hand_landmarks:
        return vector

    handedness = results.multi_handedness or []
    wrists = [None, None]
    raw_wrists = [np.zeros(3, dtype=np.float32), np.zeros(3, dtype=np.float32)]

    for hand_index, hand_landmarks in enumerate(results.multi_hand_landmarks[:2]):
        label = "Unknown"
        if hand_index < len(handedness):
            label = handedness[hand_index].classification[0].label
        slot = 0 if label == "Left" else 1

        coords = []
        for landmark in hand_landmarks.landmark:
            coords.extend([landmark.x, landmark.y, landmark.z])

        wrists[slot] = np.array([hand_landmarks.landmark[0].x, hand_landmarks.landmark[0].y, hand_landmarks.landmark[0].z], dtype=np.float32)
        raw_wrists[slot] = wrists[slot]

        start = slot * 21 * 3
        vector[start : start + 21 * 3] = normalize_hand(coords)

    # Append 12 Spatial Body Region Features
    sp_start = 2 * 21 * 3
    vector[sp_start : sp_start + 3] = raw_wrists[0] # Left Wrist raw X, Y, Z
    vector[sp_start + 3 : sp_start + 6] = raw_wrists[1] # Right Wrist raw X, Y, Z

    w1, w2 = raw_wrists[0], raw_wrists[1]
    has_w1 = wrists[0] is not None
    has_w2 = wrists[1] is not None

    if has_w1 and has_w2:
        vector[sp_start + 6] = float(np.linalg.norm(w1[:2] - w2[:2])) # 2D hand distance
        vector[sp_start + 7] = float(np.linalg.norm(w1 - w2))        # 3D hand distance
        vector[sp_start + 8] = float(w1[1] - w2[1])                   # Y displacement
        vector[sp_start + 9] = float(w1[0] - w2[0])                   # X displacement
    else:
        vector[sp_start + 6 : sp_start + 10] = 0.0

    # Body region code (1 = Head/Face, 2 = Chest/Shoulders, 3 = Lower/Neutral)
    for idx, w in enumerate([w1, w2]):
        if wrists[idx] is None:
            region = 0.0
        elif w[1] < 0.35:
            region = 1.0 # Head / Face region
        elif w[1] <= 0.65:
            region = 2.0 # Chest / Shoulder region
        else:
            region = 3.0 # Lower / Neutral space
        vector[sp_start + 10 + idx] = region

    return vector


def summarize_sequence(frames: np.ndarray) -> np.ndarray:
    if frames.size == 0:
        frames = np.zeros((1, FRAME_FEATURES), dtype=np.float32)
    velocity = np.diff(frames, axis=0)
    if velocity.size == 0:
        velocity = np.zeros_like(frames)
    summary = np.concatenate(
        [
            frames.mean(axis=0),
            frames.std(axis=0),
            frames.min(axis=0),
            frames.max(axis=0),
            velocity.mean(axis=0),
            velocity.std(axis=0),
        ]
    )
    return np.nan_to_num(summary, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def extract_video_features(
    video_path: Path, sequence_length: int = 32, hands_instance: mp.solutions.hands.Hands | None = None
) -> tuple[np.ndarray, dict]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if total_frames > 0:
        frame_indices = np.linspace(0, max(total_frames - 1, 0), sequence_length).astype(int)
    else:
        frame_indices = np.arange(sequence_length)

    vectors = []
    detected_frames = 0

    def _process_with_hands(hands):
        nonlocal detected_frames
        for frame_index in frame_indices:
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
            ok, frame = capture.read()
            if not ok:
                vectors.append(np.zeros(FRAME_FEATURES, dtype=np.float32))
                continue
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)
            vec = frame_vector(results)
            if np.any(vec):
                detected_frames += 1
            vectors.append(vec)

    if hands_instance is not None:
        _process_with_hands(hands_instance)
    else:
        mp_hands = mp.solutions.hands
        with mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=2,
            model_complexity=1,
            min_detection_confidence=0.45,
        ) as hands:
            _process_with_hands(hands)

    capture.release()
    sequence = np.vstack(vectors).astype(np.float32)
    meta = {
        "total_frames": total_frames,
        "sampled_frames": int(sequence_length),
        "detected_frames": int(detected_frames),
        "detection_rate": float(detected_frames / max(sequence_length, 1)),
    }
    return summarize_sequence(sequence), meta
