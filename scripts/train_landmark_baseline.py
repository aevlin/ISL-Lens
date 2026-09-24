from __future__ import annotations

import json
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path("/tmp") / "isl_lens_matplotlib"))

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import GroupShuffleSplit, StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tqdm import tqdm

import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.preprocessing.landmark_features import SUMMARY_FEATURES, extract_video_features


DATA_ROOT = ROOT / "data" / "raw" / "isl_isolated_40words"
METADATA = DATA_ROOT / "metadata.csv"
FEATURE_CACHE = ROOT / "data" / "processed" / "isl40_landmark_features.npz"
MODEL_DIR = ROOT / "app" / "static" / "models"
REPORT_DIR = ROOT / "reports" / "experiments"


def load_metadata() -> pd.DataFrame:
    if not METADATA.exists():
        raise FileNotFoundError("Dataset metadata not found. Run scripts/download_isl40.py first.")
    df = pd.read_csv(METADATA)
    df["label"] = df["normalized_word"].astype(str).str.upper()
    df["video_abs"] = df["video_path"].map(lambda value: DATA_ROOT / str(value))
    df = df[df["video_abs"].map(Path.exists)].copy()
    if df.empty:
        raise ValueError("No downloaded videos found.")
    return df


def _process_single_row(row_dict: dict, sequence_length: int = 32) -> tuple[np.ndarray | None, str, dict]:
    video_abs = Path(row_dict["video_abs"])
    label = row_dict["label"]
    video_path = row_dict["video_path"]
    signer = None if pd.isna(row_dict.get("signer")) else str(row_dict.get("signer"))
    try:
        feature, meta = extract_video_features(video_abs, sequence_length=sequence_length)
        if feature.shape[0] != SUMMARY_FEATURES:
            raise ValueError(f"Unexpected feature shape {feature.shape}")
        meta.update({"video_path": video_path, "label": label, "signer": signer})
        return feature, label, meta
    except Exception as exc:
        return None, label, {"video_path": video_path, "label": label, "error": str(exc)}


def build_or_load_features(df: pd.DataFrame, sequence_length: int = 32) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    if FEATURE_CACHE.exists():
        cache = np.load(FEATURE_CACHE, allow_pickle=True)
        return cache["X"], cache["y"], list(cache["meta"])

    FEATURE_CACHE.parent.mkdir(parents=True, exist_ok=True)
    rows = df.to_dict(orient="records")
    n_jobs = max(1, min(os.cpu_count() or 4, 8))
    results = joblib.Parallel(n_jobs=n_jobs)(
        joblib.delayed(_process_single_row)(row, sequence_length)
        for row in tqdm(rows, desc=f"Extracting landmarks ({n_jobs} workers)")
    )
    features = []
    labels = []
    metadata = []
    for feature, label, meta in results:
        metadata.append(meta)
        if feature is not None:
            features.append(feature)
            labels.append(label)

    X = np.vstack(features).astype(np.float32)
    y = np.asarray(labels)
    np.savez_compressed(FEATURE_CACHE, X=X, y=y, meta=np.asarray(metadata, dtype=object))
    return X, y, metadata


def split_data(extraction_meta: list[dict], y: np.ndarray) -> tuple[np.ndarray, np.ndarray, str]:
    signers = [str(m.get("signer") or "") for m in extraction_meta if isinstance(m, dict) and "error" not in m]
    usable_groups = len(signers) == len(y) and len(set(signers)) >= 4 and min(Counter(signers).values()) >= 2
    if usable_groups:
        splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        train_idx, test_idx = next(splitter.split(np.zeros(len(y)), y, signers))
        return train_idx, test_idx, "signer_group_holdout"

    class_counts = Counter(y)
    if min(class_counts.values()) >= 2:
        splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        train_idx, test_idx = next(splitter.split(np.zeros(len(y)), y))
        return train_idx, test_idx, "stratified_random"

    from sklearn.model_selection import ShuffleSplit
    splitter = ShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(splitter.split(np.zeros(len(y)), y))
    return train_idx, test_idx, "random_shuffle"


def top_k_accuracy(probabilities: np.ndarray, truth: np.ndarray, k: int) -> float:
    top = np.argsort(probabilities, axis=1)[:, -k:]
    return float(np.mean([truth[i] in top[i] for i in range(len(truth))]))


def export_browser_model(pipeline: Pipeline, label_encoder: LabelEncoder, threshold: float, metrics: dict) -> None:
    scaler = pipeline.named_steps["scaler"]
    classifier = pipeline.named_steps["classifier"]
    payload = {
        "version": "v0.1.0-landmark-logreg",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model_type": "multinomial_logistic_regression",
        "feature_spec": "2-hand MediaPipe landmarks normalized per hand; sequence summary mean/std/min/max/velocity mean/velocity std",
        "sequence_length": 32,
        "input_features": int(SUMMARY_FEATURES),
        "classes": label_encoder.classes_.tolist(),
        "unknown_threshold": float(threshold),
        "scaler_mean": scaler.mean_.astype(float).tolist(),
        "scaler_scale": scaler.scale_.astype(float).tolist(),
        "coef": classifier.coef_.astype(float).tolist(),
        "intercept": classifier.intercept_.astype(float).tolist(),
        "metrics": metrics,
    }
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    (MODEL_DIR / "landmark_logreg.json").write_text(json.dumps(payload), encoding="utf-8")
    joblib.dump(pipeline, MODEL_DIR / "landmark_logreg.joblib")


def main() -> None:
    df = load_metadata()
    X, y_text, extraction_meta = build_or_load_features(df)
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_text)
    train_idx, test_idx, split_strategy = split_data(extraction_meta, y)

    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    max_iter=2500,
                    class_weight="balanced",
                    solver="lbfgs",
                    n_jobs=1,
                    random_state=42,
                ),
            ),
        ]
    )
    pipeline.fit(X[train_idx], y[train_idx])

    probabilities = pipeline.predict_proba(X[test_idx])
    predictions = np.argmax(probabilities, axis=1)
    confidence = probabilities.max(axis=1)
    correct_confidence = confidence[predictions == y[test_idx]]
    threshold = float(np.percentile(correct_confidence, 10)) if len(correct_confidence) else 0.75
    threshold = max(0.45, min(threshold, 0.90))

    metrics = {
        "dataset": "vidit031/isl-isolated-40words",
        "split_strategy": split_strategy,
        "samples_total": int(len(y)),
        "samples_train": int(len(train_idx)),
        "samples_test": int(len(test_idx)),
        "classes": int(len(label_encoder.classes_)),
        "top1_accuracy": float(accuracy_score(y[test_idx], predictions)),
        "top3_accuracy": top_k_accuracy(probabilities, y[test_idx], 3),
        "macro_f1": float(f1_score(y[test_idx], predictions, average="macro")),
        "weighted_f1": float(f1_score(y[test_idx], predictions, average="weighted")),
        "unknown_threshold": threshold,
        "threshold_note": "Threshold estimated from held-out correct predictions only; not a true open-set threshold because no unknown validation set is present.",
    }

    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "metrics": metrics,
        "classification_report": classification_report(
            y[test_idx],
            predictions,
            labels=list(range(len(label_encoder.classes_))),
            target_names=label_encoder.classes_,
            output_dict=True,
            zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(y[test_idx], predictions).tolist(),
        "extraction_summary": {
            "mean_detection_rate": float(
                np.mean([item.get("detection_rate", 0.0) for item in extraction_meta if "detection_rate" in item])
            ),
            "feature_cache": str(FEATURE_CACHE),
        },
    }

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORT_DIR / "isl40_landmark_logreg_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    export_browser_model(pipeline, label_encoder, threshold, metrics)

    print(json.dumps(metrics, indent=2))
    print(f"Report: {report_path}")
    print(f"Browser model: {MODEL_DIR / 'landmark_logreg.json'}")


if __name__ == "__main__":
    main()
