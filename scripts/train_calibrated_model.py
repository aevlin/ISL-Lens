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
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
FEATURE_CACHE = ROOT / "data" / "processed" / "isl40_landmark_features.npz"
USER_CORRECTIONS = ROOT / "data" / "corrections" / "user_corrections.json"
MODEL_DIR = ROOT / "app" / "static" / "models"
REPORT_DIR = ROOT / "reports" / "experiments"


def load_user_corrections() -> tuple[list[np.ndarray], list[str]]:
    if not USER_CORRECTIONS.exists():
        return [], []
    try:
        data = json.loads(USER_CORRECTIONS.read_text(encoding="utf-8") or "[]")
    except Exception:
        return [], []
    user_X = []
    user_y = []
    for entry in data:
        gloss = str(entry.get("correct_gloss") or "").upper().strip()
        if not gloss:
            continue
        exemplars = entry.get("exemplars") or ([entry.get("features")] if entry.get("features") else [])
        for feat in exemplars:
            if feat and len(feat) == 828:
                user_X.append(np.array(feat, dtype=np.float32))
                user_y.append(gloss)
    return user_X, user_y


def top_k_accuracy(probabilities: np.ndarray, truth_indices: np.ndarray, k: int) -> float:
    top = np.argsort(probabilities, axis=1)[:, -k:]
    return float(np.mean([truth_indices[i] in top[i] for i in range(len(truth_indices))]))


def main() -> None:
    if not FEATURE_CACHE.exists():
        raise FileNotFoundError(f"Feature cache {FEATURE_CACHE} not found.")

    cache = np.load(FEATURE_CACHE, allow_pickle=True)
    X_base = cache["X"].astype(np.float32)
    y_base = cache["y"].astype(str)

    user_X_list, user_y_list = load_user_corrections()
    print(f"Loaded base features: {len(X_base)} samples, user exemplars: {len(user_X_list)} samples.")

    # Keep evaluation samples out of fitting. Singleton classes stay in training
    # because they cannot support a meaningful held-out estimate.
    counts = Counter(y_base)
    eligible = np.flatnonzero(np.array([counts[label] >= 2 for label in y_base]))
    singleton_idx = np.flatnonzero(np.array([counts[label] < 2 for label in y_base]))
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    eligible_train, test_idx = next(splitter.split(X_base[eligible], y_base[eligible]))
    train_idx = np.concatenate([eligible[eligible_train], singleton_idx])
    X_base_test = X_base[test_idx]
    y_base_test = y_base[test_idx]

    # Combine training data: base data + 8x user exemplars for strong calibrated weighting
    X_train_list = [X_base[train_idx]]
    y_train_list = [y_base[train_idx]]
    if user_X_list:
        user_X_arr = np.array(user_X_list, dtype=np.float32)
        user_y_arr = np.array(user_y_list, dtype=str)
        for _ in range(8):
            X_train_list.append(user_X_arr)
            y_train_list.append(user_y_arr)

    # Unsupported vocabulary labels must not be given synthetic mean-vector
    # samples; they would be confidently predicted despite having no examples.
    X_train = np.vstack(X_train_list)
    y_train = np.concatenate(y_train_list)

    label_encoder = LabelEncoder()
    y_train_enc = label_encoder.fit_transform(y_train)

    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    max_iter=2500,
                    class_weight="balanced",
                    solver="lbfgs",
                    C=1.0,
                    random_state=42,
                ),
            ),
        ]
    )
    pipeline.fit(X_train, y_train_enc)

    # Only held-out corpus examples count toward the reported accuracy. User
    # exemplars are training data and must not also appear in evaluation.
    X_eval = X_base_test
    y_eval = y_base_test

    y_eval_enc = label_encoder.transform(y_eval)

    probabilities = pipeline.predict_proba(X_eval)
    predictions_enc = np.argmax(probabilities, axis=1)
    predictions_labels = label_encoder.inverse_transform(predictions_enc)

    top1_acc = float(accuracy_score(y_eval_enc, predictions_enc))
    top3_acc = top_k_accuracy(probabilities, y_eval_enc, 3)
    macro_f1 = float(f1_score(y_eval_enc, predictions_enc, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_eval_enc, predictions_enc, average="weighted", zero_division=0))

    if top1_acc < 0.90:
        raise RuntimeError(
            f"Candidate model scored {top1_acc * 100:.1f}% on the held-out corpus; "
            "the 90% quality gate failed, so the production model was not replaced."
        )

    threshold = 0.35

    metrics = {
        "dataset": "ISL-Lens-Integrated-Corpus (isl-isolated-40words + User-Taught Webcam Exemplars)",
        "split_strategy": "stratified_holdout_base_corpus",
        "samples_total": int(len(X_base)),
        "samples_train": int(len(train_idx)),
        "samples_eval": int(len(X_eval)),
        "user_exemplars": int(len(user_X_list)),
        "classes": int(len(label_encoder.classes_)),
        "top1_accuracy": top1_acc,
        "top3_accuracy": top3_acc,
        "accuracy_percentage": f"{top1_acc * 100:.1f}%",
        "top3_percentage": f"{top3_acc * 100:.1f}%",
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "unknown_threshold": threshold,
        "status": "HELD_OUT_ESTIMATE",
    }

    # Evaluate per class
    eval_labels_unique = np.unique(y_eval)
    report_dict = classification_report(
        y_eval,
        predictions_labels,
        labels=eval_labels_unique,
        output_dict=True,
        zero_division=0,
    )

    conf_mat = confusion_matrix(y_eval, predictions_labels, labels=eval_labels_unique).tolist()

    full_report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model_version": "v1.2.0-holdout-validated",
        "architecture": "Spatial Landmark Logistic Regression with Held-Out Corpus Evaluation and Weighted User Corrections",
        "metrics": metrics,
        "classification_report": report_dict,
        "confusion_matrix": conf_mat,
        "classes_evaluated": eval_labels_unique.tolist(),
        "evaluation_summary": {
            "samples_eval": len(X_eval),
            "base_test_samples": len(X_base_test),
            "user_samples_excluded_from_evaluation": len(user_X_list),
            "top1_accuracy": top1_acc,
            "top3_accuracy": top3_acc,
        },
    }

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORT_DIR / "isl40_landmark_logreg_report.json"
    report_path.write_text(json.dumps(full_report, indent=2), encoding="utf-8")

    # Refit the production model on every available labeled example after the
    # held-out score has been recorded. This does not change the evaluation.
    full_X_list = [X_base]
    full_y_list = [y_base]
    if user_X_list:
        full_X_list.extend([np.array(user_X_list, dtype=np.float32)] * 8)
        full_y_list.extend([np.array(user_y_list, dtype=str)] * 8)
    full_X = np.vstack(full_X_list)
    full_y = np.concatenate(full_y_list)
    label_encoder = LabelEncoder()
    full_y_enc = label_encoder.fit_transform(full_y)
    pipeline.fit(full_X, full_y_enc)

    # Export browser model
    scaler = pipeline.named_steps["scaler"]
    classifier = pipeline.named_steps["classifier"]
    browser_payload = {
        "version": "v1.2.0-holdout-validated",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model_type": "multinomial_logistic_regression",
        "feature_spec": "828-d spatial landmarks (2-hand + upper body velocity summary)",
        "sequence_length": 32,
        "input_features": 828,
        "classes": label_encoder.classes_.tolist(),
        "unknown_threshold": threshold,
        "scaler_mean": scaler.mean_.astype(float).tolist(),
        "scaler_scale": scaler.scale_.astype(float).tolist(),
        "coef": classifier.coef_.astype(float).tolist(),
        "intercept": classifier.intercept_.astype(float).tolist(),
        "metrics": metrics,
    }

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    (MODEL_DIR / "landmark_logreg.json").write_text(json.dumps(browser_payload), encoding="utf-8")
    joblib.dump(pipeline, MODEL_DIR / "landmark_logreg.joblib")

    print("\n" + "=" * 60)
    print("TRAINING & CALIBRATION COMPLETE")
    print(f"Top-1 Accuracy: {top1_acc * 100:.2f}% ({metrics['accuracy_percentage']})")
    print(f"Top-3 Accuracy: {top3_acc * 100:.2f}% ({metrics['top3_percentage']})")
    print(f"Macro F1-Score: {macro_f1 * 100:.2f}%")
    print(f"Weighted F1-Score: {weighted_f1 * 100:.2f}%")
    print(f"Saved Report: {report_path}")
    print(f"Saved Model: {MODEL_DIR / 'landmark_logreg.json'}")
    print("=" * 60)


if __name__ == "__main__":
    main()
