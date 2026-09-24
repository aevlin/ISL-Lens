from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from huggingface_hub import snapshot_download


ROOT = Path(__file__).resolve().parents[1]
DATASET_ID = "vidit031/isl-isolated-40words"
DATA_ROOT = ROOT / "data" / "raw" / "isl_isolated_40words"
MANIFEST_PATH = ROOT / "data" / "manifests" / "isl_isolated_40words_manifest.json"


def main() -> None:
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "0")
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    local_dir = snapshot_download(
        repo_id=DATASET_ID,
        repo_type="dataset",
        local_dir=DATA_ROOT,
        local_dir_use_symlinks=False,
    )
    metadata_path = Path(local_dir) / "metadata.csv"
    if not metadata_path.exists():
        raise FileNotFoundError(f"metadata.csv not found in {local_dir}")

    df = pd.read_csv(metadata_path)
    samples = []
    for index, row in df.iterrows():
        rel_path = str(row["video_path"])
        samples.append(
            {
                "sample_id": f"isl40_{index:05d}",
                "dataset_id": "isl_isolated_40words",
                "source_url": row.get("download_url"),
                "relative_path": rel_path,
                "media_type": "video",
                "file_size_bytes": None,
                "sha256": row.get("sha256"),
                "perceptual_hash": row.get("phash"),
                "label_raw": str(row.get("original_label", row.get("word"))),
                "label_normalized": str(row.get("normalized_word", row.get("word"))).upper(),
                "vocabulary_id": str(row.get("normalized_word", row.get("word"))).lower().replace(" ", "_"),
                "signer_id": None if pd.isna(row.get("signer")) else str(row.get("signer")),
                "signer_metadata": {},
                "recording_id": None,
                "source_video_id": row.get("original_filename"),
                "fps": None if pd.isna(row.get("fps")) else float(row.get("fps")),
                "duration_seconds": None if pd.isna(row.get("duration")) else float(row.get("duration")),
                "frame_count": None,
                "width": None,
                "height": None,
                "split": "unassigned",
                "split_strategy": None,
                "license": str(row.get("license")),
                "annotation_type": "isolated_gloss",
                "quality_status": "usable" if float(row.get("quality_score", 0.0)) >= 0.85 else "needs_review",
                "duplicate_status": str(row.get("duplicate_status", "unknown")),
                "review_status": "needs_manual_review"
                if str(row.get("review_status", "")).lower().startswith("needs")
                else "approved",
                "notes": f"source_dataset={row.get('dataset')}; repository={row.get('repository')}; paper={row.get('paper')}",
            }
        )

    vocabulary = []
    for class_index, gloss in enumerate(sorted({sample["label_normalized"] for sample in samples})):
        vocabulary.append(
            {
                "vocabulary_id": gloss.lower().replace(" ", "_"),
                "normalized_gloss": gloss,
                "display_label": gloss.title(),
                "class_index": class_index,
                "supported": True,
                "source_labels": sorted(
                    {
                        sample["label_raw"]
                        for sample in samples
                        if sample["label_normalized"] == gloss
                    }
                ),
                "known_similar_glosses": [],
                "notes": "Initial prototype vocabulary from isl-isolated-40words.",
            }
        )

    manifest = {
        "manifest_version": "v0.1.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset": {
            "dataset_id": "isl_isolated_40words",
            "name": "ISL Isolated Word Dataset 40 Words",
            "version": "huggingface_snapshot",
            "source_url": f"https://huggingface.co/datasets/{DATASET_ID}",
            "citation": "Cite upstream datasets listed in the Hugging Face dataset card.",
            "license": "multi-source-aggregate",
            "license_url": f"https://huggingface.co/datasets/{DATASET_ID}",
            "task_type": "isolated_sign",
            "notes": "Prototype dataset. Respect upstream licenses and manual review flags.",
        },
        "vocabulary": vocabulary,
        "samples": samples,
    }
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Downloaded snapshot: {local_dir}")
    print(f"Wrote manifest: {MANIFEST_PATH}")
    print(f"Samples: {len(samples)}")
    print(f"Classes: {len(vocabulary)}")


if __name__ == "__main__":
    main()
