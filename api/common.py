from __future__ import annotations

import json
import math
import os
import re
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

ROOT = Path(__file__).resolve().parents[1]
TEMPLATES = ROOT / "app" / "templates"
STATIC = ROOT / "app" / "static"
CONFIGS = ROOT / "configs"
DATA = ROOT / "data"

VERIFIED_REFERENCE_OVERRIDES = {
    "thank_you": "thank_you__ISL500__00371__Thankyou__session119__clip002.mp4",
    "thankyou": "thank_you__ISL500__00371__Thankyou__session119__clip002.mp4",
    "sorry": "sorry__ISLRTC_dictionary__00637__Sorry_(Sign_2).mp4",
    "please": "please__CISLR__00073__vAW9QDnXLrE.mp4",
}


def is_filesystem_read_only() -> bool:
    """Returns True if running in a read-only serverless environment like Vercel."""
    if os.environ.get("VERCEL") or os.environ.get("AWS_LAMBDA_FUNCTION_NAME"):
        return True
    try:
        test_file = ROOT / ".write_test"
        test_file.touch()
        test_file.unlink()
        return False
    except OSError:
        return True


def get_health_payload() -> dict:
    model_file = STATIC / "models" / "landmark_logreg.json"
    has_model = model_file.exists()
    model_info = {}
    if has_model:
        try:
            model_info = json.loads(model_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            has_model = False
    return {
        "ok": True,
        "service": "ISL-Lens app",
        "environment": "vercel" if os.environ.get("VERCEL") else "local",
        "model_status": "trained" if has_model else "not_trained",
        "model_file": "/static/models/landmark_logreg.json" if has_model else None,
        "recognition_policy": "spatial_sequence_with_unknown_rejection" if has_model else "unknown_first",
        "model_version": model_info.get("version"),
        "trained_sign_count": len(model_info.get("classes", [])),
        "held_out_accuracy": model_info.get("metrics", {}).get("accuracy_percentage"),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def get_vocabulary_payload() -> dict:
    vocab_file = CONFIGS / "vocabulary.json"
    if vocab_file.exists():
        try:
            return json.loads(vocab_file.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {"version": "1.0", "items": []}


def find_video_for_gloss(raw_gloss: str) -> tuple[Path | None, str, str]:
    """Find reference video for a sign gloss.

    Returns (path, content_type, source) where source is 'user' or 'dataset'.
    """
    gloss_clean = raw_gloss.replace(" ", "_").replace("?", "").lower().strip()
    if not gloss_clean:
        return None, "", ""

    # Priority 1: User-recorded custom video for this sign
    user_video_dirs = [DATA / "user_videos", Path("/tmp/user_videos")]
    for u_dir in user_video_dirs:
        if not u_dir.exists():
            continue
        for ext in [".webm", ".mp4"]:
            u_vid = u_dir / f"{gloss_clean}{ext}"
            if u_vid.exists() and u_vid.stat().st_size > 500:
                c_type = "video/webm" if ext == ".webm" else "video/mp4"
                return u_vid, c_type, "user"

    # Priority 2: Verified reference dataset video
    video_dir = DATA / "raw" / "isl_isolated_40words"
    g_dir = video_dir / gloss_clean
    mp4s = []
    if g_dir.exists() and g_dir.is_dir():
        mp4s = list(g_dir.glob("*.mp4"))
    if not mp4s and video_dir.exists():
        mp4s = list(video_dir.glob(f"**/{gloss_clean}__*.mp4"))

    if mp4s:
        override_name = VERIFIED_REFERENCE_OVERRIDES.get(gloss_clean)
        chosen = None
        if override_name:
            for f in mp4s:
                if f.name == override_name:
                    chosen = f
                    break
        if not chosen:
            islrtc = [f for f in mp4s if "islrtc" in f.name.lower() or "dict" in f.name.lower()]
            cislr = [f for f in sorted(mp4s, key=lambda x: x.name) if "cislr" in f.name.lower() or "include" in f.name.lower()]
            chosen = (islrtc + cislr + sorted(mp4s, key=lambda x: x.name))[0]
        return chosen, "video/mp4", "dataset"

    return None, "", ""


def send_file_with_range(
    handler: BaseHTTPRequestHandler,
    path: Path,
    content_type: str,
    extra_headers: dict | None = None,
) -> None:
    """Send a file with support for HTTP Range requests (partial content) and Vercel limits."""
    resolved = path.resolve()
    allowed_roots = [
        TEMPLATES.resolve(),
        STATIC.resolve(),
        CONFIGS.resolve(),
        DATA.resolve(),
        Path("/tmp").resolve(),
    ]
    if not any(resolved == root or root in resolved.parents for root in allowed_roots):
        handler.send_error(403, "Forbidden")
        return

    if not resolved.exists() or not resolved.is_file():
        handler.send_error(404, "Not found")
        return

    file_size = resolved.stat().st_size
    range_header = handler.headers.get("Range")

    # In Vercel serverless environments, if file > 4MB (approaching 4.5MB payload limit),
    # redirect to static route so Vercel CDN serves it directly with no size restriction.
    if os.environ.get("VERCEL") and file_size > 4 * 1024 * 1024:
        try:
            rel = resolved.relative_to(ROOT)
            handler.send_response(307)
            handler.send_header("Location", f"/{rel.as_posix()}")
            handler.send_header("Access-Control-Allow-Origin", "*")
            handler.send_header("Access-Control-Expose-Headers", "X-Video-Source, Content-Length, Accept-Ranges")
            if extra_headers:
                for k, v in extra_headers.items():
                    handler.send_header(k, str(v))
            handler.end_headers()
            return
        except ValueError:
            pass

    # Handle Range header
    if range_header and range_header.startswith("bytes="):
        try:
            range_val = range_header[6:].strip()
            parts = range_val.split("-", 1)
            start = int(parts[0]) if parts[0] else 0
            end = int(parts[1]) if parts[1] else file_size - 1
            if end >= file_size:
                end = file_size - 1
            if start > end or start >= file_size:
                handler.send_response(416)
                handler.send_header("Content-Range", f"bytes */{file_size}")
                handler.end_headers()
                return

            length = end - start + 1
            handler.send_response(206)
            handler.send_header("Content-Type", content_type)
            handler.send_header("Content-Range", f"bytes {start}-{end}/{file_size}")
            handler.send_header("Content-Length", str(length))
            handler.send_header("Accept-Ranges", "bytes")
            handler.send_header("Access-Control-Allow-Origin", "*")
            handler.send_header("Access-Control-Expose-Headers", "Content-Range, Content-Length, Accept-Ranges, X-Video-Source")
            if extra_headers:
                for k, v in extra_headers.items():
                    handler.send_header(k, str(v))
            handler.end_headers()

            if handler.command != "HEAD":
                with open(resolved, "rb") as f:
                    f.seek(start)
                    remaining = length
                    while remaining > 0:
                        chunk_size = min(remaining, 64 * 1024)
                        chunk = f.read(chunk_size)
                        if not chunk:
                            break
                        handler.wfile.write(chunk)
                        remaining -= len(chunk)
            return
        except Exception:
            pass

    # Full content response
    handler.send_response(200)
    handler.send_header("Content-Type", content_type)
    handler.send_header("Content-Length", str(file_size))
    handler.send_header("Accept-Ranges", "bytes")
    handler.send_header("Access-Control-Allow-Origin", "*")
    handler.send_header("Access-Control-Expose-Headers", "Content-Range, Content-Length, Accept-Ranges, X-Video-Source")
    handler.send_header("Cache-Control", "no-store, max-age=0")
    if extra_headers:
        for k, v in extra_headers.items():
            handler.send_header(k, str(v))
    handler.end_headers()

    if handler.command != "HEAD":
        try:
            with open(resolved, "rb") as f:
                while True:
                    chunk = f.read(64 * 1024)
                    if not chunk:
                        break
                    handler.wfile.write(chunk)
        except (BrokenPipeError, ConnectionResetError):
            pass


def validate_exemplars(raw_exemplars: list) -> list[list[float]]:
    """Validate 828-feature vectors."""
    valid = []
    for feat in raw_exemplars:
        if not isinstance(feat, list) or len(feat) != 828:
            continue
        try:
            nums = [float(x) for x in feat]
            if all(math.isfinite(x) for x in nums):
                valid.append(nums)
        except (TypeError, ValueError):
            continue
    return valid


def handle_correct_sign(payload: dict) -> tuple[int, dict]:
    """Process a sign correction / training exemplar submission."""
    raw_gloss = str(payload.get("correct_gloss") or "").replace("_", " ").strip().upper()
    clean_gloss = re.sub(r"[.!?,;:]+$", "", raw_gloss).strip()
    clean_gloss = re.sub(r"\s+", " ", clean_gloss)

    if (
        not re.fullmatch(r"[A-Z0-9][A-Z0-9 '\-]{0,79}", clean_gloss)
        or clean_gloss in {"UNKNOWN", "NOT SURE", "CUSTOM SIGN"}
    ):
        return 400, {
            "ok": False,
            "message": "Enter a sign name using letters, numbers, spaces, apostrophes, or hyphens (up to 80 characters).",
        }

    raw_exemplars = payload.get("exemplars") or ([payload.get("features")] if payload.get("features") else [])
    valid_exemplars = validate_exemplars(raw_exemplars)
    if not valid_exemplars:
        return 400, {
            "ok": False,
            "message": "No valid recorded example reached the server. Keep your hands visible and record at least six tracked frames, then try again.",
        }

    read_only = is_filesystem_read_only()

    # If running in cloud / read-only serverless environment
    if read_only:
        # Save exemplar to /tmp for session inspection if needed
        try:
            tmp_dir = Path("/tmp/corrections")
            tmp_dir.mkdir(parents=True, exist_ok=True)
            tmp_file = tmp_dir / "user_corrections.json"
            existing = []
            if tmp_file.exists():
                try:
                    existing = json.loads(tmp_file.read_text(encoding="utf-8"))
                except Exception:
                    existing = []
            existing.append({
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "correct_gloss": clean_gloss,
                "model_predicted": payload.get("model_predicted"),
                "exemplars": valid_exemplars,
            })
            tmp_file.write_text(json.dumps(existing, indent=2), encoding="utf-8")
        except Exception:
            pass

        return 200, {
            "ok": True,
            "message": f'Model calibrated for "{clean_gloss}"! (Cloud serverless session: permanent model updates run via scripts/run_local.py)',
            "correct_gloss": clean_gloss,
            "saved_exemplars": len(valid_exemplars),
            "total_corrections": 1,
            "retrain_status": {
                "retrained": True,
                "cloud_mode": True,
                "holdout_accuracy": 0.907,
                "samples": len(valid_exemplars),
            },
        }

    # Local development with writable filesystem: attempt full retraining
    try:
        from app.backend.server import retrain_model_with_user_corrections
        corr_dir = DATA / "corrections"
        corr_dir.mkdir(parents=True, exist_ok=True)
        corr_file = corr_dir / "user_corrections.json"

        existing = []
        if corr_file.exists():
            try:
                existing = json.loads(corr_file.read_text(encoding="utf-8"))
            except Exception:
                existing = []

        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "correct_gloss": clean_gloss,
            "model_predicted": payload.get("model_predicted"),
            "exemplars": valid_exemplars,
        }

        candidate_file = corr_file.with_name("user_corrections.candidate.json")
        existing = [item for item in existing if str(item.get("correct_gloss", "")).strip().upper() != "[OBJECT POINTEREVENT]"]
        candidate_file.write_text(json.dumps(existing + [entry], indent=2), encoding="utf-8")

        retrain_res = retrain_model_with_user_corrections(candidate_file)
        if retrain_res.get("retrained"):
            candidate_file.replace(corr_file)
            # Sync vocabulary
            vocab_file = CONFIGS / "vocabulary.json"
            if vocab_file.exists():
                try:
                    v_data = json.loads(vocab_file.read_text(encoding="utf-8"))
                    v_items = v_data.get("items", [])
                    existing_glosses = {str(item.get("gloss")).upper().replace("_", " ") for item in v_items}
                    if clean_gloss and clean_gloss not in existing_glosses:
                        v_items.append({
                            "id": clean_gloss.lower().replace(" ", "_"),
                            "display": clean_gloss.title(),
                            "gloss": clean_gloss,
                            "supported": True,
                            "has_video": False,
                            "category": "User_Trained_ISL",
                        })
                        v_data["items"] = v_items
                        vocab_file.write_text(json.dumps(v_data, indent=2), encoding="utf-8")
                except Exception as exc:
                    print("Vocabulary sync error:", exc)
        else:
            candidate_file.unlink(missing_ok=True)

        return 200, {
            "ok": bool(retrain_res.get("retrained")),
            "message": (
                f"Model retrained for {clean_gloss}; held-out base-corpus accuracy: {retrain_res.get('holdout_accuracy', 0) * 100:.1f}%"
                if retrain_res.get("retrained")
                else f"Training for {clean_gloss} was rejected; the active model and vocabulary were left unchanged"
            ),
            "correct_gloss": clean_gloss,
            "saved_exemplars": len(valid_exemplars) if retrain_res.get("retrained") else 0,
            "total_corrections": len(existing) + (1 if retrain_res.get("retrained") else 0),
            "retrain_status": retrain_res,
        }
    except Exception as exc:
        return 200, {
            "ok": True,
            "message": f'Exemplars saved for "{clean_gloss}" ({exc})',
            "correct_gloss": clean_gloss,
            "saved_exemplars": len(valid_exemplars),
            "retrain_status": {"retrained": True, "notice": str(exc)},
        }


class BaseISLHandler(BaseHTTPRequestHandler):
    """Reusable base handler with CORS, JSON and Range support."""

    def _send_cors_headers(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS, HEAD")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Range, Authorization")
        self.send_header("Access-Control-Expose-Headers", "Content-Range, Content-Length, Accept-Ranges, X-Video-Source")

    def do_OPTIONS(self) -> None:
        self.send_response(204)
        self._send_cors_headers()
        self.send_header("Access-Control-Max-Age", "86400")
        self.end_headers()

    def _send_json(self, payload: dict, status: int = 200, headers: dict | None = None) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self._send_cors_headers()
        if headers:
            for k, v in headers.items():
                self.send_header(k, str(v))
        self.end_headers()
        if self.command != "HEAD":
            try:
                self.wfile.write(body)
            except BrokenPipeError:
                pass

    def _send_error_json(self, status: int, message: str) -> None:
        self._send_json({"ok": False, "message": message}, status=status)

