from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from urllib.parse import urlparse, unquote

import numpy as np



ROOT = Path(__file__).resolve().parents[2]
TEMPLATES = ROOT / "app" / "templates"
STATIC = ROOT / "app" / "static"
CONFIGS = ROOT / "configs"


def retrain_model_with_user_corrections(corr_file: Path) -> dict:
    feature_cache = ROOT / "data" / "processed" / "isl40_landmark_features.npz"
    if not feature_cache.exists():
        return {"retrained": False, "reason": "Baseline features not found"}

    try:
        cache = np.load(feature_cache, allow_pickle=True)
        X_base = cache["X"].astype(np.float32)
        y_base = cache["y"].astype(str)

        user_entries = []
        if corr_file.exists():
            try:
                user_entries = json.loads(corr_file.read_text(encoding="utf-8") or "[]")
            except Exception:
                user_entries = []

        user_X = []
        user_y = []
        for entry in user_entries:
            gloss = str(entry.get("correct_gloss") or "").upper().strip()
            if (not re.fullmatch(r"[A-Z0-9][A-Z0-9 '\-]{0,79}", gloss)
                    or gloss in {"UNKNOWN", "NOT SURE", "CUSTOM SIGN"}):
                continue
            exemplars = entry.get("exemplars") or ([entry.get("features")] if entry.get("features") else [])
            for feat in exemplars:
                if not isinstance(feat, list) or len(feat) != 828:
                    continue
                try:
                    vector = np.asarray(feat, dtype=np.float32)
                except (TypeError, ValueError):
                    continue
                if not np.isfinite(vector).all():
                    continue
                for _ in range(8):  # Duplicate 8x for strong sample weighting
                    user_X.append(vector)
                    user_y.append(gloss)

        from collections import Counter
        from sklearn.metrics import accuracy_score
        from sklearn.model_selection import StratifiedShuffleSplit
        from sklearn.preprocessing import LabelEncoder, StandardScaler
        from sklearn.linear_model import LogisticRegression

        # Keep singleton labels in training. Only labels with at least two
        # independent corpus samples can be represented in a stratified holdout.
        counts = Counter(y_base)
        eligible = np.flatnonzero(np.array([counts[label] >= 2 for label in y_base]))
        singletons = np.flatnonzero(np.array([counts[label] < 2 for label in y_base]))
        splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        eligible_train, test_idx = next(splitter.split(X_base[eligible], y_base[eligible]))
        train_idx = np.concatenate([eligible[eligible_train], singletons])

        def fit_model(X_fit: np.ndarray, y_fit: np.ndarray):
            label_encoder = LabelEncoder()
            y_encoded = label_encoder.fit_transform(y_fit)
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_fit)
            classifier = LogisticRegression(
                max_iter=2500, class_weight="balanced", solver="lbfgs", C=1.0, random_state=42
            )
            classifier.fit(X_scaled, y_encoded)
            return label_encoder, scaler, classifier

        fit_parts_x = [X_base[train_idx]]
        fit_parts_y = [y_base[train_idx]]
        if user_X:
            user_x = np.asarray(user_X, dtype=np.float32)
            user_labels = np.asarray(user_y, dtype=str)
            fit_parts_x.extend([user_x] * 8)
            fit_parts_y.extend([user_labels] * 8)
        candidate_encoder, candidate_scaler, candidate_clf = fit_model(
            np.vstack(fit_parts_x), np.concatenate(fit_parts_y)
        )
        test_scaled = candidate_scaler.transform(X_base[test_idx])
        holdout_accuracy = float(accuracy_score(
            candidate_encoder.transform(y_base[test_idx]), candidate_clf.predict(test_scaled)
        ))
        if holdout_accuracy < 0.90:
            return {
                "retrained": False,
                "reason": "Candidate model did not meet the 90% held-out corpus accuracy gate",
                "holdout_accuracy": holdout_accuracy,
            }

        # After passing the untouched corpus holdout, refit on all available
        # labeled samples for the production model.
        full_parts_x = [X_base]
        full_parts_y = [y_base]
        if user_X:
            user_x = np.asarray(user_X, dtype=np.float32)
            user_labels = np.asarray(user_y, dtype=str)
            full_parts_x.extend([user_x] * 8)
            full_parts_y.extend([user_labels] * 8)
        le, scaler, clf = fit_model(np.vstack(full_parts_x), np.concatenate(full_parts_y))

        payload = {
            "version": "v1.2.0-validated-retrain",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "model_type": "multinomial_logistic_regression",
            "sequence_length": 32,
            "input_features": 828,
            "classes": le.classes_.tolist(),
            "unknown_threshold": 0.35,
            "scaler_mean": scaler.mean_.astype(float).tolist(),
            "scaler_scale": scaler.scale_.astype(float).tolist(),
            "coef": clf.coef_.astype(float).tolist(),
            "intercept": clf.intercept_.astype(float).tolist(),
            "metrics": {
                "total_samples": int(len(y_base) + len(user_y) * 8),
                "user_exemplars": int(len(user_y)),
                "holdout_samples": int(len(test_idx)),
                "holdout_accuracy": holdout_accuracy,
                "accuracy_percentage": f"{holdout_accuracy * 100:.1f}%",
                "status": "PASSED_90_PERCENT_BASE_CORPUS_HOLDOUT",
            },
        }

        model_out = ROOT / "app" / "static" / "models" / "landmark_logreg.json"
        model_out.write_text(json.dumps(payload), encoding="utf-8")
        return {
            "retrained": True,
            "samples": int(len(y_base) + len(user_y) * 8),
            "user_exemplars": int(len(user_y)),
            "holdout_accuracy": holdout_accuracy,
        }
    except Exception as exc:
        print("Model retraining error:", exc)
        return {"retrained": False, "error": str(exc)}


class ISLLensHandler(SimpleHTTPRequestHandler):
    server_version = "ISLLensLocal/0.1"

    def do_GET(self) -> None:
        path = unquote(urlparse(self.path).path)

        if path == "/":
            self._send_file(TEMPLATES / "index.html", "text/html; charset=utf-8")
            return

        if path == "/api/health":
            model_file = STATIC / "models" / "landmark_logreg.json"
            has_model = model_file.exists()
            model_info = {}
            if has_model:
                try:
                    model_info = json.loads(model_file.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    has_model = False
            self._send_json(
                {
                    "ok": True,
                    "service": "ISL-Lens local app",
                    "model_status": "trained" if has_model else "not_trained",
                    "model_file": "/static/models/landmark_logreg.json" if has_model else None,
                    "recognition_policy": "spatial_sequence_with_unknown_rejection" if has_model else "unknown_first",
                    "model_version": model_info.get("version"),
                    "trained_sign_count": len(model_info.get("classes", [])),
                    "held_out_accuracy": model_info.get("metrics", {}).get("accuracy_percentage"),
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
            )
            return

        if path == "/api/vocabulary":
            self._send_file(CONFIGS / "vocabulary.json", "application/json; charset=utf-8")
            return

        if path == "/api/dataset_video":
            from urllib.parse import parse_qs
            query = parse_qs(urlparse(self.path).query)
            raw_gloss = query.get("gloss", [""])[0].lower().strip()
            if not raw_gloss:
                self.send_error(400, "A sign gloss is required for a reference video")
                return
            gloss_clean = raw_gloss.replace(" ", "_").replace("?", "")

            # Priority 1: User-recorded custom video for this sign if available
            user_videos_dir = ROOT / "data" / "user_videos"
            for ext in [".webm", ".mp4"]:
                u_vid = user_videos_dir / f"{gloss_clean}{ext}"
                if u_vid.exists() and u_vid.stat().st_size > 500:
                    c_type = "video/webm" if ext == ".webm" else "video/mp4"
                    self._send_file(u_vid, c_type, extra_headers={"X-Video-Source": "user"})
                    return

            video_dir = ROOT / "data" / "raw" / "isl_isolated_40words"

            # Curated verified reference videos for authentic ISL execution
            VERIFIED_REFERENCE_OVERRIDES = {
                "thank_you": "thank_you__ISL500__00371__Thankyou__session119__clip002.mp4",
                "thankyou": "thank_you__ISL500__00371__Thankyou__session119__clip002.mp4",
                "sorry": "sorry__ISLRTC_dictionary__00637__Sorry_(Sign_2).mp4",
                "please": "please__CISLR__00073__vAW9QDnXLrE.mp4",
            }
            
            g_dir = video_dir / gloss_clean
            mp4s = []
            if g_dir.exists() and g_dir.is_dir():
                mp4s = list(g_dir.glob("*.mp4"))
            if not mp4s:
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
                self._send_file(chosen, "video/mp4", extra_headers={"X-Video-Source": "dataset"})
            else:
                self.send_error(404, f"No verified reference video is available for '{gloss_clean}'")
            return

        if path.startswith("/static/"):
            requested = STATIC / path.removeprefix("/static/")
            self._send_file(requested, self._content_type(requested))
            return

        super().do_GET()

    def do_HEAD(self) -> None:
        self.do_GET()

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        if path == "/api/upload_user_video":
            from urllib.parse import parse_qs
            query = parse_qs(urlparse(self.path).query)
            raw_gloss = query.get("gloss", [""])[0].lower().strip()
            if not raw_gloss:
                self._send_error_json(400, "A sign gloss is required")
                return
            gloss_clean = raw_gloss.replace(" ", "_").replace("?", "")
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length <= 0:
                    self._send_error_json(400, "Empty video payload")
                    return
                video_data = self.rfile.read(length)
                user_videos_dir = ROOT / "data" / "user_videos"
                user_videos_dir.mkdir(parents=True, exist_ok=True)
                dest = user_videos_dir / f"{gloss_clean}.webm"
                dest.write_bytes(video_data)
                self._send_json({"ok": True, "saved": dest.name, "gloss": gloss_clean, "size": len(video_data)})
            except Exception as e:
                self._send_error_json(500, f"Upload error: {e}")
            return

        if path == "/api/correct_sign":
            try:
                length = int(self.headers.get("Content-Length", "0"))
                payload = json.loads(self.rfile.read(length).decode("utf-8") or "{}")
            except (ValueError, json.JSONDecodeError):
                self.send_error(400, "Invalid JSON")
                return

            corr_dir = ROOT / "data" / "corrections"
            corr_dir.mkdir(parents=True, exist_ok=True)
            corr_file = corr_dir / "user_corrections.json"

            existing = []
            if corr_file.exists():
                try:
                    existing = json.loads(corr_file.read_text(encoding="utf-8"))
                except Exception:
                    existing = []

            exemplars = payload.get("exemplars") or ([payload.get("features")] if payload.get("features") else [])

            clean_gloss = str(payload.get("correct_gloss") or "").replace("_", " ").strip().upper()
            # Strip sentence-ending punctuation users commonly type (e.g. "HOW ARE YOU?").
            clean_gloss = re.sub(r"[.!?,;:]+$", "", clean_gloss).strip()
            clean_gloss = re.sub(r"\s+", " ", clean_gloss)
            if (not re.fullmatch(r"[A-Z0-9][A-Z0-9 '\-]{0,79}", clean_gloss)
                    or clean_gloss in {"UNKNOWN", "NOT SURE", "CUSTOM SIGN"}):
                self._send_error_json(400, "Enter a sign name using letters, numbers, spaces, apostrophes, or hyphens (up to 80 characters).")
                return
            valid_exemplars = []
            for feat in exemplars:
                if not isinstance(feat, list) or len(feat) != 828:
                    continue
                try:
                    vector = np.asarray(feat, dtype=np.float32)
                except (TypeError, ValueError):
                    continue
                if np.isfinite(vector).all():
                    valid_exemplars.append(vector.astype(float).tolist())
            exemplars = valid_exemplars
            if not exemplars:
                self._send_error_json(400, "No valid recorded example reached the server. Keep your hands visible and record at least six tracked frames, then try again.")
                return

            entry = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "correct_gloss": clean_gloss,
                "model_predicted": payload.get("model_predicted"),
                "exemplars": exemplars,
            }
            # Build and validate against a candidate file. Failed retraining
            # must not pollute active corrections, model labels, or vocabulary.
            candidate_file = corr_file.with_name("user_corrections.candidate.json")
            existing = [item for item in existing
                        if str(item.get("correct_gloss", "")).strip().upper() != "[OBJECT POINTEREVENT]"]
            candidate_file.write_text(json.dumps(existing + [entry], indent=2), encoding="utf-8")
            retrain_res = retrain_model_with_user_corrections(candidate_file)
            if retrain_res.get("retrained"):
                candidate_file.replace(corr_file)
            else:
                candidate_file.unlink(missing_ok=True)

            # Automatically update vocabulary.json with newly trained sign/sentence
            vocab_file = CONFIGS / "vocabulary.json"
            if retrain_res.get("retrained") and vocab_file.exists():
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
                            "category": "User_Trained_ISL"
                        })
                        v_data["items"] = v_items
                        vocab_file.write_text(json.dumps(v_data, indent=2), encoding="utf-8")
                except Exception as exc:
                    print("Vocabulary sync error:", exc)

            self._send_json({
                "ok": bool(retrain_res.get("retrained")),
                "message": (
                    f"Model retrained for {clean_gloss}; held-out base-corpus accuracy: {retrain_res.get('holdout_accuracy', 0) * 100:.1f}%"
                    if retrain_res.get("retrained")
                    else f"Training for {clean_gloss} was rejected; the active model and vocabulary were left unchanged"
                ),
                "correct_gloss": payload.get("correct_gloss"),
                "saved_exemplars": len(exemplars) if retrain_res.get("retrained") else 0,
                "total_corrections": len(existing) + (1 if retrain_res.get("retrained") else 0),
                "retrain_status": retrain_res
            })
            return

        if path != "/api/export":
            self.send_error(404, "Not found")
            return

        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length).decode("utf-8") or "{}")
        except (ValueError, json.JSONDecodeError):
            self.send_error(400, "Invalid JSON")
            return

        export = {
            "exported_at": datetime.now(timezone.utc).isoformat(),
            "project": "ISL-Lens",
            "recognition_policy": "UNKNOWN until an audited trained model is configured",
            "conversation": payload.get("conversation", []),
            "practice": payload.get("practice", []),
        }
        body = json.dumps(export, indent=2).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Disposition", "attachment; filename=isl-lens-conversation.json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: object) -> None:
        print(f"[ISL-Lens] {self.address_string()} - {format % args}")

    def _send_json(self, payload: dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        if self.command != "HEAD":
            try:
                self.wfile.write(body)
            except BrokenPipeError:
                pass

    def _send_error_json(self, status: int, message: str) -> None:
        body = json.dumps({"ok": False, "message": message}).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _send_file(self, path: Path, content_type: str, extra_headers: dict = None) -> None:
        resolved = path.resolve()
        allowed_roots = [TEMPLATES.resolve(), STATIC.resolve(), CONFIGS.resolve(), (ROOT / "data").resolve()]
        if not any(resolved == root or root in resolved.parents for root in allowed_roots):
            self.send_error(403, "Forbidden")
            return
        if not resolved.exists() or not resolved.is_file():
            self.send_error(404, "Not found")
            return
        body = resolved.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        # Static app code is edited locally during development. Prevent the
        # browser from reusing a cached older classifier after a fix ships.
        self.send_header("Cache-Control", "no-store, max-age=0")
        self.send_header("Content-Length", str(len(body)))
        if extra_headers:
            for k, v in extra_headers.items():
                self.send_header(k, str(v))
        self.end_headers()
        if self.command != "HEAD":
            try:
                self.wfile.write(body)
            except BrokenPipeError:
                pass

    @staticmethod
    def _content_type(path: Path) -> str:
        suffix = path.suffix.lower()
        return {
            ".css": "text/css; charset=utf-8",
            ".js": "application/javascript; charset=utf-8",
            ".json": "application/json; charset=utf-8",
            ".html": "text/html; charset=utf-8",
            ".svg": "image/svg+xml",
        }.get(suffix, "application/octet-stream")


def run(host: str = "127.0.0.1", port: int = 8000) -> None:
    httpd = ThreadingHTTPServer((host, port), ISLLensHandler)
    print(f"ISL-Lens local app running at http://{host}:{port}")
    print("Recognition policy: returns UNKNOWN until an audited trained model is configured.")
    httpd.serve_forever()


if __name__ == "__main__":
    run()
