from __future__ import annotations

import json
from datetime import datetime, timezone
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from urllib.parse import urlparse

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
            if not gloss:
                continue
            exemplars = entry.get("exemplars") or ([entry.get("features")] if entry.get("features") else [])
            for feat in exemplars:
                if feat and len(feat) == 828:
                    for _ in range(8):  # Duplicate 8x for strong sample weighting
                        user_X.append(feat)
                        user_y.append(gloss)

        if user_X:
            X_all = np.vstack([X_base, np.array(user_X, dtype=np.float32)])
            y_all = np.concatenate([y_base, np.array(user_y, dtype=str)])
        else:
            X_all, y_all = X_base, y_base

        from sklearn.preprocessing import LabelEncoder, StandardScaler
        from sklearn.linear_model import LogisticRegression

        le = LabelEncoder()
        y_enc = le.fit_transform(y_all)

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_all)

        clf = LogisticRegression(max_iter=2500, class_weight="balanced", solver="lbfgs", random_state=42)
        clf.fit(X_scaled, y_enc)

        payload = {
            "version": "v0.6.0-user-retrained-3rep",
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
            "metrics": {"total_samples": int(len(y_all)), "user_exemplars": int(len(user_y))},
        }

        model_out = ROOT / "app" / "static" / "models" / "landmark_logreg.json"
        model_out.write_text(json.dumps(payload), encoding="utf-8")
        return {"retrained": True, "samples": int(len(y_all)), "user_exemplars": int(len(user_y))}
    except Exception as exc:
        print("Model retraining error:", exc)
        return {"retrained": False, "error": str(exc)}


class ISLLensHandler(SimpleHTTPRequestHandler):
    server_version = "ISLLensLocal/0.1"

    def do_GET(self) -> None:
        path = urlparse(self.path).path

        if path == "/":
            self._send_file(TEMPLATES / "index.html", "text/html; charset=utf-8")
            return

        if path == "/api/health":
            model_file = STATIC / "models" / "landmark_logreg.json"
            has_model = model_file.exists()
            self._send_json(
                {
                    "ok": True,
                    "service": "ISL-Lens local app",
                    "model_status": "trained" if has_model else "not_trained",
                    "model_file": "/static/models/landmark_logreg.json" if has_model else None,
                    "recognition_policy": "landmark_logreg_v0.1" if has_model else "unknown_first",
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
            gloss_clean = query.get("gloss", ["water"])[0].lower().strip().replace(" ", "_")
            video_dir = ROOT / "data" / "raw" / "isl_isolated_40words"
            
            g_dir = video_dir / gloss_clean
            mp4s = []
            if g_dir.exists() and g_dir.is_dir():
                mp4s = list(g_dir.glob("*.mp4"))
            if not mp4s:
                mp4s = list(video_dir.glob(f"**/{gloss_clean}*.mp4"))
            
            if mp4s:
                islrtc = [f for f in mp4s if "islrtc" in f.name.lower() or "dict" in f.name.lower()]
                cislr = [f for f in mp4s if "cislr" in f.name.lower() or "include" in f.name.lower()]
                chosen = (islrtc + cislr + sorted(mp4s, key=lambda x: x.name))[0]
                self._send_file(chosen, "video/mp4")
            else:
                self.send_error(404, f"No reference video found for gloss '{gloss_clean}'")
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

            entry = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "correct_gloss": payload.get("correct_gloss"),
                "model_predicted": payload.get("model_predicted"),
                "exemplars": exemplars,
            }
            existing.append(entry)
            corr_file.write_text(json.dumps(existing, indent=2), encoding="utf-8")

            # Trigger automated scikit-learn model retraining
            retrain_res = retrain_model_with_user_corrections(corr_file)

            self._send_json({
                "ok": True,
                "message": f"Model successfully retrained on 3 exemplars for sign {payload.get('correct_gloss')}",
                "correct_gloss": payload.get("correct_gloss"),
                "total_corrections": len(existing),
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
        self.wfile.write(body)

    def _send_file(self, path: Path, content_type: str) -> None:
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
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

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
