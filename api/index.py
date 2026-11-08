from __future__ import annotations

import json
import sys
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from api.common import (
    DATA,
    BaseISLHandler,
    find_video_for_gloss,
    get_health_payload,
    get_vocabulary_payload,
    handle_correct_sign,
    is_filesystem_read_only,
    send_file_with_range,
)


class handler(BaseISLHandler):
    """Unified API entrypoint handling all endpoints."""

    def do_GET(self) -> None:
        path = unquote(urlparse(self.path).path).rstrip("/")
        # Normalize matching (e.g. /api/health or /api/health/)
        subpath = path.removeprefix("/api")

        if subpath == "/health" or path == "/health":
            self._send_json(get_health_payload())
            return

        if subpath == "/vocabulary" or path == "/vocabulary":
            self._send_json(
                get_vocabulary_payload(),
                headers={"Cache-Control": "public, max-age=3600, stale-while-revalidate=86400"},
            )
            return

        if subpath == "/dataset_video" or path == "/dataset_video":
            query = parse_qs(urlparse(self.path).query)
            raw_gloss = query.get("gloss", [""])[0]
            if not raw_gloss:
                self._send_error_json(400, "A sign gloss query parameter is required (e.g. ?gloss=water)")
                return
            video_path, content_type, source = find_video_for_gloss(raw_gloss)
            if not video_path:
                self.send_error(404, f"No reference video available for '{raw_gloss}'")
                return
            send_file_with_range(self, video_path, content_type, extra_headers={"X-Video-Source": source})
            return

        # Default API welcome/status
        self._send_json({
            "service": "ISL-Lens API",
            "status": "online",
            "endpoints": [
                "/api/health",
                "/api/vocabulary",
                "/api/dataset_video",
                "/api/correct_sign",
                "/api/upload_user_video",
                "/api/export",
            ],
        })

    def do_HEAD(self) -> None:
        self.do_GET()

    def do_POST(self) -> None:
        path = unquote(urlparse(self.path).path).rstrip("/")
        subpath = path.removeprefix("/api")

        if subpath == "/upload_user_video" or path == "/upload_user_video":
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
                target_dir = Path("/tmp/user_videos") if is_filesystem_read_only() else DATA / "user_videos"
                target_dir.mkdir(parents=True, exist_ok=True)
                dest = target_dir / f"{gloss_clean}.webm"
                dest.write_bytes(video_data)
                self._send_json({"ok": True, "saved": dest.name, "gloss": gloss_clean, "size": len(video_data)})
            except Exception as e:
                self._send_error_json(500, f"Upload error: {e}")
            return

        if subpath == "/correct_sign" or path == "/correct_sign":
            try:
                length = int(self.headers.get("Content-Length", "0"))
                payload = json.loads(self.rfile.read(length).decode("utf-8") or "{}")
            except (ValueError, json.JSONDecodeError):
                self._send_error_json(400, "Invalid JSON payload")
                return
            status, response = handle_correct_sign(payload)
            self._send_json(response, status=status)
            return

        if subpath == "/export" or path == "/export":
            try:
                length = int(self.headers.get("Content-Length", "0"))
                payload = json.loads(self.rfile.read(length).decode("utf-8") or "{}")
            except (ValueError, json.JSONDecodeError):
                self._send_error_json(400, "Invalid JSON payload")
                return
            from datetime import datetime, timezone
            export = {
                "exported_at": datetime.now(timezone.utc).isoformat(),
                "project": "ISL-Lens",
                "recognition_policy": "spatial_sequence_with_unknown_rejection",
                "conversation": payload.get("conversation", []),
                "practice": payload.get("practice", []),
            }
            body = json.dumps(export, indent=2).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Disposition", "attachment; filename=isl-lens-conversation.json")
            self.send_header("Content-Length", str(len(body)))
            self._send_cors_headers()
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(body)
            return

        self._send_error_json(404, f"API endpoint not found: {path}")

