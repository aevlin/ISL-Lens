from __future__ import annotations

import sys
from pathlib import Path
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from api.common import DATA, BaseISLHandler, is_filesystem_read_only


class handler(BaseISLHandler):
    def do_POST(self) -> None:
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
            self._send_json({
                "ok": True,
                "saved": dest.name,
                "gloss": gloss_clean,
                "size": len(video_data),
            })
        except Exception as e:
            self._send_error_json(500, f"Upload error: {e}")

