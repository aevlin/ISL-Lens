from __future__ import annotations

import sys
from pathlib import Path
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from api.common import BaseISLHandler, find_video_for_gloss, send_file_with_range


class handler(BaseISLHandler):
    def do_GET(self) -> None:
        query = parse_qs(urlparse(self.path).query)
        raw_gloss = query.get("gloss", [""])[0]
        if not raw_gloss:
            self._send_error_json(400, "A sign gloss query parameter is required (e.g. ?gloss=water)")
            return

        video_path, content_type, source = find_video_for_gloss(raw_gloss)
        if not video_path:
            self.send_error(404, f"No reference video available for '{raw_gloss}'")
            return

        extra_headers = {"X-Video-Source": source}
        send_file_with_range(self, video_path, content_type, extra_headers=extra_headers)

    def do_HEAD(self) -> None:
        self.do_GET()

