from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from api.common import BaseISLHandler


class handler(BaseISLHandler):
    def do_POST(self) -> None:
        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length).decode("utf-8") or "{}")
        except (ValueError, json.JSONDecodeError):
            self._send_error_json(400, "Invalid JSON payload")
            return

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

