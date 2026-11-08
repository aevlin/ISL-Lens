from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from api.common import BaseISLHandler, get_vocabulary_payload


class handler(BaseISLHandler):
    def do_GET(self) -> None:
        payload = get_vocabulary_payload()
        self._send_json(
            payload,
            status=200,
            headers={
                "Cache-Control": "public, max-age=3600, stale-while-revalidate=86400",
            },
        )

    def do_HEAD(self) -> None:
        self.do_GET()

