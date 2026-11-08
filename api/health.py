from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from api.common import BaseISLHandler, get_health_payload


class handler(BaseISLHandler):
    def do_GET(self) -> None:
        self._send_json(get_health_payload(), status=200)

    def do_HEAD(self) -> None:
        self.do_GET()

