from __future__ import annotations

import hashlib
import time
from typing import Any, Optional


class RoutingCache:
    def __init__(self, ttl: int = 300):
        self._store: dict[str, tuple[Any, float]] = {}
        self._ttl = ttl

    def get(self, query: str) -> Optional[Any]:
        key = hashlib.sha256(query.encode()).hexdigest()
        entry = self._store.get(key)
        if entry and (time.time() - entry[1]) < self._ttl:
            return entry[0]
        return None

    def set(self, query: str, intent: Any) -> None:
        key = hashlib.sha256(query.encode()).hexdigest()
        self._store[key] = (intent, time.time())