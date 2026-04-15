"""
ContextIsolation — per-request context isolation to prevent state leakage between runs.

Provides:
    • A scoped context manager that clears state after each request
    • Thread-local storage so concurrent requests never share mutable state
    • Explicit binding/unbinding of secrets, credentials, or user-supplied data
    • A read-only snapshot view of the active context

Usage
-----
    isolation = ContextIsolation()

    with isolation.request_scope(user_id="u-123", role="analyst") as ctx:
        ctx["query"] = sanitized_query
        do_analysis(ctx["query"])
        # ctx is wiped on __exit__

    # Or in a multi-threaded scenario
    #   Each thread gets its own isolated namespace automatically.
"""

from __future__ import annotations

import copy
import threading
import time
import uuid
from contextlib import contextmanager
from typing import Any, Dict, Generator, Optional


class ContextIsolation:
    """
    Thread-safe, per-request context manager.

    Each ``request_scope`` call creates a fresh, isolated namespace.
    The namespace is attached to a thread-local so concurrent calls in
    different threads can never read each other's data.  The namespace is
    wiped on exit regardless of exceptions.
    """

    def __init__(self) -> None:
        self._local = threading.local()

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    @contextmanager
    def request_scope(
        self,
        *,
        user_id: Optional[str] = None,
        role: Optional[str] = None,
        request_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Generator["_IsolatedContext", None, None]:
        """
        Return an isolated mutable context for one request.

        Parameters
        ----------
        user_id:
            Identifier of the caller (used for audit).
        role:
            RBAC role of the caller.
        request_id:
            Caller-supplied correlation ID; auto-generated if omitted.
        metadata:
            Optional extra metadata embedded in the context at creation.
        """
        rid = request_id or str(uuid.uuid4())
        ctx = _IsolatedContext(
            request_id=rid,
            user_id=user_id,
            role=role,
            created_at=time.monotonic(),
        )
        if metadata:
            for k, v in metadata.items():
                ctx[k] = v

        # Attach to this thread
        previous = getattr(self._local, "active_context", None)
        self._local.active_context = ctx
        try:
            yield ctx
        finally:
            # Explicit wipe — clear all mutable state
            ctx._data.clear()
            ctx._sealed = True
            # Restore previous context (if nested, which is unusual)
            self._local.active_context = previous

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @property
    def current_context(self) -> Optional["_IsolatedContext"]:
        """Return the active context for the calling thread, or None."""
        return getattr(self._local, "active_context", None)

    def snapshot(self) -> Dict[str, Any]:
        """Return a deep-copy snapshot of the current context (safe to share)."""
        ctx = self.current_context
        if ctx is None:
            return {}
        return copy.deepcopy({"request_id": ctx.request_id, "user_id": ctx.user_id, "role": ctx.role, **ctx._data})


class _IsolatedContext:
    """
    A mutable dict-like namespace bound to a single request.

    Attributes are stored in a plain dict so they are wiped on scope exit.
    After sealing, all writes raise ``RuntimeError``.
    """

    __slots__ = ("request_id", "user_id", "role", "created_at", "_data", "_sealed")

    def __init__(
        self,
        *,
        request_id: str,
        user_id: Optional[str],
        role: Optional[str],
        created_at: float,
    ) -> None:
        self.request_id = request_id
        self.user_id = user_id
        self.role = role
        self.created_at = created_at
        self._data: Dict[str, Any] = {}
        self._sealed = False

    # dict-like interface
    def __setitem__(self, key: str, value: Any) -> None:
        if self._sealed:
            raise RuntimeError("Context is sealed — no writes after scope exit")
        self._data[key] = value

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default)

    def __contains__(self, key: str) -> bool:
        return key in self._data

    def __repr__(self) -> str:
        return (
            f"_IsolatedContext(request_id={self.request_id!r}, "
            f"user_id={self.user_id!r}, role={self.role!r}, "
            f"keys={list(self._data.keys())})"
        )
