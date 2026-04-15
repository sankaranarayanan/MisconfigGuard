"""
Observability — structured audit logging and security event tracking.

Provides:
    AuditEvent   — immutable record of a security-relevant event
    AuditLogger  — thread-safe logger with in-memory buffer + file sink

Security events captured:
    • access_denied    — RBAC check failure
    • input_rejected   — InputSanitizer rejection
    • rag_poisoning    — RAGPoisonDefense detected suspicious chunk
    • llm_guardrail    — LLM guardrail triggered
    • approval_request — Human-in-the-loop approval requested
    • approval_decision — Human approved / rejected a finding
    • scan_start / scan_complete — lifecycle events
    • tool_invoked     — which tool was called, by whom

Usage
-----
    logger = AuditLogger()
    logger.log(AuditEvent(event_type="access_denied", actor="viewer",
                          resource="iam_analyzer", detail="lacks RUN_SCAN"))

    # Retrieve buffered events
    events = logger.query(event_type="access_denied")

    # Persist to file
    logger.flush("./logs/audit.jsonl")
"""

from __future__ import annotations

import json
import logging
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


_stdlib_logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AuditEvent:
    """
    Immutable audit record.

    Attributes
    ----------
    event_type:   Category / type of security event.
    actor:        User ID or role triggering the event.
    resource:     Resource or tool involved.
    detail:       Human-readable explanation string.
    severity:     One of DEBUG / INFO / WARNING / ERROR / CRITICAL.
    metadata:     Optional structured context (request_id, file path, …).
    event_id:     Auto-generated unique identifier.
    timestamp:    Unix epoch (seconds, float).
    """

    event_type: str
    actor: str = "unknown"
    resource: str = ""
    detail: str = ""
    severity: str = "INFO"
    metadata: Dict[str, Any] = field(default_factory=dict, compare=False, hash=False)
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()), compare=False, hash=False)
    timestamp: float = field(default_factory=time.time, compare=False, hash=False)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict())


class AuditLogger:
    """
    Thread-safe structured audit event logger.

    Parameters
    ----------
    max_buffer_size:
        Maximum number of events kept in the in-memory ring buffer.
        Oldest events are evicted when the buffer is full.
    log_to_stdlib:
        If True (default), also emit events through Python's standard
        logging framework at the event's severity level.
    """

    def __init__(
        self,
        max_buffer_size: int = 10_000,
        log_to_stdlib: bool = True,
    ) -> None:
        self._buffer: List[AuditEvent] = []
        self._max = max_buffer_size
        self._lock = threading.Lock()
        self._log_to_stdlib = log_to_stdlib

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def log(self, event: AuditEvent) -> None:
        """Append *event* to the buffer (and emit to stdlib if configured)."""
        with self._lock:
            if len(self._buffer) >= self._max:
                self._buffer.pop(0)   # evict oldest
            self._buffer.append(event)

        if self._log_to_stdlib:
            level = getattr(logging, event.severity.upper(), logging.INFO)
            _stdlib_logger.log(
                level,
                "[AUDIT] %s | actor=%s | resource=%s | %s",
                event.event_type,
                event.actor,
                event.resource,
                event.detail,
            )

    # ------------------------------------------------------------------
    # Convenience factories
    # ------------------------------------------------------------------

    def access_denied(self, *, actor: str, resource: str, reason: str, request_id: Optional[str] = None) -> None:
        self.log(AuditEvent(
            event_type="access_denied",
            actor=actor,
            resource=resource,
            detail=reason,
            severity="WARNING",
            metadata={"request_id": request_id} if request_id else {},
        ))

    def input_rejected(self, *, actor: str, detail: str, request_id: Optional[str] = None) -> None:
        self.log(AuditEvent(
            event_type="input_rejected",
            actor=actor,
            resource="input_sanitizer",
            detail=detail,
            severity="WARNING",
            metadata={"request_id": request_id} if request_id else {},
        ))

    def rag_poisoning(self, *, actor: str, chunk_id: str, flags: List[str], request_id: Optional[str] = None) -> None:
        self.log(AuditEvent(
            event_type="rag_poisoning",
            actor=actor,
            resource=chunk_id,
            detail=f"Poisoning flags: {flags}",
            severity="ERROR",
            metadata={"flags": flags, "request_id": request_id},
        ))

    def llm_guardrail(self, *, actor: str, field_name: str, reason: str, request_id: Optional[str] = None) -> None:
        self.log(AuditEvent(
            event_type="llm_guardrail",
            actor=actor,
            resource=field_name,
            detail=reason,
            severity="WARNING",
            metadata={"request_id": request_id} if request_id else {},
        ))

    def tool_invoked(self, *, actor: str, tool: str, request_id: Optional[str] = None) -> None:
        self.log(AuditEvent(
            event_type="tool_invoked",
            actor=actor,
            resource=tool,
            detail=f"Tool '{tool}' invoked by '{actor}'",
            severity="INFO",
            metadata={"request_id": request_id} if request_id else {},
        ))

    def scan_start(self, *, actor: str, target: str, request_id: Optional[str] = None) -> None:
        self.log(AuditEvent(
            event_type="scan_start",
            actor=actor,
            resource=target,
            detail=f"Scan started on '{target}'",
            severity="INFO",
            metadata={"request_id": request_id} if request_id else {},
        ))

    def scan_complete(self, *, actor: str, target: str, issue_count: int, request_id: Optional[str] = None) -> None:
        self.log(AuditEvent(
            event_type="scan_complete",
            actor=actor,
            resource=target,
            detail=f"Scan completed — {issue_count} issue(s) found",
            severity="INFO",
            metadata={"issue_count": issue_count, "request_id": request_id},
        ))

    def approval_requested(self, *, actor: str, approval_id: str, finding_title: str, request_id: Optional[str] = None) -> None:
        self.log(AuditEvent(
            event_type="approval_request",
            actor=actor,
            resource=approval_id,
            detail=f"Human approval requested for: {finding_title}",
            severity="WARNING",
            metadata={"approval_id": approval_id, "request_id": request_id},
        ))

    def approval_decision(self, *, actor: str, approval_id: str, decision: str, request_id: Optional[str] = None) -> None:
        self.log(AuditEvent(
            event_type="approval_decision",
            actor=actor,
            resource=approval_id,
            detail=f"Decision '{decision}' for approval {approval_id}",
            severity="INFO",
            metadata={"decision": decision, "approval_id": approval_id, "request_id": request_id},
        ))

    # ------------------------------------------------------------------
    # Query / retrieval
    # ------------------------------------------------------------------

    def query(
        self,
        *,
        event_type: Optional[str] = None,
        actor: Optional[str] = None,
        severity: Optional[str] = None,
        since: Optional[float] = None,
    ) -> List[AuditEvent]:
        """
        Return a filtered snapshot of buffered events.
        All parameters are optional; omit to return all buffered events.
        """
        with self._lock:
            events = list(self._buffer)

        if event_type:
            events = [e for e in events if e.event_type == event_type]
        if actor:
            events = [e for e in events if e.actor == actor]
        if severity:
            events = [e for e in events if e.severity.upper() == severity.upper()]
        if since is not None:
            events = [e for e in events if e.timestamp >= since]

        return events

    def flush(self, path: str) -> int:
        """
        Append all buffered events to a JSONL file at *path*.

        Returns the number of events written.
        """
        dest = Path(path)
        dest.parent.mkdir(parents=True, exist_ok=True)

        with self._lock:
            events = list(self._buffer)

        with dest.open("a", encoding="utf-8") as fh:
            for event in events:
                fh.write(event.to_json() + "\n")

        return len(events)

    def clear(self) -> None:
        """Clear the in-memory buffer."""
        with self._lock:
            self._buffer.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._buffer)
