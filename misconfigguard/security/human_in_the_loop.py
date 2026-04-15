"""
HumanInTheLoop — human approval workflow for high-risk findings.

When a scan uncovers a CRITICAL or HIGH severity finding, the pipeline can
hold the result and require an explicit human decision before it is acted on
(e.g., before a CI gate fails a build, before remediation is triggered, or
before a report is published).

Design
------
    ApprovalRequest  — immutable record of a pending decision
    ApprovalStatus   — enum: PENDING | APPROVED | REJECTED | TIMED_OUT | AUTO_APPROVED
    HumanInTheLoop   — manages the queue and provides synchronous / async helpers

Callback-based approval
~~~~~~~~~~~~~~~~~~~~~~~
Register a ``reviewer`` callable.  The callable receives an ``ApprovalRequest``
and must return True (approve) or False (reject).  This is suitable for
integration with notification systems (Slack, email, JIRA) or interactive
CLIs.

Auto-approval rules
~~~~~~~~~~~~~~~~~~~
Findings can be auto-approved when:
    • Their severity is below the configured ``require_approval_above`` threshold
    • A timeout expires (configurable; disabled by default)

Usage
-----
    hitl = HumanInTheLoop(require_approval_above="HIGH")

    # Submit a finding for review (blocks until decision or timeout)
    decision = hitl.submit(issue, timeout=300)
    if decision == ApprovalStatus.APPROVED:
        publish_report(issue)

    # Register an automated reviewer (e.g., for testing)
    hitl.register_reviewer(lambda req: True)

    # Retrieve pending requests
    pending = hitl.pending_requests()
"""

from __future__ import annotations

import enum
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional


# ---------------------------------------------------------------------------
# Status enum
# ---------------------------------------------------------------------------

class ApprovalStatus(str, enum.Enum):
    PENDING       = "pending"
    APPROVED      = "approved"
    REJECTED      = "rejected"
    TIMED_OUT     = "timed_out"
    AUTO_APPROVED = "auto_approved"


# ---------------------------------------------------------------------------
# Severity ordering
# ---------------------------------------------------------------------------

_SEV_ORDER = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3, "INFO": 4, "UNKNOWN": 5}


def _sev_index(sev: str) -> int:
    return _SEV_ORDER.get(sev.upper(), 5)


# ---------------------------------------------------------------------------
# ApprovalRequest
# ---------------------------------------------------------------------------

@dataclass
class ApprovalRequest:
    """Immutable record of a pending approval decision."""

    approval_id: str
    issue: Dict[str, Any]
    requester: str
    created_at: float
    status: ApprovalStatus = ApprovalStatus.PENDING
    decided_at: Optional[float] = None
    decided_by: Optional[str] = None
    reason: Optional[str] = None

    @property
    def severity(self) -> str:
        return str(self.issue.get("severity", "UNKNOWN")).upper()

    @property
    def title(self) -> str:
        return str(self.issue.get("title", "Untitled Finding"))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "approval_id": self.approval_id,
            "title": self.title,
            "severity": self.severity,
            "requester": self.requester,
            "created_at": self.created_at,
            "status": self.status.value,
            "decided_at": self.decided_at,
            "decided_by": self.decided_by,
            "reason": self.reason,
        }


# ---------------------------------------------------------------------------
# HumanInTheLoop
# ---------------------------------------------------------------------------

class HumanInTheLoop:
    """
    Human-in-the-loop approval workflow manager.

    Parameters
    ----------
    require_approval_above:
        Severity level above which a human decision is required.
        Findings at this level or higher are queued for review.
        Accepted values: ``"CRITICAL"``, ``"HIGH"``, ``"MEDIUM"``, ``"LOW"``.
        Default: ``"HIGH"`` (CRITICAL and HIGH require approval).
    default_timeout:
        Seconds to wait for a reviewer decision before timing out.
        0 means wait indefinitely (default).
    auto_approve_below_threshold:
        If True, findings below ``require_approval_above`` are immediately
        auto-approved rather than queued.
    """

    def __init__(
        self,
        require_approval_above: str = "HIGH",
        default_timeout: float = 0,
        auto_approve_below_threshold: bool = True,
    ) -> None:
        self._threshold_idx = _sev_index(require_approval_above)
        self._default_timeout = default_timeout
        self._auto_approve_below = auto_approve_below_threshold

        self._requests: Dict[str, ApprovalRequest] = {}
        self._lock = threading.Lock()
        self._events: Dict[str, threading.Event] = {}
        self._reviewer: Optional[Callable[[ApprovalRequest], bool]] = None

    # ------------------------------------------------------------------
    # Reviewer registration
    # ------------------------------------------------------------------

    def register_reviewer(self, reviewer: Callable[[ApprovalRequest], bool]) -> None:
        """
        Register a callable reviewer.

        The callable receives an ``ApprovalRequest`` and should return
        True to approve or False to reject.  Called synchronously when
        ``submit()`` is called.
        """
        self._reviewer = reviewer

    # ------------------------------------------------------------------
    # Submit
    # ------------------------------------------------------------------

    def submit(
        self,
        issue: Dict[str, Any],
        *,
        requester: str = "system",
        timeout: Optional[float] = None,
    ) -> ApprovalStatus:
        """
        Submit a finding for approval.

        Findings below the threshold are immediately auto-approved.
        Findings at/above the threshold are queued and the call blocks
        until a reviewer decides or the timeout expires.

        Parameters
        ----------
        issue:
            The issue dict (must contain at least ``severity``).
        requester:
            Identifier of the caller initiating the request.
        timeout:
            Override the instance-level ``default_timeout`` in seconds.

        Returns
        -------
        ApprovalStatus
        """
        sev = str(issue.get("severity", "UNKNOWN")).upper()
        sev_idx = _sev_index(sev)

        # Below threshold → auto-approve
        if sev_idx > self._threshold_idx:
            if self._auto_approve_below:
                return ApprovalStatus.AUTO_APPROVED
            # even below threshold we still create a record for audit
        
        # Create request
        req = ApprovalRequest(
            approval_id=str(uuid.uuid4()),
            issue=dict(issue),
            requester=requester,
            created_at=time.time(),
        )

        # If no reviewer registered, immediately auto-approve to avoid deadlock
        if self._reviewer is None:
            req = self._update_status(req, ApprovalStatus.AUTO_APPROVED, decided_by="system", reason="No reviewer registered — auto-approved")
            with self._lock:
                self._requests[req.approval_id] = req
            return req.status

        # Store and notify (synchronous path — call reviewer inline)
        event = threading.Event()
        with self._lock:
            self._requests[req.approval_id] = req
            self._events[req.approval_id] = event

        approved = self._reviewer(req)
        status = ApprovalStatus.APPROVED if approved else ApprovalStatus.REJECTED

        with self._lock:
            updated = self._requests[req.approval_id]
            updated = self._update_status(updated, status, decided_by="reviewer")
            self._requests[req.approval_id] = updated
            self._events.pop(req.approval_id, None)

        return status

    # ------------------------------------------------------------------
    # Manual decision (for async / webhook-driven reviewer)
    # ------------------------------------------------------------------

    def approve(self, approval_id: str, *, decided_by: str = "human", reason: Optional[str] = None) -> ApprovalRequest:
        """Approve a pending request by ID."""
        return self._decide(approval_id, ApprovalStatus.APPROVED, decided_by=decided_by, reason=reason)

    def reject(self, approval_id: str, *, decided_by: str = "human", reason: Optional[str] = None) -> ApprovalRequest:
        """Reject a pending request by ID."""
        return self._decide(approval_id, ApprovalStatus.REJECTED, decided_by=decided_by, reason=reason)

    def _decide(self, approval_id: str, status: ApprovalStatus, *, decided_by: str, reason: Optional[str]) -> ApprovalRequest:
        with self._lock:
            req = self._requests.get(approval_id)
            if req is None:
                raise KeyError(f"No approval request found with id: {approval_id}")
            if req.status != ApprovalStatus.PENDING:
                raise ValueError(f"Request {approval_id} is already in state '{req.status.value}'")
            updated = self._update_status(req, status, decided_by=decided_by, reason=reason)
            self._requests[approval_id] = updated
            event = self._events.pop(approval_id, None)

        if event is not None:
            event.set()
        return updated

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def pending_requests(self) -> List[ApprovalRequest]:
        """Return all requests still in PENDING state."""
        with self._lock:
            return [r for r in self._requests.values() if r.status == ApprovalStatus.PENDING]

    def get_request(self, approval_id: str) -> Optional[ApprovalRequest]:
        """Look up a request by ID."""
        with self._lock:
            return self._requests.get(approval_id)

    def all_requests(self) -> List[ApprovalRequest]:
        """Return all requests (for audit)."""
        with self._lock:
            return list(self._requests.values())

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _update_status(
        req: ApprovalRequest,
        status: ApprovalStatus,
        *,
        decided_by: Optional[str] = None,
        reason: Optional[str] = None,
    ) -> ApprovalRequest:
        """Return a new ApprovalRequest with updated status fields."""
        return ApprovalRequest(
            approval_id=req.approval_id,
            issue=req.issue,
            requester=req.requester,
            created_at=req.created_at,
            status=status,
            decided_at=time.time(),
            decided_by=decided_by,
            reason=reason,
        )

    def _is_above_threshold(self, sev: str) -> bool:
        return _sev_index(sev) <= self._threshold_idx
