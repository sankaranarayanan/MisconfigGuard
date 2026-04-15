"""
MisconfigGuard — Security sub-package.

Exports the key security components used by the broader pipeline:

    InputSanitizer       — validate and sanitize all user / external inputs
    ContextIsolation     — per-request isolation; prevents state leakage
    RAGPoisonDefense     — trust-aware indexing, content validation, filter retrieval
    RBACEnforcer         — role-based access control + tool access control
    LLMGuardrails        — output schema enforcement + LLM output validation
    OutputControls       — output filtering, redaction, and controls
    AuditLogger          — structured audit logging and observability
    HumanInTheLoop       — human-approval workflow for high-risk findings
"""

from .context_isolation import ContextIsolation
from .human_in_the_loop import ApprovalRequest, ApprovalStatus, HumanInTheLoop
from .input_sanitizer import InputSanitizer, SanitizationError
from .llm_guardrails import LLMGuardrails, OutputValidationError
from .observability import AuditEvent, AuditLogger
from .output_controls import OutputControls
from .rag_poison_defense import ContentValidationError, RAGPoisonDefense
from .rbac import Permission, RBACEnforcer, Role

__all__ = [
    "InputSanitizer",
    "SanitizationError",
    "ContextIsolation",
    "RAGPoisonDefense",
    "ContentValidationError",
    "RBACEnforcer",
    "Role",
    "Permission",
    "LLMGuardrails",
    "OutputValidationError",
    "OutputControls",
    "AuditLogger",
    "AuditEvent",
    "HumanInTheLoop",
    "ApprovalRequest",
    "ApprovalStatus",
]
