#!/usr/bin/env python3
"""
MisconfigGuard CLI

Scan IaC files and analyse them for security misconfigurations using
a fully local LLM + RAG stack (no external API calls).

Commands
--------
  scan-local   Ingest a local directory tree
  scan-repo    Clone and ingest a remote Git repository
  query        Retrieve relevant context and run LLM security analysis

Usage examples
--------------
  python cli.py scan-local ./terraform-project
  python cli.py scan-repo https://github.com/org/infra --token ghp_xxxx
  python cli.py query "Are there any hardcoded secrets?"
  python cli.py query "Check for open security groups" --top-k 5 --output results.json
  python cli.py --config custom.yaml scan-local ./iac
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Optional

import yaml

from misconfigguard.config import resolve_config_path


# ------------------------------------------------------------------
# Config loader
# ------------------------------------------------------------------

def load_config(config_path: str = "config.yaml") -> dict:
    """Load YAML config, falling back to the canonical config folder."""
    resolved_path = resolve_config_path(config_path)
    try:
        with open(resolved_path, "r") as fh:
            cfg = yaml.safe_load(fh) or {}
        logging.getLogger(__name__).debug("Loaded config from %s", resolved_path)
        return cfg
    except FileNotFoundError:
        return {}
    except yaml.YAMLError as exc:
        print(f"[WARN] Failed to parse config {resolved_path}: {exc}", file=sys.stderr)
        return {}


# ------------------------------------------------------------------
# Logging setup
# ------------------------------------------------------------------

def setup_logging(level: str = "INFO", log_file: Optional[str] = None) -> None:
    handlers: list = [logging.StreamHandler(sys.stdout)]
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=handlers,
    )


# ------------------------------------------------------------------
# Pipeline factory
# ------------------------------------------------------------------

def build_pipeline(cfg: dict):
    """Construct a fully configured RAGPipeline from *cfg*."""
    from embedding_generator import EmbeddingGenerator
    from file_parser import FileParser
    from file_scanner import FileScanner
    from intelligent_chunker import IntelligentChunker
    from local_llm_client import LocalLLMClient
    from rag_pipeline import RAGPipeline
    from vector_store_manager import VectorStoreManager

    ing = cfg.get("ingestion", {})
    ck  = cfg.get("chunking", {})
    emb = cfg.get("embedding", {})
    vs  = cfg.get("vector_store", {})
    llm = cfg.get("llm", {})

    chunker = IntelligentChunker(
        max_tokens_per_chunk=ck.get("max_tokens_per_chunk", 500),
        overlap_tokens=ck.get("overlap_tokens", 50),
        chunking_strategy=ck.get("strategy", "semantic"),
        resolve_dependencies=ck.get("resolve_dependencies", True),
    )

    return RAGPipeline(
        scanner=FileScanner(
            max_file_size_mb=ing.get("max_file_size_mb", 10),
        ),
        parser=FileParser(),
        chunker=chunker,
        embedder=EmbeddingGenerator(
            model_name=emb.get("model", "sentence-transformers/all-MiniLM-L6-v2"),
            cache_dir=emb.get("cache_dir", "./cache/embeddings"),
            batch_size=emb.get("batch_size", 32),
        ),
        vector_store=VectorStoreManager(
            backend=vs.get("backend", "faiss"),
            index_path=vs.get("index_path", "./cache/faiss_index"),
            chroma_persist_dir=vs.get("chroma_persist_dir", "./cache/chroma"),
        ),
        llm_client=LocalLLMClient(
            base_url=llm.get("base_url", "http://localhost:11434"),
            model=llm.get("model", "llama3"),
            timeout=llm.get("timeout", 120),
            max_tokens=llm.get("max_tokens", 2048),
        ),
        expand_dependencies=ck.get("resolve_dependencies", True),
        query_routing_cfg=cfg.get("query_routing", {}),
        retrieval_cfg=cfg.get("retrieval", {}),
        iam_cfg=cfg.get("iam", {}),
        workload_identity_cfg=cfg.get("workload_identity", {}),
        prompt_injection_cfg=cfg.get("prompt_injection", {}),
        secrets_cfg=cfg.get("secrets", {}),
    )


# ------------------------------------------------------------------
# Command handlers
# ------------------------------------------------------------------

def cmd_scan_local(args: argparse.Namespace, cfg: dict) -> None:
    """Ingest a local directory."""
    if getattr(args, "chunking_strategy", None):
        cfg.setdefault("chunking", {})["strategy"] = args.chunking_strategy
    pipeline = build_pipeline(cfg)
    total = pipeline.ingest_directory(args.path)
    print(f"\n✅  Ingested {total:,} chunks from: {args.path}")


def cmd_scan_repo(args: argparse.Namespace, cfg: dict) -> None:
    """Clone and ingest a remote Git repository."""
    if getattr(args, "chunking_strategy", None):
        cfg.setdefault("chunking", {})["strategy"] = args.chunking_strategy
    pipeline = build_pipeline(cfg)
    total = pipeline.ingest_repository(
        url=args.url,
        token=args.token,
        branch=args.branch,
        clone_dir=cfg.get("git", {}).get("clone_dir", "./tmp/repos"),
    )
    print(f"\n✅  Ingested {total:,} chunks from repository: {args.url}")


def _parse_metadata_filter(args: argparse.Namespace) -> Optional[dict]:
    """Build metadata filter dict from CLI flags (cloud, file-type)."""
    filt = {}
    if getattr(args, "cloud", None):
        filt["cloud_provider"] = args.cloud
    if getattr(args, "file_type", None):
        filt["file_type"] = args.file_type
    return filt or None


def _print_plain(result: dict, args: argparse.Namespace) -> None:
    """Print the standard (non-structured) analysis output."""
    sep = "=" * 72
    print(f"\n{sep}")
    print(f"🔍 QUERY:  {result['query']}")
    print(sep)

    print(f"\n📄 RETRIEVED {len(result['results'])} CHUNK(S):")
    for i, chunk in enumerate(result["results"], 1):
        print(
            f"   [{i}] {chunk.get('file_path', '')}  "
            f"(chunk #{chunk.get('chunk_index', '')})"
        )

    print("\n🤖 LLM SECURITY ANALYSIS:")
    print("-" * 72)
    print(result["analysis"])
    print(sep)

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as fh:
            json.dump(result, fh, indent=2)
        print(f"\n💾 Full results saved → {args.output}")


def _print_structured(result: dict, args: argparse.Namespace) -> None:
    """Print structured analysis output with issue list."""
    sep = "=" * 72
    issues = result.get("issues", [])
    print(f"\n{sep}")
    print(f"🔍 QUERY:  {result['query']}")
    print(sep)

    meta = result.get("metadata", {})
    retr = meta.get("retrieval", {})
    print(
        f"\n📄 Retrieved: {retr.get('code_count', 0)} code chunk(s), "
        f"{retr.get('security_count', 0)} security rule(s)"
        + ("  [cached]" if meta.get("cached") else "")
    )

    if issues:
        print(f"\n⚠️  SECURITY FINDINGS ({len(issues)}):")
        print("-" * 72)
        for i, issue in enumerate(issues, 1):
            sev = issue.get("severity", "INFO")
            icon = {
                "CRITICAL": "🔴", "HIGH": "🟠", "MEDIUM": "🟡",
                "LOW": "🟢", "INFO": "ℹ️",
            }.get(sev, "❓")
            print(f"\n  {icon} [{i}] {issue.get('title', 'Issue')}")
            print(f"      Severity:  {sev}")
            res = issue.get("affected_resource", "")
            if res:
                print(f"      Resource:  {res}")
            print(f"      Detail:    {issue.get('description', '')[:200]}")
            rec = issue.get("recommendation", "")
            if rec:
                print(f"      Fix:       {rec[:200]}")
    else:
        print("\n✅  No security issues detected.")

    summary = result.get("summary", "")
    if summary:
        print(f"\n📝 SUMMARY:\n   {summary}")
    print(sep)

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as fh:
            json.dump(result, fh, indent=2)
        print(f"\n💾 Full results saved → {args.output}")


def cmd_query(args: argparse.Namespace, cfg: dict) -> None:
    """Load the vector index and run a RAG security query."""
    pipeline = build_pipeline(cfg)
    if not pipeline.load_index():
        print(
            "⚠️   No vector index found.\n"
            "    Run  scan-local <path>  or  scan-repo <url>  first.",
            file=sys.stderr,
        )
        sys.exit(1)

    result = pipeline.query(
        args.query,
        top_k=args.top_k,
        use_llm_routing=not getattr(args, "no_llm_routing", False),
        structured=getattr(args, "structured", False),
        stream=args.stream,
    )
    print(f"\n[Routed to module: {result['intent']}]\n")
    print(result.get("analysis", ""))
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as fh:
            json.dump(result, fh, indent=2)


# ------------------------------------------------------------------
# Argument parser
# ------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="misconfigguard",
        description="MisconfigGuard — IaC security analysis with local LLM + RAG",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        metavar="FILE",
        help="Path to YAML config file (default: config.yaml, fallback: config/config.yaml)",
    )
    parser.add_argument(
        "--log-level",
        default=None,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Override log level from config",
    )

    sub = parser.add_subparsers(dest="command", required=True)

    # scan-local
    p_local = sub.add_parser("scan-local", help="Ingest a local directory")
    p_local.add_argument("path", help="Directory path to scan recursively")
    p_local.add_argument(
        "--chunking-strategy",
        choices=["semantic", "fixed", "hybrid"],
        default=None,
        dest="chunking_strategy",
        help="Override chunking strategy from config (default: semantic)",
    )

    # scan-repo
    p_repo = sub.add_parser(
        "scan-repo", help="Clone and ingest a remote Git repository"
    )
    p_repo.add_argument("url", help="Git repository URL")
    p_repo.add_argument(
        "--token",
        default=None,
        metavar="PAT",
        help="Personal access token for private repositories",
    )
    p_repo.add_argument(
        "--branch",
        default=None,
        help="Branch to check out (default: remote HEAD)",
    )
    p_repo.add_argument(
        "--chunking-strategy",
        choices=["semantic", "fixed", "hybrid"],
        default=None,
        dest="chunking_strategy",
        help="Override chunking strategy from config (default: semantic)",
    )

    # query
    p_query = sub.add_parser(
        "query", help="Run a RAG security query against the indexed content"
    )
    p_query.add_argument("query", help="Security question or analysis directive")
    p_query.add_argument(
        "--top-k",
        type=int,
        default=5,
        metavar="K",
        help="Number of chunks to retrieve (default: 5)",
    )
    p_query.add_argument(
        "--stream",
        action="store_true",
        help="Stream LLM tokens to stdout as they are generated",
    )
    p_query.add_argument(
        "--output",
        default=None,
        metavar="FILE",
        help="Save full JSON results to this file",
    )
    p_query.add_argument(
        "--structured",
        action="store_true",
        help=(
            "Use hybrid retrieval + security KB for structured JSON output "
            "(issues list with severity, recommendations, and evidence)"
        ),
    )
    p_query.add_argument(
        "--no-llm-routing",
        action="store_true",
        dest="no_llm_routing",
        help="Skip LLM classification and use keyword heuristics only",
    )
    p_query.add_argument(
        "--cloud",
        default=None,
        choices=["aws", "azure", "gcp", "k8s"],
        help="Filter retrieval to a specific cloud provider (structured mode only)",
    )
    p_query.add_argument(
        "--file-type",
        default=None,
        choices=["terraform", "yaml", "json"],
        dest="file_type",
        help="Filter retrieval to a specific file type (structured mode only)",
    )

    # analyze-iam
    p_iam = sub.add_parser(
        "analyze-iam",
        help="Detect Azure managed identity over-permissions in IaC files",
    )
    p_iam.add_argument(
        "path",
        help="File or directory to analyse (Terraform, YAML, JSON)",
    )
    p_iam.add_argument(
        "--no-llm",
        action="store_true",
        dest="no_llm",
        help="Run rule-based detection only (skip LLM enrichment)",
    )
    p_iam.add_argument(
        "--max-roles",
        type=int,
        default=None,
        dest="max_roles",
        metavar="N",
        help="Flag identities with more than N role assignments (default from config)",
    )
    p_iam.add_argument(
        "--output",
        default=None,
        metavar="FILE",
        help="Save full JSON results to this file",
    )

    # analyze-workload-identity
    p_wi = sub.add_parser(
        "analyze-workload-identity",
        help="Detect workload identity federation misconfigurations in IaC files",
    )
    p_wi.add_argument(
        "path",
        help="File or directory to analyse (Terraform, YAML, JSON)",
    )
    p_wi.add_argument(
        "--no-llm",
        action="store_true",
        dest="no_llm",
        help="Run rule-based detection only (skip LLM enrichment)",
    )
    p_wi.add_argument(
        "--output",
        default=None,
        metavar="FILE",
        help="Save full JSON results to this file",
    )

    # scan-secrets
    p_secrets = sub.add_parser(
        "scan-secrets",
        help="Detect hardcoded secrets in Terraform, YAML, and JSON files",
    )
    p_secrets.add_argument(
        "path",
        help="File or directory to analyse for hardcoded secrets",
    )
    p_secrets.add_argument(
        "--no-llm",
        action="store_true",
        dest="no_llm",
        help="Run rule-based detection only (skip LLM enrichment)",
    )
    p_secrets.add_argument(
        "--output",
        default=None,
        metavar="FILE",
        help="Save full JSON results to this file",
    )

    return parser


# ------------------------------------------------------------------
# IAM command helpers
# ------------------------------------------------------------------

def cmd_analyze_iam(args: argparse.Namespace, cfg: dict) -> None:
    """Detect Azure managed identity over-permissions."""
    from iam_analyzer import IAMSecurityAnalyzer

    iam_cfg = cfg.get("iam", {})
    max_roles = args.max_roles or iam_cfg.get("max_roles_per_identity", 3)
    use_llm   = not args.no_llm and iam_cfg.get("use_llm", True)

    # Only attach the pipeline for LLM enrichment if requested
    pipeline = None
    if use_llm:
        try:
            pipeline = build_pipeline(cfg)
            pipeline.load_index()
        except Exception:
            pipeline = None

    analyzer = IAMSecurityAnalyzer(
        pipeline               = pipeline,
        use_llm                = use_llm,
        max_roles_per_identity = max_roles,
    )

    target = args.path
    if Path(target).is_file():
        result = analyzer.analyze_file(target)
    else:
        result = analyzer.analyze_directory(target)

    _print_iam_results(result, args)


def cmd_analyze_workload_identity(args: argparse.Namespace, cfg: dict) -> None:
    """Detect workload identity federation misconfigurations."""
    from workload_identity_analyzer import WorkloadIdentitySecurityAnalyzer

    wi_cfg = cfg.get("workload_identity", {})
    use_llm = not args.no_llm and wi_cfg.get("use_llm", True)

    pipeline = None
    if use_llm:
        try:
            pipeline = build_pipeline(cfg)
            pipeline.load_index()
        except Exception:
            pipeline = None

    analyzer = WorkloadIdentitySecurityAnalyzer(
        pipeline=pipeline,
        use_llm=use_llm,
        top_k_code=wi_cfg.get("top_k_code", 5),
        top_k_security=wi_cfg.get("top_k_security", 3),
    )

    if Path(args.path).is_file():
        result = analyzer.analyze_file(args.path)
    else:
        result = analyzer.analyze_directory(args.path)

    _print_workload_identity_results(result, args)


def cmd_scan_secrets(args: argparse.Namespace, cfg: dict) -> None:
    """Detect hardcoded secrets in IaC files."""
    from secrets_analyzer import HardcodedSecretsAnalyzer

    secrets_cfg = cfg.get("secrets", {})
    use_llm = not args.no_llm and secrets_cfg.get("use_llm", True)

    pipeline = None
    if use_llm:
        try:
            pipeline = build_pipeline(cfg)
            pipeline.load_index()
        except Exception:
            pipeline = None

    analyzer = HardcodedSecretsAnalyzer(
        pipeline=pipeline,
        use_llm=use_llm,
        top_k_code=secrets_cfg.get("top_k_code", 5),
        top_k_security=secrets_cfg.get("top_k_security", 3),
        entropy_threshold=secrets_cfg.get("entropy_threshold", 4.5),
    )

    if Path(args.path).is_file():
        result = analyzer.analyze_file(args.path)
    else:
        result = analyzer.analyze_directory(args.path)

    _print_secrets_results(result, args)


def _print_iam_results(result: dict, args: argparse.Namespace) -> None:
    """Pretty-print IAM analysis results to stdout."""
    sep    = "=" * 72
    issues = result.get("issues", [])
    meta   = result.get("metadata", {})
    summ   = result.get("summary", {})

    print(f"\n{sep}")
    print("🔐  AZURE IAM OVER-PERMISSION ANALYSIS")
    print(sep)

    files = meta.get("files_analyzed", [])
    print(f"\n📁  Files analysed  : {len(files)}")
    print(f"🔗  Assignments found: {summ.get('total_assignments', 0)}")
    print(f"⚠️   Findings        : {len(issues)}  "
          f"(HIGH:{summ.get('high_severity',0)}  "
          f"MEDIUM:{summ.get('medium_severity',0)}  "
          f"LOW:{summ.get('low_severity',0)})")

    if not issues:
        print("\n✅  No over-permission issues detected.")
    else:
        print(f"\n{'─' * 72}")
        for i, issue in enumerate(issues, 1):
            sev  = issue.get("severity", "INFO")
            icon = {"HIGH": "🔴", "MEDIUM": "🟠", "LOW": "🟢"}.get(sev, "❓")
            print(f"\n{icon}  [{i}] {issue.get('issue', '')}")
            print(f"     Identity : {issue.get('identity', '')}")
            print(f"     Role     : {issue.get('role', '')}")
            print(f"     Scope    : {issue.get('scope', '')}  "
                  f"({issue.get('scope_value', '') or 'see source'})")
            print(f"     Severity : {sev}  ({issue.get('rule_id', '')})")
            expl = issue.get("explanation", "")[:200]
            if expl:
                print(f"     Why      : {expl}")
            fix = issue.get("fix", "")[:200]
            if fix:
                print(f"     Fix      : {fix}")
            recs = issue.get("recommendations", [])
            if recs:
                print(f"     Alt roles: {', '.join(recs[:3])}")
            fp = issue.get("file_path", "")
            if fp:
                print(f"     File     : {fp}")

    print(f"\n{sep}")

    if getattr(args, "output", None):
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as fh:
            json.dump(result, fh, indent=2)
        print(f"\n💾 Full results saved → {args.output}")


def _print_workload_identity_results(result: dict, args: argparse.Namespace) -> None:
    """Pretty-print workload identity analysis results to stdout."""
    sep = "=" * 72
    issues = result.get("issues", [])
    meta = result.get("metadata", {})
    summary = result.get("summary", {})

    print(f"\n{sep}")
    print("🪪  WORKLOAD IDENTITY MISCONFIGURATION ANALYSIS")
    print(sep)

    print(f"\n📁  Files analysed : {len(meta.get('files_analyzed', []))}")
    print(f"🔗  Configs found  : {summary.get('total_configs', 0)}")
    print(
        f"⚠️   Findings       : {len(issues)}  "
        f"(HIGH:{summary.get('high_severity', 0)}  "
        f"MEDIUM:{summary.get('medium_severity', 0)}  "
        f"LOW:{summary.get('low_severity', 0)})"
    )

    if not issues:
        print("\n✅  No workload identity misconfigurations detected.")
    else:
        print(f"\n{'─' * 72}")
        for index, issue in enumerate(issues, 1):
            severity = issue.get("severity", "INFO")
            icon = {"HIGH": "🔴", "MEDIUM": "🟠", "LOW": "🟢"}.get(severity, "❓")
            print(f"\n{icon}  [{index}] {issue.get('issue', '')}")
            print(f"     Identity : {issue.get('identity', '')}")
            print(f"     Type     : {issue.get('type', '')}")
            print(f"     Issuer   : {issue.get('issuer', '') or 'missing'}")
            print(f"     Subject  : {issue.get('subject', '') or 'missing'}")
            audiences = issue.get("audiences", [])
            print(f"     Audience : {', '.join(audiences) if audiences else 'missing'}")
            print(f"     Severity : {severity}  ({issue.get('rule_id', '')})")
            explanation = issue.get("explanation", "")[:200]
            if explanation:
                print(f"     Why      : {explanation}")
            fix = issue.get("fix", "")[:200]
            if fix:
                print(f"     Fix      : {fix}")
            file_path = issue.get("file_path", "")
            if file_path:
                print(f"     File     : {file_path}")

    print(f"\n{sep}")

    if getattr(args, "output", None):
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as fh:
            json.dump(result, fh, indent=2)
        print(f"\n💾 Full results saved → {args.output}")


def _print_secrets_results(result: dict, args: argparse.Namespace) -> None:
    """Pretty-print hardcoded secrets analysis results to stdout."""
    sep = "=" * 72
    issues = result.get("issues", [])
    meta = result.get("metadata", {})
    summary = result.get("summary", {})

    print(f"\n{sep}")
    print("🔑  HARDCODED SECRETS ANALYSIS")
    print(sep)

    print(f"\n📁  Files analysed : {len(meta.get('files_analyzed', []))}")
    print(f"🔎  Matches found  : {summary.get('total_findings', 0)}")
    print(
        f"⚠️   Findings       : {len(issues)}  "
        f"(HIGH:{summary.get('high_severity', 0)}  "
        f"MEDIUM:{summary.get('medium_severity', 0)}  "
        f"LOW:{summary.get('low_severity', 0)})"
    )

    if not issues:
        print("\n✅  No hardcoded secrets detected.")
    else:
        print(f"\n{'─' * 72}")
        for index, issue in enumerate(issues, 1):
            severity = issue.get("severity", "INFO")
            icon = {"HIGH": "🔴", "MEDIUM": "🟠", "LOW": "🟢"}.get(severity, "❓")
            print(f"\n{icon}  [{index}] {issue.get('issue', '')}")
            print(f"     Type     : {issue.get('secret_type', '')}")
            print(f"     Severity : {severity}  ({issue.get('rule_id', '')})")
            print(f"     Match    : {issue.get('match', '')}")
            print(f"     Line     : {issue.get('line_number', '')}")
            explanation = issue.get("explanation", "")[:200]
            if explanation:
                print(f"     Why      : {explanation}")
            fix = issue.get("fix", "")[:200]
            if fix:
                print(f"     Fix      : {fix}")
            file_path = issue.get("file_path", "")
            if file_path:
                print(f"     File     : {file_path}")

    print(f"\n{sep}")

    if getattr(args, "output", None):
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as fh:
            json.dump(result, fh, indent=2)
        print(f"\n💾 Full results saved → {args.output}")


# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------

def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    cfg = load_config(args.config)

    # Resolve log level: CLI flag > config > default INFO
    log_cfg = cfg.get("logging", {})
    log_level = args.log_level or log_cfg.get("level", "INFO")
    setup_logging(level=log_level, log_file=log_cfg.get("file"))

    dispatch = {
        "scan-local": cmd_scan_local,
        "scan-repo": cmd_scan_repo,
        "query": cmd_query,
        "analyze-iam": cmd_analyze_iam,
        "analyze-workload-identity": cmd_analyze_workload_identity,
        "scan-secrets": cmd_scan_secrets,
    }
    dispatch[args.command](args, cfg)


if __name__ == "__main__":
    main()
