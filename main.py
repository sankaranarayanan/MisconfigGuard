#!/usr/bin/env python3
"""
MisconfigGuard — Example usage script.

Demonstrates three core workflows:
  1. Ingest a local IaC directory
  2. Ingest a remote Git repository (public or private)
  3. Run RAG security queries against the indexed content

Run:
    python main.py
"""

import logging
import sys
from pathlib import Path

# ------------------------------------------------------------------
# Logging — configure before importing pipeline modules so that their
# module-level loggers inherit the root configuration.
# ------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("main")

# ------------------------------------------------------------------
# Pipeline imports
# ------------------------------------------------------------------
from chunker import Chunker
from embedding_generator import EmbeddingGenerator
from file_parser import FileParser
from file_scanner import FileScanner
from local_llm_client import LocalLLMClient
from rag_pipeline import RAGPipeline
from vector_store_manager import VectorStoreManager


# ------------------------------------------------------------------
# Helper: build a default pipeline
# ------------------------------------------------------------------

def build_pipeline(
    backend: str = "faiss",
    llm_model: str = "llama3",
) -> RAGPipeline:
    """Return a RAGPipeline wired with sensible defaults."""
    return RAGPipeline(
        scanner=FileScanner(max_file_size_mb=10),
        parser=FileParser(),
        chunker=Chunker(chunk_size=800, overlap=100),
        embedder=EmbeddingGenerator(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            cache_dir="./cache/embeddings",
            batch_size=32,
        ),
        vector_store=VectorStoreManager(
            backend=backend,
            index_path="./cache/faiss_index",
            chroma_persist_dir="./cache/chroma",
        ),
        llm_client=LocalLLMClient(
            base_url="http://localhost:11434",
            model=llm_model,
            timeout=120,
        ),
    )


# ------------------------------------------------------------------
# Workflow 1 — Local directory scan
# ------------------------------------------------------------------

def example_scan_local(directory: str) -> RAGPipeline:
    """Ingest all supported IaC files under *directory*."""
    logger.info("=" * 60)
    logger.info("WORKFLOW 1 — Scanning local directory: %s", directory)
    logger.info("=" * 60)

    pipeline = build_pipeline()
    total = pipeline.ingest_directory(directory)
    logger.info("Indexed %d chunks from %s", total, directory)
    return pipeline


# ------------------------------------------------------------------
# Workflow 2 — Git repository scan
# ------------------------------------------------------------------

def example_scan_repo(url: str, token: str = None) -> RAGPipeline:
    """Clone *url* and ingest all supported files."""
    logger.info("=" * 60)
    logger.info("WORKFLOW 2 — Scanning Git repository: %s", url)
    logger.info("=" * 60)

    pipeline = build_pipeline()
    total = pipeline.ingest_repository(url=url, token=token)
    logger.info("Indexed %d chunks from %s", total, url)
    return pipeline


# ------------------------------------------------------------------
# Workflow 3 — RAG security query
# ------------------------------------------------------------------

def example_rag_query(pipeline: RAGPipeline, query: str) -> dict:
    """Retrieve relevant chunks and run LLM security analysis."""
    logger.info("=" * 60)
    logger.info("WORKFLOW 3 — RAG Security Query")
    logger.info("=" * 60)
    logger.info("Query: %r", query)

    result = pipeline.analyze(query, top_k=5)

    sep = "=" * 72
    print(f"\n{sep}")
    print(f"QUERY:  {result['query']}")
    print(f"CHUNKS: {len(result['results'])} retrieved")
    print("-" * 72)
    for i, chunk in enumerate(result["results"], 1):
        print(f"  [{i}] {chunk['file_path']}")
    print("\nLLM ANALYSIS:")
    print("-" * 72)
    print(result["analysis"])
    print(sep)

    return result


# ------------------------------------------------------------------
# Sample IaC fixtures (created at runtime if ./sample_iac is absent)
# ------------------------------------------------------------------

def create_sample_iac(directory: str = "./sample_iac") -> None:
    """Write intentionally misconfigured sample files for demo purposes."""
    root = Path(directory)
    root.mkdir(exist_ok=True)

    # Terraform — insecure S3 bucket + open security group
    (root / "main.tf").write_text(
        '''\
resource "aws_s3_bucket" "data" {
  bucket = "company-public-data"
  acl    = "public-read"  # INSECURE: bucket is publicly readable
}

resource "aws_s3_bucket_versioning" "data" {
  bucket = aws_s3_bucket.data.id
  versioning_configuration {
    status = "Disabled"  # INSECURE: no versioning for audit trail
  }
}

resource "aws_security_group" "wide_open" {
  name = "allow-all"
  ingress {
    from_port   = 0
    to_port     = 65535
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]  # INSECURE: open to the entire internet
  }
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

resource "aws_db_instance" "prod" {
  identifier           = "prod-db"
  engine               = "mysql"
  instance_class       = "db.t3.micro"
  username             = "admin"
  password             = "SuperSecret123!"  # INSECURE: hardcoded password
  publicly_accessible  = true               # INSECURE: RDS exposed to internet
  storage_encrypted    = false              # INSECURE: unencrypted storage
}
'''
    )

    # YAML — hardcoded credentials + disabled TLS
    (root / "app-config.yaml").write_text(
        '''\
database:
  host: prod-db.internal
  port: 5432
  username: admin
  password: "P@ssw0rd!hardcoded"  # INSECURE: secret in plaintext config
  ssl_mode: disable               # INSECURE: no TLS for DB connection

api:
  secret_key: "abc123insecure"    # INSECURE: weak/hardcoded API secret
  debug: true                     # INSECURE: debug mode in production
  cors_origins: ["*"]             # INSECURE: unrestricted CORS

storage:
  provider: s3
  bucket: public-assets
  acl: public-read                # INSECURE: public bucket ACL
'''
    )

    # JSON — IAM wildcard permissions
    (root / "iam-policy.json").write_text(
        '''\
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "OverlyPermissive",
      "Effect": "Allow",
      "Action": "*",
      "Resource": "*"
    }
  ]
}
'''
    )

    logger.info("Sample IaC files created in %s", directory)


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

if __name__ == "__main__":
    SAMPLE_DIR = "./sample_iac"

    # Ensure sample files exist
    if not Path(SAMPLE_DIR).exists():
        logger.info("Creating sample IaC fixtures in %s ...", SAMPLE_DIR)
        create_sample_iac(SAMPLE_DIR)

    # ----------------------------------------------------------------
    # Workflow 1 — scan local directory
    # ----------------------------------------------------------------
    pipeline = example_scan_local(SAMPLE_DIR)

    # ----------------------------------------------------------------
    # Workflow 2 — scan a Git repository (requires network + GitPython)
    # Uncomment and set a real URL to test:
    # ----------------------------------------------------------------
    # pipeline = example_scan_repo(
    #     url="https://github.com/bridgecrewio/checkov",
    # )
    # pipeline = example_scan_repo(
    #     url="https://github.com/your-org/private-infra",
    #     token="ghp_your_personal_access_token",
    # )

    # ----------------------------------------------------------------
    # Workflow 3 — RAG security queries
    # ----------------------------------------------------------------
    security_queries = [
        "Are there any hardcoded credentials, passwords, or API secrets?",
        "Are there any S3 buckets with public read or public write access?",
        "Check for overly permissive security groups open to 0.0.0.0/0",
        "Are there any IAM policies with wildcard Action or Resource?",
        "Is there any disabled encryption or TLS/SSL configuration?",
    ]

    for query in security_queries:
        example_rag_query(pipeline, query)
        print()
