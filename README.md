# MisconfigGuard 🛡️

A **fully local** security analysis platform for Infrastructure-as-Code (IaC)
and application configurations, powered by a RAG pipeline and a locally running
LLM — **no external API calls**.

---

## Architecture

```
FileScanner ──► FileParser ──► Chunker
                                   │
                           EmbeddingGenerator
                         (sentence-transformers)
                                   │
                           VectorStoreManager
                           (FAISS / Chroma)
                                   │
            query ──► similarity search (Top-K)
                                   │
                           LocalLLMClient
                           (Ollama HTTP API)
                                   │
                          Security Analysis Report
```

### Module Map

| Module | Class | Responsibility |
|---|---|---|
| `file_scanner.py` | `FileScanner` | Recursive directory walk; generator-based; size-limit guard |
| `git_ingestor.py` | `GitIngestor` | Clone / pull public & private Git repos (PAT injection) |
| `file_parser.py` | `FileParser` | Stream-read `.tf` / `.yml` / `.yaml` / `.json`; normalise to `FileRecord` |
| `chunker.py` | `Chunker` | Overlapping sliding-window token chunking |
| `embedding_generator.py` | `EmbeddingGenerator` | Local sentence-transformers; SHA-256 disk cache |
| `vector_store_manager.py` | `VectorStoreManager` | FAISS (default) or Chroma; save/load; Top-K cosine search |
| `local_llm_client.py` | `LocalLLMClient` | Ollama HTTP client; streaming support; security prompt template |
| `rag_pipeline.py` | `RAGPipeline` | Orchestrates all components; parallel parsing; incremental indexing |
| `iam_parser.py` | `IAMParser` | Extract Azure managed identities and role assignments from Terraform / ARM JSON / YAML |
| `permission_analyzer.py` | `PermissionAnalyzer` | Deterministic least-privilege checks for Azure managed identities |
| `iam_analyzer.py` | `IAMSecurityAnalyzer` | Hybrid rule-based + RAG-enriched IAM over-permission analysis |
| `workload_identity_parser.py` | `WorkloadIdentityParser` | Extract workload identity federation config from Terraform, Kubernetes YAML, and ARM docs |
| `federation_analyzer.py` | `FederationAnalyzer` | Deterministic validation of issuer, audience, subject, and trust boundaries |
| `workload_identity_analyzer.py` | `WorkloadIdentitySecurityAnalyzer` | Hybrid rule-based + RAG-enriched workload identity misconfiguration analysis |
| `pipeline_runner.py` | `PipelineRunner` | CI/CD orchestration across IAM, workload identity, and secrets detectors |
| `report_generator.py` | `ReportGenerator` | JSON, SARIF, and console summaries for pipeline output |
| `exit_handler.py` | `ExitHandler` | Pipeline pass/fail behavior based on severity thresholds |
| `scanner.py` | `main()` | CI-oriented CLI for repository and changed-file scanning |

---

## Prerequisites

### 1. Python dependencies

```bash
pip install -r requirements.txt
```

### 2. Ollama (local LLM runtime)

```bash
# Install from https://ollama.com
ollama pull llama3      # or: ollama pull mistral
ollama serve            # starts the API on http://localhost:11434
```

### 3. (Optional) Upgrade embedding model for higher accuracy

```python
# In config.yaml, change:
embedding:
  model: "BAAI/bge-large-en"   # 1024-dim, higher accuracy
```

---

## Quick Start

### Scan a local IaC directory

```bash
python cli.py scan-local ./my-terraform-project
```

### Scan a public Git repository

```bash
python cli.py scan-repo https://github.com/org/infra-repo
```

### Scan a private Git repository

```bash
python cli.py scan-repo https://github.com/org/private-infra \
    --token ghp_your_personal_access_token \
    --branch main
```

### Run a security query

```bash
python cli.py query "Are there any hardcoded credentials or API secrets?"
python cli.py query "Check for open security groups (0.0.0.0/0)" --top-k 5
python cli.py query "Any S3 buckets with public-read ACL?" --output results.json
```

### Analyze Azure managed identity permissions

```bash
python cli.py analyze-iam ./azure-infra
python cli.py analyze-iam ./azure-infra --no-llm
python cli.py analyze-iam ./azure-infra --max-roles 2 --output iam-findings.json
```

The IAM analyzer detects:

1. High-privilege roles such as `Owner` and `Contributor` at subscription or resource-group scope.
2. Excessive, overlapping, or redundant role assignments on the same managed identity.
3. Broad service-wide access roles and over-broad scopes that should be reduced.

Example JSON output:

```json
{
  "issues": [
    {
      "identity": "app-mi",
      "role": "Contributor",
      "scope": "subscription",
      "severity": "HIGH",
      "issue": "Contributor role assigned at subscription scope",
      "explanation": "The managed identity can modify resources across the entire subscription.",
      "fix": "Replace with a least-privilege role and reduce scope to the required resource group or resource."
    }
  ]
}
```

### Analyze workload identity federation

```bash
python cli.py analyze-workload-identity ./azure-infra
python cli.py analyze-workload-identity ./azure-infra --no-llm
python cli.py analyze-workload-identity ./azure-infra --output workload-identity-findings.json
```

The workload identity analyzer detects:

1. Invalid or missing OIDC issuer configuration.
2. Missing or wildcard audience restrictions.
3. Overly broad subject claims and weak trust boundaries.
4. External or cross-tenant trust relationships without sufficient restriction.

Example JSON output:

```json
{
  "issues": [
    {
      "identity": "api",
      "type": "workload_identity",
      "severity": "HIGH",
      "issue": "Missing or overly broad audience restriction",
      "explanation": "The federated credential does not restrict token audience tightly enough.",
      "fix": "Define explicit audience values and remove wildcard trust."
    }
  ]
}
```

### Run the CI/CD scanner locally

```bash
python scanner.py --path ./azure-infra --output results.json
python scanner.py --path ./azure-infra --output results.json --fail-on-high
python scanner.py --path ./azure-infra --output results.json --sarif-output results.sarif
python scanner.py --path ./azure-infra --changed-file main.tf --changed-file values.yaml
```

The CI scanner aggregates all supported deterministic detectors and emits:

1. Console summary with total, HIGH, MEDIUM, and LOW counts.
2. JSON output for pipeline artifacts.
3. Optional SARIF output for GitHub code scanning integration.

### CI/CD pipeline integration

Reusable examples are included in:

1. `.github/workflows/security-scan.yml` for GitHub Actions PR and main-branch scans.
2. `azure-pipelines.yml` for Azure DevOps PR validation.

Both examples install dependencies, run `scanner.py`, and publish JSON results as a pipeline artifact.

### Stream LLM output token-by-token

```bash
python cli.py query "Summarise all critical findings" --stream
```

### Run all example workflows

```bash
python main.py
```

---

## Configuration

`config.yaml` (at project root):

```yaml
ingestion:
  chunk_size: 800          # target words per chunk
  chunk_overlap: 100       # overlap between adjacent chunks
  max_file_size_mb: 10

git:
  clone_dir: "./tmp/repos"
  depth: 1                 # shallow clone

embedding:
  model: "sentence-transformers/all-MiniLM-L6-v2"
  cache_dir: "./cache/embeddings"
  batch_size: 32

vector_store:
  backend: "faiss"         # or "chroma"
  index_path: "./cache/faiss_index"
  top_k: 5

llm:
  base_url: "http://localhost:11434"
  model: "llama3"
  timeout: 120
  max_tokens: 2048

logging:
  level: "INFO"
  file: "./logs/misconfigguard.log"

workload_identity:
  use_llm: true
  top_k_code: 5
  top_k_security: 3

secrets:
  use_llm: true
  top_k_code: 5
  top_k_security: 3
  entropy_threshold: 4.5
```

---

## Python API

```python
from rag_pipeline import RAGPipeline
from file_scanner import FileScanner
from chunker import Chunker
from embedding_generator import EmbeddingGenerator
from vector_store_manager import VectorStoreManager
from local_llm_client import LocalLLMClient

pipeline = RAGPipeline(
    scanner=FileScanner(max_file_size_mb=10),
    chunker=Chunker(chunk_size=800, overlap=100),
    embedder=EmbeddingGenerator(model_name="sentence-transformers/all-MiniLM-L6-v2"),
    vector_store=VectorStoreManager(backend="faiss"),
    llm_client=LocalLLMClient(model="llama3"),
    incremental=True,      # skip already-indexed files
    max_workers=4,         # parallel file parsing threads
)

# Ingest
pipeline.ingest_directory("./terraform-project")
pipeline.ingest_repository("https://github.com/org/infra", token="ghp_...")

# Query
result = pipeline.analyze("Are there any hardcoded passwords?", top_k=5)
print(result["analysis"])

# Reload a saved index without re-indexing
pipeline.load_index()
result = pipeline.analyze("Open security groups?")
```

---

## FileRecord Schema

Every parsed file is normalised into:

```json
{
  "file_path": "/path/to/main.tf",
  "file_type": "terraform",
  "content": "resource \"aws_s3_bucket\" ...",
  "metadata": {
    "repo": "https://github.com/org/infra",
    "branch": "main",
    "commit": "a1b2c3d4"
  }
}
```

---

## Performance Features

| Feature | Details |
|---|---|
| **Generator-based scanning** | `FileScanner.scan()` is a generator — the full file list is never held in memory |
| **Streaming file reads** | `FileParser` reads files in configurable byte chunks |
| **Parallel file parsing** | `ThreadPoolExecutor` with configurable `max_workers` |
| **Batched embedding** | Chunks are embedded in batches of `batch_embed_size` |
| **Embedding cache** | Per-text SHA-256 cache; re-indexing unchanged files costs zero GPU time |
| **Incremental indexing** | Content-hash registry skips already-indexed files on re-runs |
| **Shallow git clones** | `depth=1` clones only the latest snapshot |

---

## Supported File Types

| Extension | Type | Notes |
|---|---|---|
| `.tf` | Terraform | HCL syntax; kept as raw text |
| `.yaml` / `.yml` | YAML | Validated with PyYAML before indexing |
| `.json` | JSON | Validated with stdlib `json` |

---

## Running Tests

```bash
pip install pytest pytest-cov faiss-cpu
pytest tests.py -v
pytest tests.py -v --cov=. --cov-report=term-missing
```

---

## Optional: Package Structure

Run `setup_dirs.py` to reorganise flat modules into a proper Python package:

```bash
python setup_dirs.py
```

This creates:

```
src/
├── scanner/   file_scanner.py  git_ingestor.py
├── parser/    file_parser.py
├── rag/       chunker.py  embedding_generator.py  vector_store_manager.py
├── llm/       local_llm_client.py
└── pipeline/  rag_pipeline.py
```

---

## Security Prompt Template

The LLM is instructed to:

1. Identify every vulnerability / misconfiguration in the retrieved context
2. Assign **CRITICAL / HIGH / MEDIUM / LOW / INFO** severity
3. Quote the exact file path and offending code snippet
4. Provide a concrete remediation recommendation
5. Map findings to CWE / OWASP categories where applicable
6. Explicitly state "No issues detected" rather than hallucinating findings

---

## License

MIT
