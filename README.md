# MisconfigGuard RAG Pipeline

MisconfigGuard provides a routed Retrieval-Augmented Generation pipeline for IaC
and configuration security analysis. The system reads the user's query,
classifies the intent, and routes the request to the right analyzer module or
to the general RAG path.

## What This README Covers

This README is intentionally limited to the routed RAG flow:

1. How content is ingested into the vector index.
2. How a query is classified and routed.
3. How to run the routed RAG pipeline from the CLI.
4. How to use the routed query API from Python.
5. Which configuration controls the routing behavior.

## Project Layout

The repo now uses a layered layout without breaking the legacy root entry points:

```text
config/
  config.yaml
  policy.yaml
tests/
  test_suite.py
misconfigguard/
  analysis/
  parsing/
  rag/
  scanning/
  config/
  models/
  services/
```

The root-level files remain as compatibility shims for imports and CLI usage.

## Routed RAG Architecture

```text
FileScanner -> FileParser -> Chunker -> EmbeddingGenerator -> VectorStoreManager
                                                              |
query -> QueryRouter -> QueryDispatcher ----------------------+
                         |                    |
                         |                    -> RAGOrchestrator -> ContextBuilder -> LocalLLMClient
                         |
                         +-> IAMSecurityAnalyzer
                         +-> WorkloadIdentitySecurityAnalyzer
                         +-> HardcodedSecretsAnalyzer
                         +-> PromptInjectionAnalyzer
```

## Query Flow

For every routed query, the pipeline does this:

1. Loads the indexed chunks from the vector store.
2. Reads the user's natural-language question.
3. Classifies the question with `QueryRouter`.
4. Routes the query with `QueryDispatcher`.
5. Executes the matching analyzer module or the general RAG orchestrator.
6. Returns a normalized result payload with intent, analysis, findings, and sources.

## Routed Intents

The current router can classify queries into these intents:

| Intent | Routed Module | Use For |
|---|---|---|
| `iam` | `IAMSecurityAnalyzer` | Managed identities, RBAC, roles, permissions |
| `workload_identity` | `WorkloadIdentitySecurityAnalyzer` | OIDC federation, trust, issuer, audience, subject |
| `secrets` | `HardcodedSecretsAnalyzer` | Passwords, tokens, API keys, connection strings, secret exposure |
| `prompt_injection` | `PromptInjectionAnalyzer` | CI/CD workflow abuse, prompt injection, unsafe AI inputs |
| `network` | `RAGOrchestrator` with network intent hint | Public access, ingress, egress, firewall, 0.0.0.0/0 |
| `compliance` | `RAGOrchestrator` with compliance intent hint | CIS, benchmark, policy, standards questions |
| `general_security` | `RAGOrchestrator` | Queries that do not map to a specialized analyzer |

## Prerequisites

### Python dependencies

```powershell
python -m pip install -r requirements.txt
```

### Local LLM runtime

If you want LLM-backed query routing and analysis, start Ollama:

```powershell
ollama pull llama3
ollama serve
```

If Ollama is not available, routed analysis can still fall back to keyword
classification and retrieval-only behavior in the legacy analyze path.

## Quick Start

### 1. Ingest a local folder

Build the vector index from a deployment or IaC directory:

```powershell
python cli.py scan-local "C:\path\to\deployment-folder"
```

Example:

```powershell
python cli.py scan-local "C:\workspace\terraform-on-azure-cloud-main"
```

### 2. Ask routed security questions

After ingestion, query the saved index:

```powershell
python cli.py query "Are there any over-privileged managed identities?"
python cli.py query "Check workload identity federation trust issues"
python cli.py query "Find hardcoded API keys or passwords"
python cli.py query "Any prompt injection risks in GitHub Actions YAML?"
python cli.py query "Are security groups open to 0.0.0.0/0?"
python cli.py query "Summarise all security findings"
```

The CLI prints the selected route before the analysis:

```text
[Routed to module: iam]
```

### 3. Use keyword-only routing

To disable LLM-based classification and use keyword heuristics only:

```powershell
python cli.py query "Find hardcoded secrets" --no-llm-routing
```

### 4. Request structured output

```powershell
python cli.py query "Check for public access issues" --structured
```

## Recommended Usage Pattern

1. Install dependencies.
2. Start Ollama if you want LLM routing and LLM analysis.
3. Run `python cli.py scan-local "<folder>"` or `python cli.py scan-repo <repo>`.
4. Run `python cli.py query "<question>"`.
5. Review the routed intent, analysis, findings, and sources.

## CLI Commands for Routed RAG

### Scan a local directory

```powershell
python cli.py scan-local ./my-terraform-project
```

### Scan a Git repository

```powershell
python cli.py scan-repo https://github.com/org/infra-repo
```

### Scan a private Git repository

```powershell
python cli.py scan-repo https://github.com/org/private-infra --token ghp_your_personal_access_token --branch main
```

### Query the routed RAG pipeline

```powershell
python cli.py query "Are there any hardcoded credentials or API secrets?"
python cli.py query "Check for open security groups (0.0.0.0/0)" --top-k 5
python cli.py query "Any prompt injection risks in CI workflow YAML?"
python cli.py query "Check RBAC over-permission" --no-llm-routing
python cli.py query "Summarise all critical findings" --stream
```

## Python API

Use `pipeline.query(...)` as the main routed RAG entry point.

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
    incremental=True,
    max_workers=4,
)

pipeline.ingest_directory("./terraform-project")

result = pipeline.query("Are there any hardcoded passwords?", top_k=5)
print(result["intent"])
print(result["analysis"])

result = pipeline.query(
    "Check for open security groups",
    top_k=5,
    use_llm_routing=False,
)
```

The routed result payload can include:

1. `intent`
2. `query`
3. `analysis`
4. `findings`
5. `issues`
6. `sources`
7. `results`
8. `context`
9. `summary`

## Routing Configuration

The main routing controls are in `config.yaml`:

```yaml
query_routing:
  enabled: true
  use_llm_routing: true
  routing_model: "llama3"
  routing_max_tokens: 20
  routing_cache_ttl: 300
  log_intent: true

retrieval:
  semantic_weight: 0.7
  keyword_weight: 0.3
  top_k_security: 3
  cache_ttl: 300
  max_context_tokens: 3000

rule_aware:
  enabled: true
  cache_ttl: 300
  include_generic_rules: true
```

### Routing options

1. `enabled`: turns routed query dispatch on or off.
2. `use_llm_routing`: allows the router to ask the local LLM to classify the query.
3. `routing_model`: the model used only for query classification.
4. `routing_max_tokens`: keeps the classification call small and cheap.
5. `routing_cache_ttl`: caches repeated query classifications.
6. `log_intent`: writes the selected route to the logs.

## Rule-Aware Retrieval

The general RAG path does more than plain vector similarity search.

1. `ResourceTagger` tags retrieved code chunks with `resource_type`, `cloud_provider`, and `category`.
2. `RuleAwareRetriever` narrows security guidance to the matching resource family.
3. `ContextBuilder` injects matched-resource context into the LLM prompt.
4. The orchestrator can enrich findings with `rule_id` and rule description metadata.

## Troubleshooting

1. `No vector index found`: run `python cli.py scan-local "<folder>"` first.
2. Slow first run: the embedding model may download the first time it is used.
3. Wrong route selected: retry with `--no-llm-routing` to compare keyword routing behavior.
4. No LLM response: confirm `ollama serve` is running and the configured model is available.

## Core Files

| File | Purpose |
|---|---|
| `rag_pipeline.py` | Main ingestion and routed query entry point |
| `query_router.py` | Query intent classification |
| `query_dispatcher.py` | Intent-to-analyzer dispatch |
| `rag_orchestrator.py` | General RAG analysis path |
| `rule_aware_retriever.py` | Resource-aware security rule retrieval |
| `context_builder.py` | Prompt assembly for routed analysis |
| `cli.py` | CLI entry points for scan and query |

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
