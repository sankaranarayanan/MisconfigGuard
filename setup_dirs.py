"""
setup_dirs.py — Scaffold the MisconfigGuard package directory structure.

Running this script reorganises the flat module files into a proper
Python package layout suitable for larger projects or pip-installable
distribution:

    MisconfigGuard/
    ├── src/
    │   ├── __init__.py
    │   ├── scanner/
    │   │   ├── __init__.py
    │   │   ├── file_scanner.py
    │   │   └── git_ingestor.py
    │   ├── parser/
    │   │   ├── __init__.py
    │   │   └── file_parser.py
    │   ├── rag/
    │   │   ├── __init__.py
    │   │   ├── chunker.py
    │   │   ├── embedding_generator.py
    │   │   └── vector_store_manager.py
    │   ├── llm/
    │   │   ├── __init__.py
    │   │   └── local_llm_client.py
    │   └── pipeline/
    │       ├── __init__.py
    │       └── rag_pipeline.py
    ├── config/
    │   └── config.yaml
    ├── tests/
    │   └── tests.py
    ├── cache/          (runtime artefacts — gitignored)
    ├── config.yaml     (root copy for convenience)
    ├── cli.py
    ├── main.py
    └── requirements.txt

Usage:
    python setup_dirs.py
"""

import os
import shutil

# ---------------------------------------------------------------------------
# Directory layout
# ---------------------------------------------------------------------------

DIRS = [
    "src/scanner",
    "src/parser",
    "src/rag",
    "src/llm",
    "src/pipeline",
    "config",
    "tests",
    "cache",
    "cache/embeddings",
    "tmp/repos",
    "logs",
]

# source_file → destination_path (relative to project root)
FILE_MOVES = {
    "file_scanner.py":        "src/scanner/file_scanner.py",
    "git_ingestor.py":        "src/scanner/git_ingestor.py",
    "file_parser.py":         "src/parser/file_parser.py",
    "chunker.py":             "src/rag/chunker.py",
    "embedding_generator.py": "src/rag/embedding_generator.py",
    "vector_store_manager.py":"src/rag/vector_store_manager.py",
    "local_llm_client.py":    "src/llm/local_llm_client.py",
    "rag_pipeline.py":        "src/pipeline/rag_pipeline.py",
    "config.yaml":            "config/config.yaml",
    "tests.py":               "tests/tests.py",
}

# Packages that need an __init__.py
PACKAGES = [
    "src",
    "src/scanner",
    "src/parser",
    "src/rag",
    "src/llm",
    "src/pipeline",
]

# ---------------------------------------------------------------------------
# .gitignore content
# ---------------------------------------------------------------------------

GITIGNORE = """\
# Runtime cache
cache/
tmp/
logs/
*.pkl
*.faiss

# Python
__pycache__/
*.pyc
*.pyo
.pytest_cache/
.coverage
dist/
build/
*.egg-info/

# Environment
.env
.venv/
venv/
"""

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    base = os.path.dirname(os.path.abspath(__file__))

    print("📁  Creating directories...")
    for d in DIRS:
        path = os.path.join(base, d)
        os.makedirs(path, exist_ok=True)
        print(f"   ✓  {d}/")

    print("\n📦  Creating package __init__.py files...")
    for pkg in PACKAGES:
        init = os.path.join(base, pkg, "__init__.py")
        if not os.path.exists(init):
            open(init, "w").close()
            print(f"   ✓  {pkg}/__init__.py")

    print("\n📄  Moving module files into package structure...")
    for src_name, dest_rel in FILE_MOVES.items():
        src = os.path.join(base, src_name)
        dest = os.path.join(base, dest_rel)
        if os.path.exists(src) and not os.path.exists(dest):
            shutil.move(src, dest)
            print(f"   ✓  {src_name}  →  {dest_rel}")
        elif not os.path.exists(src):
            print(f"   ⚠  {src_name} not found (skipped)")
        else:
            print(f"   –  {dest_rel} already exists (skipped)")

    print("\n📝  Writing .gitignore...")
    gitignore_path = os.path.join(base, ".gitignore")
    if not os.path.exists(gitignore_path):
        with open(gitignore_path, "w") as f:
            f.write(GITIGNORE)
        print("   ✓  .gitignore created")
    else:
        print("   –  .gitignore already exists (skipped)")

    print("\n✅  Package structure ready.")
    print(
        "\nNOTE: After reorganising, update import paths in cli.py / main.py "
        "to use the new package paths, e.g.:\n"
        "    from src.scanner.file_scanner import FileScanner\n"
        "    from src.rag.chunker import Chunker\n"
    )


if __name__ == "__main__":
    main()
