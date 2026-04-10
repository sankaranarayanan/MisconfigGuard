"""Compatibility test entry point for the relocated test suite."""

from pathlib import Path


exec((Path(__file__).with_name("tests") / "test_suite.py").read_text(encoding="utf-8"))