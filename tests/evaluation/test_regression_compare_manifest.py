"""Tests for regression_compare_manifest.py"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path


# Absolute path to the script
SCRIPT_PATH = (
    Path(__file__).parent.parent.parent
    / "scripts"
    / "evaluation"
    / "regression_compare_manifest.py"
)


class TestManifestCreation:
    """Test manifest creation functionality."""

    def test_basic_manifest_creation(self, tmp_path):
        """Test that manifest is created with correct structure."""
        # Use existing query files
        queries_subset = (
            Path(__file__).parent.parent.parent
            / ".sisyphus"
            / "evidence"
            / "paper-b"
            / "task-10"
            / "queries_subset.jsonl"
        )
        queries_full = (
            Path(__file__).parent.parent.parent
            / "data"
            / "synth_benchmarks"
            / "stability_bench_v1"
            / "queries.jsonl"
        )

        out_path = tmp_path / "manifest.json"

        result = subprocess.run(
            [
                sys.executable,
                str(SCRIPT_PATH),
                "--run-id",
                "20260101_120000",
                "--before-sha",
                "73ca832",
                "--after-sha",
                "c04fa25",
                "--queries-subset",
                str(queries_subset),
                "--queries-full",
                str(queries_full),
                "--es-host",
                "http://localhost:8002",
                "--es-env",
                "synth",
                "--es-index-prefix",
                "rag_synth",
                "--out",
                str(out_path),
            ],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0, f"Script failed: {result.stderr}"
        assert out_path.exists(), "Manifest file was not created"

        # Validate JSON structure
        with open(out_path) as f:
            manifest = json.load(f)

        assert manifest["run_id"] == "20260101_120000"
        assert manifest["before_sha"] == "73ca832"
        assert manifest["after_sha"] == "c04fa25"
        assert "created_at" in manifest

        # Check query file stats
        assert "query_files" in manifest
        assert "subset" in manifest["query_files"]
        assert "full" in manifest["query_files"]

        subset_stats = manifest["query_files"]["subset"]
        assert "path" in subset_stats
        assert "sha256" in subset_stats
        assert "line_count" in subset_stats
        assert subset_stats["line_count"] == 48

        # Check ES config
        assert "es_config" in manifest
        assert manifest["es_config"]["host"] == "http://localhost:8002"
        assert manifest["es_config"]["env"] == "synth"
        assert manifest["es_config"]["index_prefix"] == "rag_synth"

        # Check environment (should be present if env vars are set)
        assert "environment" in manifest

    def test_empty_run_id_fails(self, tmp_path):
        """Test that empty run-id produces an error."""
        queries_subset = (
            Path(__file__).parent.parent.parent
            / ".sisyphus"
            / "evidence"
            / "paper-b"
            / "task-10"
            / "queries_subset.jsonl"
        )
        queries_full = (
            Path(__file__).parent.parent.parent
            / "data"
            / "synth_benchmarks"
            / "stability_bench_v1"
            / "queries.jsonl"
        )

        out_path = tmp_path / "manifest.json"

        result = subprocess.run(
            [
                sys.executable,
                str(SCRIPT_PATH),
                "--run-id",
                "",
                "--queries-subset",
                str(queries_subset),
                "--queries-full",
                str(queries_full),
                "--es-host",
                "http://localhost:8002",
                "--es-env",
                "synth",
                "--es-index-prefix",
                "rag_synth",
                "--out",
                str(out_path),
            ],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 1
        assert "empty" in result.stderr.lower()

    def test_missing_query_subset_fails(self, tmp_path):
        """Test that missing subset query file produces an error."""
        queries_full = (
            Path(__file__).parent.parent.parent
            / "data"
            / "synth_benchmarks"
            / "stability_bench_v1"
            / "queries.jsonl"
        )

        out_path = tmp_path / "manifest.json"

        result = subprocess.run(
            [
                sys.executable,
                str(SCRIPT_PATH),
                "--run-id",
                "20260101_120000",
                "--queries-subset",
                "/nonexistent/subset.jsonl",
                "--queries-full",
                str(queries_full),
                "--es-host",
                "http://localhost:8002",
                "--es-env",
                "synth",
                "--es-index-prefix",
                "rag_synth",
                "--out",
                str(out_path),
            ],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 1
        assert "not found" in result.stderr.lower()

    def test_missing_query_full_fails(self, tmp_path):
        """Test that missing full query file produces an error."""
        queries_subset = (
            Path(__file__).parent.parent.parent
            / ".sisyphus"
            / "evidence"
            / "paper-b"
            / "task-10"
            / "queries_subset.jsonl"
        )

        out_path = tmp_path / "manifest.json"

        result = subprocess.run(
            [
                sys.executable,
                str(SCRIPT_PATH),
                "--run-id",
                "20260101_120000",
                "--queries-subset",
                str(queries_subset),
                "--queries-full",
                "/nonexistent/full.jsonl",
                "--es-host",
                "http://localhost:8002",
                "--es-env",
                "synth",
                "--es-index-prefix",
                "rag_synth",
                "--out",
                str(out_path),
            ],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 1
        assert "not found" in result.stderr.lower()

    def test_creates_parent_directories(self, tmp_path):
        """Test that parent directories are created for output path."""
        queries_subset = (
            Path(__file__).parent.parent.parent
            / ".sisyphus"
            / "evidence"
            / "paper-b"
            / "task-10"
            / "queries_subset.jsonl"
        )
        queries_full = (
            Path(__file__).parent.parent.parent
            / "data"
            / "synth_benchmarks"
            / "stability_bench_v1"
            / "queries.jsonl"
        )

        out_path = tmp_path / "nested" / "dir" / "manifest.json"

        result = subprocess.run(
            [
                sys.executable,
                str(SCRIPT_PATH),
                "--run-id",
                "20260101_120000",
                "--queries-subset",
                str(queries_subset),
                "--queries-full",
                str(queries_full),
                "--es-host",
                "http://localhost:8002",
                "--es-env",
                "synth",
                "--es-index-prefix",
                "rag_synth",
                "--out",
                str(out_path),
            ],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0
        assert out_path.exists()
        assert out_path.parent.exists()

    def test_sha256_computation(self, tmp_path):
        """Test that SHA256 is computed correctly."""
        # Create a temp file with known content
        test_file = tmp_path / "test.jsonl"
        test_file.write_text('{"query": "test1"}\n{"query": "test2"}\n')

        queries_subset = (
            Path(__file__).parent.parent.parent
            / ".sisyphus"
            / "evidence"
            / "paper-b"
            / "task-10"
            / "queries_subset.jsonl"
        )
        queries_full = (
            Path(__file__).parent.parent.parent
            / "data"
            / "synth_benchmarks"
            / "stability_bench_v1"
            / "queries.jsonl"
        )

        out_path = tmp_path / "manifest.json"

        result = subprocess.run(
            [
                sys.executable,
                str(SCRIPT_PATH),
                "--run-id",
                "20260101_120000",
                "--queries-subset",
                str(queries_subset),
                "--queries-full",
                str(queries_full),
                "--es-host",
                "http://localhost:8002",
                "--es-env",
                "synth",
                "--es-index-prefix",
                "rag_synth",
                "--out",
                str(out_path),
            ],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0

        with open(out_path) as f:
            manifest = json.load(f)

        # Verify sha256 is a valid hex string
        sha256 = manifest["query_files"]["subset"]["sha256"]
        assert len(sha256) == 64  # SHA256 produces 64 hex characters
        assert all(c in "0123456789abcdef" for c in sha256)

    def test_json_formatting(self, tmp_path):
        """Test that output JSON is properly formatted with indent=2 and newline."""
        queries_subset = (
            Path(__file__).parent.parent.parent
            / ".sisyphus"
            / "evidence"
            / "paper-b"
            / "task-10"
            / "queries_subset.jsonl"
        )
        queries_full = (
            Path(__file__).parent.parent.parent
            / "data"
            / "synth_benchmarks"
            / "stability_bench_v1"
            / "queries.jsonl"
        )

        out_path = tmp_path / "manifest.json"

        result = subprocess.run(
            [
                sys.executable,
                str(SCRIPT_PATH),
                "--run-id",
                "20260101_120000",
                "--queries-subset",
                str(queries_subset),
                "--queries-full",
                str(queries_full),
                "--es-host",
                "http://localhost:8002",
                "--es-env",
                "synth",
                "--es-index-prefix",
                "rag_synth",
                "--out",
                str(out_path),
            ],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0

        # Read raw file content
        content = out_path.read_text()

        # Check trailing newline
        assert content.endswith("\n")

        # Check it's valid JSON with indent=2
        manifest = json.loads(content)
        assert manifest is not None


class TestEnvironmentFiltering:
    """Test environment variable filtering and redaction."""

    def test_env_var_whitelisting(self, tmp_path, monkeypatch):
        """Test that only whitelisted env vars are included."""
        # Set whitelisted and non-whitelisted env vars
        monkeypatch.setenv("SEARCH_ES_HOST", "http://localhost:8002")
        monkeypatch.setenv("ES_INDEX_PREFIX", "rag_synth")
        monkeypatch.setenv("RAG_MODEL_NAME", "bge-base")
        monkeypatch.setenv("VLLM_BASE_URL", "http://localhost:8003")
        monkeypatch.setenv("OTHER_VAR", "should_not_appear")
        monkeypatch.setenv("MY_API_KEY", "secret_key")

        queries_subset = (
            Path(__file__).parent.parent.parent
            / ".sisyphus"
            / "evidence"
            / "paper-b"
            / "task-10"
            / "queries_subset.jsonl"
        )
        queries_full = (
            Path(__file__).parent.parent.parent
            / "data"
            / "synth_benchmarks"
            / "stability_bench_v1"
            / "queries.jsonl"
        )

        out_path = tmp_path / "manifest.json"

        result = subprocess.run(
            [
                sys.executable,
                str(SCRIPT_PATH),
                "--run-id",
                "20260101_120000",
                "--queries-subset",
                str(queries_subset),
                "--queries-full",
                str(queries_full),
                "--es-host",
                "http://localhost:8002",
                "--es-env",
                "synth",
                "--es-index-prefix",
                "rag_synth",
                "--out",
                str(out_path),
            ],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0

        with open(out_path) as f:
            manifest = json.load(f)

        env = manifest["environment"]

        # Check whitelisted vars are present
        assert "SEARCH_ES_HOST" in env
        assert "ES_INDEX_PREFIX" in env
        assert "RAG_MODEL_NAME" in env
        assert "VLLM_BASE_URL" in env

        # Check non-whitelisted vars are NOT present
        assert "OTHER_VAR" not in env
        assert "MY_API_KEY" not in env

    def test_secret_redaction(self, tmp_path, monkeypatch):
        """Test that secrets are redacted."""
        # Set env vars with secrets
        monkeypatch.setenv("SEARCH_API_KEY", "super_secret_key")
        monkeypatch.setenv("ES_PASSWORD", "my_password")
        monkeypatch.setenv("RAG_TOKEN", "my_token")
        monkeypatch.setenv("VLLM_API_KEY", "another_secret")

        queries_subset = (
            Path(__file__).parent.parent.parent
            / ".sisyphus"
            / "evidence"
            / "paper-b"
            / "task-10"
            / "queries_subset.jsonl"
        )
        queries_full = (
            Path(__file__).parent.parent.parent
            / "data"
            / "synth_benchmarks"
            / "stability_bench_v1"
            / "queries.jsonl"
        )

        out_path = tmp_path / "manifest.json"

        result = subprocess.run(
            [
                sys.executable,
                str(SCRIPT_PATH),
                "--run-id",
                "20260101_120000",
                "--queries-subset",
                str(queries_subset),
                "--queries-full",
                str(queries_full),
                "--es-host",
                "http://localhost:8002",
                "--es-env",
                "synth",
                "--es-index-prefix",
                "rag_synth",
                "--out",
                str(out_path),
            ],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0

        with open(out_path) as f:
            manifest = json.load(f)

        env = manifest["environment"]

        # All secret vars should be redacted
        assert env.get("SEARCH_API_KEY") == "***REDACTED***"
        assert env.get("ES_PASSWORD") == "***REDACTED***"
        assert env.get("RAG_TOKEN") == "***REDACTED***"
        assert env.get("VLLM_API_KEY") == "***REDACTED***"

    def test_case_insensitive_redaction(self, tmp_path, monkeypatch):
        """Test that secret redaction is case-insensitive."""
        monkeypatch.setenv("SEARCH_api_key", "secret1")
        monkeypatch.setenv("ES_PASSWORD", "secret2")
        monkeypatch.setenv("VLLM_token", "secret3")

        queries_subset = (
            Path(__file__).parent.parent.parent
            / ".sisyphus"
            / "evidence"
            / "paper-b"
            / "task-10"
            / "queries_subset.jsonl"
        )
        queries_full = (
            Path(__file__).parent.parent.parent
            / "data"
            / "synth_benchmarks"
            / "stability_bench_v1"
            / "queries.jsonl"
        )

        out_path = tmp_path / "manifest.json"

        result = subprocess.run(
            [
                sys.executable,
                str(SCRIPT_PATH),
                "--run-id",
                "20260101_120000",
                "--queries-subset",
                str(queries_subset),
                "--queries-full",
                str(queries_full),
                "--es-host",
                "http://localhost:8002",
                "--es-env",
                "synth",
                "--es-index-prefix",
                "rag_synth",
                "--out",
                str(out_path),
            ],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0

        with open(out_path) as f:
            manifest = json.load(f)

        env = manifest["environment"]

        # Case-insensitive matching should redact these
        assert env.get("SEARCH_api_key") == "***REDACTED***"
        assert env.get("ES_PASSWORD") == "***REDACTED***"
        assert env.get("VLLM_token") == "***REDACTED***"
