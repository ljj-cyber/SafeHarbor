"""SafeHarbor smoke test.

This script is intentionally **offline-friendly**: it only checks that the
repository is laid out correctly, the core Python modules import cleanly,
and (when the heavy artifacts are present) that the SafetyProjector
checkpoint and the Risk Tree pickle can be loaded end-to-end.

Run it as:

    python -m tests.test_smoke
    # or
    pytest tests/test_smoke.py -v

It does NOT contact any remote LLM API and does NOT require GPUs.
"""
from __future__ import annotations

import importlib
import os
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"

for path in (REPO_ROOT, SRC_DIR):
    p = str(path)
    if p not in sys.path:
        sys.path.insert(0, p)


# ---------------------------------------------------------------------------
# Layout / required files
# ---------------------------------------------------------------------------
REQUIRED_PATHS = [
    "agentharm.py",
    "proxy_server.py",
    "run_agentharm.sh",
    "scorer.py",
    "metric.py",
    "utils.py",
    "prompts.py",
    "requirements.txt",
    "README.md",
    "agents/agent.py",
    "agents/default_agent.py",
    "agents/guardagent_agent.py",
    "benchmark/__init__.py",
    "benchmark/harmful_tools",
    "benchmark/benign_tools",
    "src/risk_tree.py",
    "src/SafetyProjector.py",
    "src/memory_defender.py",
    "src/attacker.py",
    "src/llama_guard.py",
    "baselines/rag_baseline.py",
    "baselines/guardagent/guardagent.py",
    "baselines/guardagent/config.py",
    "A_mem/agentic_memory/memory_system.py",
    "Agent-SafetyBench/evaluation/eval.py",
    "Agent-SafetyBench/score/eval_with_shield.py",
]


# ---------------------------------------------------------------------------
# Lightweight import targets. Heavy modules (risk_tree, SafetyProjector,
# proxy_server) are exercised separately in their own test methods so a
# native crash inside torch/faiss does not bring down the rest of the suite.
# ---------------------------------------------------------------------------
IMPORT_TARGETS = [
    ("scorer", ["RefusalJudgeLLM", "combined_scorer"]),
    ("metric", ["avg_score", "avg_full_score", "avg_refusals"]),
    ("utils", ["load_dataset", "filter_dataset", "setup_tools_from_metadata"]),
    ("prompts", ["get_system_prompt", "get_jailbreak_template"]),
    ("agents.agent", ["get_agent", "AGENT_DICT"]),
    ("agents.default_agent", ["default_agent"]),
    ("agents.refusal_agent", ["refusal_agent"]),
    ("baselines.rag_baseline", ["init_RAG", "query_rag", "query_mem"]),
]


# ---------------------------------------------------------------------------
# Optional artifacts (skipped if missing — they are gitignored on purpose)
# ---------------------------------------------------------------------------
PKL_PATH = SRC_DIR / "final_memory_after_benign_calibration.pkl"
PROJECTOR_PATH = SRC_DIR / "models" / "safety_projector.pth"


class TestRepoLayout(unittest.TestCase):
    def test_required_paths_present(self):
        missing = [p for p in REQUIRED_PATHS if not (REPO_ROOT / p).exists()]
        self.assertEqual(
            missing,
            [],
            f"Missing required paths (relative to repo root):\n  - "
            + "\n  - ".join(missing),
        )

    def test_no_obvious_secrets_in_tracked_code(self):
        """Make sure we did not re-introduce hard-coded keys."""
        # Built dynamically so the test file itself never matches.
        bad_substrings = [
            "fk" + "3468961406",
            "sedU5Tw2" + "ibHgvj78hfE8GYxTEZOQ2TwL3240e317",
        ]
        scan_roots = [REPO_ROOT]
        skip_dirs = {"_deprecated", "tests", ".git", "__pycache__"}
        offenders = []
        for root in scan_roots:
            for f in root.rglob("*"):
                if not f.is_file() or f.suffix not in {".py", ".sh", ".md"}:
                    continue
                if any(part in skip_dirs for part in f.parts):
                    continue
                try:
                    text = f.read_text(encoding="utf-8")
                except UnicodeDecodeError:
                    continue
                for needle in bad_substrings:
                    if needle in text:
                        offenders.append(f"{f.relative_to(REPO_ROOT)} leaked '{needle}'")
        self.assertEqual(
            offenders, [], "Secrets re-introduced:\n" + "\n".join(offenders)
        )


def _import_in_subprocess(module_name: str) -> tuple[bool, str]:
    """Run ``import <module_name>`` in a fresh subprocess.

    This isolates native-extension crashes (e.g. faiss/torch loading order)
    so the rest of the suite keeps running.
    """
    import subprocess
    code = f"import sys; sys.path.insert(0, '{SRC_DIR}'); import {module_name}"
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
        timeout=120,
    )
    return proc.returncode == 0, (proc.stderr or proc.stdout).strip()


class TestCoreImports(unittest.TestCase):
    """Modules that need third-party deps are skipped (not failed) when those
    deps are absent, so the test is useful even on a bare interpreter."""

    def test_modules_import(self):
        for mod_name, attrs in IMPORT_TARGETS:
            with self.subTest(module=mod_name):
                try:
                    mod = importlib.import_module(mod_name)
                except ModuleNotFoundError as e:
                    self.skipTest(f"{mod_name} skipped (missing dep: {e.name})")
                except Exception as e:  # pragma: no cover - exercised by humans
                    self.fail(f"Failed to import {mod_name}: {e}")
                else:
                    for attr in attrs:
                        self.assertTrue(
                            hasattr(mod, attr),
                            f"{mod_name} is missing expected attribute '{attr}'",
                        )

    def test_proxy_server_imports(self):
        """proxy_server.py must be import-clean when memory is disabled.

        Run in a subprocess because importing it pulls in the full Memory-Tree
        + autogen + faiss + torch stack on the first call.
        """
        env_setup = (
            "import os; "
            "os.environ.pop('MEMORY_SYSTEM_TYPE', None); "
            "os.environ['ENABLE_LLAMA_GUARD'] = 'false'; "
            "os.environ['ENABLE_GUARDAGENT'] = 'False'; "
        )
        import subprocess
        proc = subprocess.run(
            [sys.executable, "-c", env_setup + "import proxy_server; "
             "assert hasattr(proxy_server, 'app'), 'flask app missing'"],
            capture_output=True, text=True, cwd=str(REPO_ROOT), timeout=180,
        )
        if proc.returncode != 0:
            stderr = (proc.stderr or "").strip()
            if "ModuleNotFoundError" in stderr:
                self.skipTest(f"proxy_server skipped (missing dep): {stderr.splitlines()[-1]}")
            self.fail(f"proxy_server import failed:\n{stderr}")

    def test_risk_tree_imports(self):
        """RiskTree pulls in torch + sentence-transformers; isolate in subprocess."""
        ok, err = _import_in_subprocess("risk_tree")
        if not ok:
            if "ModuleNotFoundError" in err:
                self.skipTest(f"risk_tree skipped (missing dep): {err.splitlines()[-1]}")
            self.fail(f"risk_tree import failed:\n{err}")

    def test_safety_projector_imports(self):
        ok, err = _import_in_subprocess("SafetyProjector")
        if not ok:
            if "ModuleNotFoundError" in err:
                self.skipTest(f"SafetyProjector skipped (missing dep): {err.splitlines()[-1]}")
            self.fail(f"SafetyProjector import failed:\n{err}")


class TestSafetyProjector(unittest.TestCase):
    """Exercise the SafetyProjector forward pass on synthetic inputs."""

    def test_forward_pass_synthetic(self):
        import subprocess
        code = f"""
import sys; sys.path.insert(0, '{SRC_DIR}')
import torch
from SafetyProjector import SafetyProjector
model = SafetyProjector(input_dim=384)
x = torch.randn(4, 384)
emb, logits = model(x)
assert emb.shape == (4, 128), emb.shape
assert logits.shape == (4, 1), logits.shape
print('OK')
"""
        proc = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True, text=True, cwd=str(REPO_ROOT), timeout=180,
        )
        if proc.returncode != 0:
            err = (proc.stderr or proc.stdout or "").strip()
            if "ModuleNotFoundError" in err:
                self.skipTest(f"torch not installed: {err.splitlines()[-1]}")
            self.fail(f"SafetyProjector forward pass failed:\n{err}")
        self.assertIn("OK", proc.stdout)


class TestPretrainedArtifacts(unittest.TestCase):
    """Loaded only when the heavy artifacts are present locally."""

    def test_safety_projector_checkpoint(self):
        if not PROJECTOR_PATH.exists():
            self.skipTest(
                f"{PROJECTOR_PATH.relative_to(REPO_ROOT)} not present — "
                "build it via 'python src/SafetyProjector.py'."
            )
        import subprocess
        code = f"""
import sys; sys.path.insert(0, '{SRC_DIR}')
import torch
from SafetyProjector import SafetyProjector
ckpt = torch.load(r'{PROJECTOR_PATH}', map_location='cpu')
assert 'model_state_dict' in ckpt
assert 'input_dim' in ckpt
m = SafetyProjector(input_dim=ckpt['input_dim'])
missing, _ = m.load_state_dict(ckpt['model_state_dict'], strict=False)
assert not any('classifier' in k for k in missing), f'classifier missing: {{missing}}'
print('OK')
"""
        proc = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True, text=True, cwd=str(REPO_ROOT), timeout=180,
        )
        if proc.returncode != 0:
            err = (proc.stderr or proc.stdout or "").strip()
            if "ModuleNotFoundError" in err:
                self.skipTest(f"torch not installed: {err.splitlines()[-1]}")
            self.fail(f"Safety Projector checkpoint failed to load:\n{err}")
        self.assertIn("OK", proc.stdout)

    def test_risk_tree_pickle(self):
        if not PKL_PATH.exists():
            self.skipTest(
                f"{PKL_PATH.relative_to(REPO_ROOT)} not present — "
                "build it via the training pipeline in README.md."
            )
        import subprocess
        code = f"""
import sys; sys.path.insert(0, '{SRC_DIR}')
from risk_tree import RiskTree
t = RiskTree().load(r'{PKL_PATH}')
assert hasattr(t, 'root')
assert hasattr(t, 'retrieve_query')
print('OK')
"""
        proc = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True, text=True, cwd=str(REPO_ROOT), timeout=300,
        )
        if proc.returncode != 0:
            err = (proc.stderr or proc.stdout or "").strip()
            if "ModuleNotFoundError" in err:
                self.skipTest(f"risk_tree dep missing: {err.splitlines()[-1]}")
            # SentenceTransformer needs to fetch ``all-MiniLM-L6-v2`` from
            # HuggingFace the first time. If we are offline / behind a
            # restrictive proxy, skip rather than fail – the test is about
            # whether the pkl is structurally loadable, not about HF reachability.
            network_markers = (
                "ProxyError",
                "ConnectionError",
                "Max retries exceeded",
                "ConnectTimeout",
                "Could not reach huggingface.co",
                "Tunnel connection failed",
            )
            if any(m in err for m in network_markers):
                self.skipTest(
                    "Skipping pkl load test: SentenceTransformer base model "
                    "could not be downloaded (offline / proxy)."
                )
            self.fail(f"Risk Tree pickle failed to load:\n{err}")
        self.assertIn("OK", proc.stdout)


def main():
    unittest.main(module=__name__, verbosity=2)


if __name__ == "__main__":
    main()
