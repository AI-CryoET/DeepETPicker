"""Shared pytest configuration for the reversion test suite.

Deliberately does NOT import train.py anywhere: it calls
torch.set_float32_matmul_precision('high') at module import time in current
code (the baseline doesn't), which would silently make current-vs-baseline
comparisons apples-to-oranges, and it pulls in the full Lightning Trainer
stack this suite intentionally bypasses.
"""
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from tests.fixtures import CONFIG_NAMES, build_args, build_cfg, build_fixture_dataset

REPO_ROOT = Path(__file__).resolve().parent.parent
GOLDEN_DIR = REPO_ROOT / "tests" / "golden"
FIXTURES_TMP = REPO_ROOT / "tests" / "fixtures_tmp"


@pytest.fixture(scope="session", autouse=True)
def _deterministic_algorithms():
    """Session-wide determinism. Affects algorithm selection, not values --
    per-test value reproducibility comes from explicit local seeding."""
    torch.use_deterministic_algorithms(True)
    yield
    torch.use_deterministic_algorithms(False)


@pytest.fixture(params=CONFIG_NAMES)
def config(request):
    """Runs every dependent test once per fixture config:

      core       -- minimal, pins the shared path all configs route through
      production -- mirrors train_AuNPs.py's flags (image pyramid, coord
                    conv, softmax, 3 classes)
    """
    return request.param


@pytest.fixture
def fixture_root(request):
    """Repo-local scratch dir (not tmp_path) so a failing test's generated
    mrc/coords files stay inspectable by hand afterward."""
    root = FIXTURES_TMP / request.node.name.replace("/", "_")
    build_fixture_dataset(root)
    return root


@pytest.fixture
def cfg(fixture_root, config):
    return build_cfg(fixture_root, config)


@pytest.fixture
def args(config):
    return build_args(config)


def load_golden_npz(name):
    return np.load(GOLDEN_DIR / name)


def load_golden_json(name):
    with open(GOLDEN_DIR / name) as f:
        return json.load(f)


requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="no CUDA device available")
