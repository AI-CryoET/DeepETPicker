"""Reversion tests for the end-to-end training math.

Runs a manual 10-step loop (dataset -> model -> DiceLoss -> AdamW) with no
Lightning Trainer, so a Lightning version bump can't make this test noisy.
This is the closest thing in the suite to "training still behaves the same".

Runs once per fixture config, so the production config's real flags are
covered alongside the minimal core path.
"""
import pytest

from tests.conftest import load_golden_json, requires_cuda
from tests.fixtures import SEED
from tests.trajectory import run_training_trajectory


def test_training_trajectory_matches_golden(config, cfg, args):
    golden = load_golden_json(f"train_trajectory_{config}.json")["losses"]

    losses = run_training_trajectory(cfg, args, n_steps=10, batch_size=2,
                                      seed=SEED, device="cpu")

    assert len(losses) == len(golden)
    for step, (actual, expected) in enumerate(zip(losses, golden)):
        assert actual == pytest.approx(expected, rel=1e-3), f"diverged at step {step}"


def test_training_trajectory_decreases(cfg, args):
    """Sanity guard independent of the golden values: catches a training loop
    that has stopped learning entirely (dead gradients, frozen weights)."""
    losses = run_training_trajectory(cfg, args, n_steps=10, batch_size=2,
                                      seed=SEED, device="cpu")
    assert losses[-1] < losses[0]


@pytest.mark.gpu
@requires_cuda
def test_training_trajectory_on_cuda(config, cfg, args):
    """Loosest tolerance in the suite: primarily catches gross regressions
    (crash, NaN, tensors stranded on the wrong device), not tight numerical
    agreement with the CPU golden."""
    golden = load_golden_json(f"train_trajectory_{config}.json")["losses"]

    losses = run_training_trajectory(cfg, args, n_steps=10, batch_size=2,
                                      seed=SEED, device="cuda")

    assert len(losses) == len(golden)
    for step, (actual, expected) in enumerate(zip(losses, golden)):
        assert actual == pytest.approx(expected, rel=5e-2), f"diverged at step {step}"
