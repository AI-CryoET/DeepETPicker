"""Reversion tests for the ResidualUNet3D forward pass.

The architecture code was untouched by the refactors this suite guards
against, so these tests' job is to pin it down so a future refactor can't
change it unnoticed.

Runs once per fixture config, so the production config's image-pyramid
(use_IP), coord-conv (use_coord) and softmax branches are exercised, not
just the minimal core path.
"""
import numpy as np
import pytest
import torch

from tests.conftest import GOLDEN_DIR, requires_cuda
from tests.trajectory import build_model_input, build_seeded_model


def test_model_forward_matches_golden(config, args):
    golden = np.load(GOLDEN_DIR / f"model_forward_{config}.npy")

    model = build_seeded_model(args)
    x = build_model_input(args)
    with torch.no_grad():
        out = model(x).numpy()

    assert out.shape == golden.shape
    np.testing.assert_allclose(out, golden, rtol=1e-4, atol=1e-6)


def test_model_output_channels_match_num_classes(config, args):
    model = build_seeded_model(args)
    with torch.no_grad():
        out = model(build_model_input(args))
    assert out.shape[1] == args.num_classes


def test_model_init_state_matches_golden(config, args):
    """Seeded weight initialization must be reproducible -- otherwise the
    forward-pass and trajectory goldens would drift for reasons unrelated to
    the code under test."""
    golden_state = torch.load(GOLDEN_DIR / f"model_init_state_{config}.pt",
                               map_location="cpu")

    state = build_seeded_model(args).state_dict()

    assert set(state.keys()) == set(golden_state.keys())
    for key, tensor in state.items():
        np.testing.assert_allclose(
            tensor.numpy(), golden_state[key].numpy(), rtol=1e-6, atol=1e-8,
            err_msg=f"init weights diverged for {key}")


@pytest.mark.gpu
@requires_cuda
def test_model_forward_on_cuda_matches_cpu_golden(config, args):
    """Looser than the CPU check: cuDNN kernels diverge from CPU more than
    CPU-to-CPU version churn does. Catches gross device-handling regressions
    (crash, NaN, wrong device), not bit-exact CPU/GPU equivalence."""
    golden = np.load(GOLDEN_DIR / f"model_forward_{config}.npy")

    model = build_seeded_model(args).cuda()
    x = build_model_input(args).cuda()
    with torch.no_grad():
        out = model(x).cpu().numpy()

    assert np.all(np.isfinite(out))
    np.testing.assert_allclose(out, golden, rtol=1e-2, atol=1e-4)
