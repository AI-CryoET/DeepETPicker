"""Reversion tests for DiceLoss and seg_metrics.

DiceLoss is checked against a golden value captured from the baseline.

seg_metrics is checked against an INDEPENDENT NumPy reference oracle rather
than a golden snapshot, because at the baseline commit seg_metrics called
torch.zeros(1).cuda() unconditionally and therefore could not run on CPU at
all -- there is no CPU baseline value to freeze. The current
`device=y_pred.device` form is a strict fix for that, and a from-scratch
reference implementation is a stronger check than a snapshot anyway: it
validates the metric definitions themselves, so e.g. a swapped iou/f1 return
order is caught.
"""
import numpy as np
import pytest
import torch

from tests.conftest import load_golden_json, requires_cuda
from tests.trajectory import build_metrics_tensors
from utils.loss import DiceLoss
from utils.metrics import seg_metrics

SMOOTH = 1e-7


def reference_seg_metrics(pred, target, threshold=0.5, smooth=SMOOTH):
    """Independent NumPy implementation of seg_metrics, per-channel then
    averaged. Returns (precision, recall, iou, f1) -- the order
    utils/metrics.py actually returns them in.
    """
    y_pred = (pred.numpy() >= threshold).astype(np.float64)
    y_true = target.numpy().astype(np.float64)

    # flatten channel-first: (N, C, D, H, W) -> (C, N*D*H*W)
    c = y_pred.shape[1]
    y_pred = np.moveaxis(y_pred, 1, 0).reshape(c, -1)
    y_true = np.moveaxis(y_true, 1, 0).reshape(c, -1)

    tp = (y_true * y_pred).sum(-1)
    fp = ((1 - y_true) * y_pred).sum(-1)
    fn = (y_true * (1 - y_pred)).sum(-1)

    precision = (tp + smooth) / (tp + fp + smooth)
    recall = (tp + smooth) / (tp + fn + smooth)
    iou = (tp + smooth) / (tp + fn + fp + smooth)
    f1 = 2 * (precision * recall) / (precision + recall + smooth)

    return precision.mean(), recall.mean(), iou.mean(), f1.mean()


def test_dice_loss_matches_golden(config, args):
    golden = load_golden_json(f"metrics_{config}.json")
    pred, target = build_metrics_tensors(channels=args.num_classes)

    loss = DiceLoss(args=args)(pred, target).item()

    assert loss == pytest.approx(golden["dice_loss"], rel=1e-6, abs=1e-8)


@pytest.mark.parametrize("channels", [1, 3])
def test_seg_metrics_matches_numpy_reference(channels):
    pred, target = build_metrics_tensors(channels=channels)

    precision, recall, iou, f1 = seg_metrics(pred, target, threshold=0.5)
    ref_precision, ref_recall, ref_iou, ref_f1 = reference_seg_metrics(pred, target)

    assert float(precision) == pytest.approx(ref_precision, rel=1e-5)
    assert float(recall) == pytest.approx(ref_recall, rel=1e-5)
    assert float(iou) == pytest.approx(ref_iou, rel=1e-5)
    assert float(f1) == pytest.approx(ref_f1, rel=1e-5)


@pytest.mark.parametrize("channels", [1, 3])
def test_seg_metrics_runs_on_cpu(channels):
    """Regression guard for the .cuda() -> device=y_pred.device fix.

    The baseline form hard-crashes here with
    'RuntimeError: No CUDA GPUs are available' on a CPU-only machine.
    """
    pred, target = build_metrics_tensors(channels=channels)
    result = seg_metrics(pred, target, threshold=0.5)
    assert len(result) == 4
    assert all(np.isfinite(float(v)) for v in result)


@pytest.mark.gpu
@requires_cuda
def test_seg_metrics_on_cuda_matches_cpu():
    """Exercises seg_metrics on a non-default device -- the most direct check
    that the threshold tensors follow the input's device."""
    pred, target = build_metrics_tensors()
    cpu_result = [float(v) for v in seg_metrics(pred, target, threshold=0.5)]
    gpu_result = [float(v) for v in seg_metrics(pred.cuda(), target.cuda(), threshold=0.5)]

    for actual, expected in zip(gpu_result, cpu_result):
        assert actual == pytest.approx(expected, rel=1e-4)
