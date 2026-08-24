"""Reversion tests for Dataset_ClsBased's train/val loading path.

This is the code path the mrcfile-handle fix and the CL-code removal
touched, so it gets both a golden-value check and a structural check.

Every test runs once per fixture config (see tests/conftest.py::config), so
the multiclass label path the real AuNP runs take is covered alongside the
minimal single-class path.
"""
import mrcfile
import numpy as np
import pytest
import torch

from dataset.dataloader_DynamicLoad import Dataset_ClsBased
from tests.conftest import load_golden_npz
from tests.fixtures import SEED


def build_dataset(mode, cfg, args):
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    return Dataset_ClsBased(
        mode=mode,
        block_size=args.block_size,
        num_class=args.num_classes,
        random_num=0,
        use_bg=True,
        data_split=[0, 1, 0, 1, 0, 1],
        test_use_pad=False,
        pad_size=18,
        use_paf=False,
        cfg=cfg,
        args=args,
    )


def collect_samples(ds):
    imgs, labels, positions = [], [], []
    for i in range(len(ds)):
        img, label, position = ds[i]
        imgs.append(img.numpy())
        labels.append(label.numpy())
        positions.append(np.array(position))
    return np.stack(imgs), np.stack(labels), np.stack(positions)


@pytest.mark.parametrize("mode", ["train", "val"])
def test_dataset_matches_golden(mode, config, cfg, args):
    """Golden-value check: crops, labels and positions must be bit-identical
    to what the known-good baseline (7851bd0) produced."""
    ds = build_dataset(mode, cfg, args)
    golden = load_golden_npz(f"dataset_{config}_{mode}.npz")

    imgs, labels, positions = collect_samples(ds)

    np.testing.assert_array_equal(imgs, golden["imgs"])
    np.testing.assert_array_equal(labels, golden["labels"])
    np.testing.assert_array_equal(positions, golden["positions"])


@pytest.mark.parametrize("mode", ["train", "val"])
def test_volumes_are_plain_arrays_not_mrcfile_handles(mode, cfg, args):
    """Structural check specific to the mrcfile-handle fix.

    The pre-fix code stored live mrcfile.MrcFile objects in self.origin /
    self.label and indexed them via `.data`; the fix reads the array inside a
    context manager and stores plain ndarrays, so no file handles are left
    open. Golden-value equality alone does NOT catch a regression back to the
    handle-holding pattern -- verified empirically: reverting the fix leaves
    every golden test green and trips only this one.
    """
    ds = build_dataset(mode, cfg, args)

    assert isinstance(ds.origin[0], np.ndarray)
    assert isinstance(ds.label[0], np.ndarray)
    assert not isinstance(ds.origin[0], mrcfile.mrcfile.MrcFile)
    assert not isinstance(ds.label[0], mrcfile.mrcfile.MrcFile)


def test_class_flag_15_oversampling_branch(cfg, args):
    """The fixture's coords file has one ordinary row and one class-flag-15
    row; the latter is duplicated 13x at construction. Pins that branch."""
    ds = build_dataset("train", cfg, args)
    assert len(ds) == 1 + 13


def test_label_channels_match_num_classes(config, cfg, args):
    """num_classes>1 routes through multiclass_label(), which turns a 3D
    class-index volume into one binary channel per class. Pins that the
    multiclass expansion actually happens for the production config."""
    ds = build_dataset("val", cfg, args)
    _, label, _ = ds[0]

    assert label.shape[0] == args.num_classes
    if config == "production":
        assert args.num_classes == 3
        # multiclass_label emits strictly binary channels
        assert set(np.unique(label.numpy())).issubset({0.0, 1.0})
