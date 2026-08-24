"""Version-independent fixture builder for the reversion test suite.

Deliberately depends only on numpy/pandas/mrcfile/argparse/pathlib (no
torch, no lightning) so this exact file can be copied verbatim into a
scratch build of the baseline worktree (commit 7851bd0) to (re)generate
golden snapshots, and also imported directly here for pytest.

See tests/README.md for the full regeneration procedure.
"""
from argparse import Namespace
from pathlib import Path

import mrcfile
import numpy as np
import pandas as pd

SEED = 20260818

VOLUME_SHAPE = (48, 48, 48)  # (z, y, x)
BLOCK_SIZE = 32
LABEL_DIAMETER = 3.0  # gaussian sigma for the synthetic label blob
MULTICLASS_RADIUS = 3  # sphere radius for the integer-class label volume

# The two fixture configs the suite runs everything over:
#   "core"       -- minimal: every optional architecture flag off. Pins the
#                   shared code path that all configs route through.
#   "production" -- mirrors train_AuNPs.py's real flags (image pyramid,
#                   coord conv, softmax, 3 classes, 4-level encoder), so the
#                   branches the actual AuNP runs take are covered too.
# Sizes (block_size, f_maps widths) stay small in both: they only cost CPU
# time and don't change which branches execute. f_maps DEPTH is matched to
# production's 4 levels, since that's what drives the pyramid/decoder path.
CONFIG_NAMES = ("core", "production")

DIR_NAME = "dir0"

# coords file rows: [class_flag, x, y, z]
#   class_flag == 15 triggers Dataset_ClsBased's 13x oversample-on-load branch
PRIMARY_COORD = (1, 24, 24, 24)
DUP_COORD = (15, 20, 20, 20)


def build_fixture_dataset(root):
    """(Re)create a tiny synthetic tomogram/label/coords tree under `root`.

    Layout:
        root/tomo/dir0.mrc      -- seeded gaussian-noise volume
        root/label/dir0.mrc     -- gaussian-blob label volume, same shape
        root/coords/dir0.coords -- tab-sep coords, no header
        root/coords/num_name.csv -- tab-sep [idx, dir_name], no header

    Idempotent: safe to call repeatedly, always overwrites.
    """
    root = Path(root)
    tomo_dir = root / "tomo"
    label_dir = root / "label"
    coord_dir = root / "coords"
    for d in (tomo_dir, label_dir, coord_dir):
        d.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(SEED)
    tomo = rng.normal(size=VOLUME_SHAPE).astype(np.float32)
    with mrcfile.new(str(tomo_dir / f"{DIR_NAME}.mrc"), overwrite=True) as m:
        m.set_data(tomo)

    zz, yy, xx = np.meshgrid(*[np.arange(s) for s in VOLUME_SHAPE], indexing="ij")
    label = np.zeros(VOLUME_SHAPE, dtype=np.float32)
    for _, x, y, z in (PRIMARY_COORD, DUP_COORD):
        d2 = (xx - x) ** 2 + (yy - y) ** 2 + (zz - z) ** 2
        label += np.exp(-d2 / (2 * LABEL_DIAMETER ** 2))
    label = np.clip(label, 0, 1).astype(np.float32)
    with mrcfile.new(str(label_dir / f"{DIR_NAME}.mrc"), overwrite=True) as m:
        m.set_data(label)

    # Multiclass label volume for the "production" config: integer class
    # indices (0 background, 1, 2) rather than a continuous blob, because
    # Dataset_ClsBased routes num_class>1 through multiclass_label(), which
    # does np.where(label == i, 1, 0) per class and needs discrete values.
    multicls = np.zeros(VOLUME_SHAPE, dtype=np.float32)
    for class_idx, (_, x, y, z) in enumerate((PRIMARY_COORD, DUP_COORD), start=1):
        d2 = (xx - x) ** 2 + (yy - y) ** 2 + (zz - z) ** 2
        multicls[d2 <= MULTICLASS_RADIUS ** 2] = class_idx
    multicls_dir = root / "label_multicls"
    multicls_dir.mkdir(parents=True, exist_ok=True)
    with mrcfile.new(str(multicls_dir / f"{DIR_NAME}.mrc"), overwrite=True) as m:
        m.set_data(multicls)

    pd.DataFrame([[0, DIR_NAME]]).to_csv(
        coord_dir / "num_name.csv", sep="\t", header=False, index=False)
    pd.DataFrame([list(PRIMARY_COORD), list(DUP_COORD)]).to_csv(
        coord_dir / f"{DIR_NAME}.coords", sep="\t", header=False, index=False)


def build_cfg(root, config="core"):
    """Return the `cfg` dict consumed by Dataset_ClsBased(cfg=...).

    The `production` config points at the integer-class label volume, since
    it runs num_classes=3 through the multiclass label path.
    """
    root = Path(root)
    label_dir = "label" if config == "core" else "label_multicls"
    return {
        "base_path": str(root),
        "label_name": label_dir,
        "coord_format": ".coords",
        "tomo_format": ".mrc",
        "norm_type": "standardization",
        "ocp_name": "ocp",
        "label_path": str(root / label_dir),
        "coord_path": str(root / "coords"),
        "tomo_path": str(root / "tomo"),
        "ocp_path": str(root / "ocp"),
    }


# Default args fields. Every one cross-checked against actual args.*
# accesses in Dataset_ClsBased.__init__/__getitem__, ResidualUNet3D.__init__,
# and DiceLoss.__init__ (see tests/README.md for details).
_DEFAULT_ARGS = {
    # baseline-only legacy flags -- required so capture against 7851bd0
    # doesn't raise AttributeError; harmless/ignored by current code
    "use_CL": False,
    "use_CL_DA": False,
    "use_bg_part": False,
    "use_ice_part": False,
    "Sel_Referance": False,

    # Dataset_ClsBased
    "input_cat": False,
    "input_cat_items": ["None"],
    "use_cluster": False,

    # get_model / ResidualUNet3D -- minimal core-path config
    "network": "ResUNet",
    "f_maps": [8, 16],
    "num_classes": 1,
    "in_channels": 1,
    "use_att": False,
    "use_paf": False,
    "use_uncert": False,
    "norm": "bn",
    "act": "relu",
    "use_lw": False,
    "lw_kernel": 3,
    "use_aspp": False,
    "pif_sigmoid": False,
    "paf_sigmoid": False,
    "use_tanh": False,
    "use_IP": False,
    "use_softmax": False,
    "use_sigmoid": True,
    "use_coord": False,
    "use_softpool": False,
    "use_se_loss": False,
    "final_double": False,

    # used by tests.trajectory / dataset construction, not read as a
    # library default by any of the modules under test
    "block_size": BLOCK_SIZE,
}


# Overrides that turn the core config into one mirroring train_AuNPs.py's
# real flags. Only the flags differ -- sizes stay small for CPU speed.
_PRODUCTION_OVERRIDES = {
    "f_maps": [8, 16, 24, 32],  # 4 levels, matching production's depth
    "num_classes": 3,
    "use_IP": True,
    "use_coord": True,
    "use_softmax": True,
    "use_sigmoid": False,
}

_CONFIG_OVERRIDES = {
    "core": {},
    "production": _PRODUCTION_OVERRIDES,
}


def build_args(config="core", **overrides):
    if config not in _CONFIG_OVERRIDES:
        raise ValueError(f"unknown config {config!r}; expected one of {CONFIG_NAMES}")
    fields = dict(_DEFAULT_ARGS)
    fields.update(_CONFIG_OVERRIDES[config])
    fields.update(overrides)
    return Namespace(**fields)
