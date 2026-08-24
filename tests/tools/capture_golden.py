"""One-time golden-snapshot capture script.

NOT a pytest file. Run this exactly once from inside the KNOWN-GOOD baseline
worktree (commit 7851bd0) to (re)generate tests/golden/*. Never run it
against current code to "fix" a failing regression test -- that defeats the
purpose. See tests/README.md for the full regeneration procedure and when
it's legitimate to refresh the golden files (a deliberate, reviewed math
change -- not a silent regression).

This script uses FLAT imports (`import fixtures`, `import trajectory`)
because it is meant to be copied, along with fixtures.py and trajectory.py,
flat into a scratch copy of the baseline worktree and run from there -- the
baseline commit has no `tests/` package for these to live under.

Everything is captured once per fixture config ("core" and "production" --
see tests/fixtures.py), so the branches the real AuNP runs take (image
pyramid, coord conv, softmax, multiclass labels) are covered too.

Usage:
    uv run python capture_golden.py --out-dir /path/to/tests/golden
"""
import argparse
import json
import tempfile
from pathlib import Path

import numpy as np
import torch

import fixtures
import trajectory

from model.model_loader import get_model
from utils.loss import DiceLoss


def capture_dataset(cfg, args, mode, out_path):
    from dataset.dataloader_DynamicLoad import Dataset_ClsBased

    np.random.seed(fixtures.SEED)
    torch.manual_seed(fixtures.SEED)
    ds = Dataset_ClsBased(
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

    imgs, labels, positions = [], [], []
    for i in range(len(ds)):
        img, label, position = ds[i]
        imgs.append(img.numpy())
        labels.append(label.numpy())
        positions.append(np.array(position))

    np.savez(
        out_path,
        imgs=np.stack(imgs),
        labels=np.stack(labels),
        positions=np.stack(positions),
    )
    print(f"  wrote {out_path.name} ({len(ds)} samples, label shape {labels[0].shape})")


def capture_loss(args, out_path):
    """Capture DiceLoss only.

    seg_metrics is deliberately NOT captured here: at the baseline commit it
    calls torch.zeros(1).cuda() unconditionally, so it cannot run on CPU at
    all (RuntimeError: No CUDA GPUs are available). The current code's
    device=y_pred.device change is a strict fix for that, not a behavior
    change worth freezing -- so seg_metrics is instead validated in the test
    suite against an independent NumPy reference oracle, which is a stronger
    check than a snapshot anyway. See tests/README.md.
    """
    pred, target = trajectory.build_metrics_tensors(channels=args.num_classes)

    loss_fn = DiceLoss(args=args)
    golden = {"dice_loss": loss_fn(pred, target).item()}

    with open(out_path, "w") as f:
        json.dump(golden, f, indent=2)
    print(f"  wrote {out_path.name}: {golden}")


def capture_model_forward(args, out_npy_path, out_state_path):
    model = trajectory.build_seeded_model(args)
    x = trajectory.build_model_input(args)
    with torch.no_grad():
        out = model(x)

    np.save(out_npy_path, out.numpy())
    torch.save(model.state_dict(), out_state_path)
    print(f"  wrote {out_npy_path.name} (shape {tuple(out.shape)}) and {out_state_path.name}")


def capture_train_trajectory(cfg, args, out_path):
    losses = trajectory.run_training_trajectory(cfg, args, n_steps=10, batch_size=2,
                                                  seed=fixtures.SEED, device="cpu")
    with open(out_path, "w") as f:
        json.dump({"losses": losses}, f, indent=2)
    print(f"  wrote {out_path.name}: {[round(v, 5) for v in losses]}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", required=True)
    cli_args = parser.parse_args()

    out_dir = Path(cli_args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    scratch_root = Path(tempfile.mkdtemp(prefix="regfixture_"))
    fixtures.build_fixture_dataset(scratch_root)

    for config in fixtures.CONFIG_NAMES:
        print(f"\n=== config: {config} ===")
        cfg = fixtures.build_cfg(scratch_root, config)
        args = fixtures.build_args(config)

        for mode in ("train", "val"):
            capture_dataset(cfg, args, mode, out_dir / f"dataset_{config}_{mode}.npz")
        capture_loss(args, out_dir / f"metrics_{config}.json")
        capture_model_forward(args,
                              out_dir / f"model_forward_{config}.npy",
                              out_dir / f"model_init_state_{config}.pt")
        capture_train_trajectory(cfg, args, out_dir / f"train_trajectory_{config}.json")

    print(f"\nGolden snapshots written to {out_dir}")


if __name__ == "__main__":
    main()
