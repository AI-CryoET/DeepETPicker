"""Manual (no Lightning Trainer) training loop, used by both the golden
capture script and the regression tests.

Deliberately bypasses pytorch_lightning.Trainer so the suite isolates
model/loss/optimizer/data-loading math from Lightning-version-driven (2.5->2.6)
behavior differences, which are a separate, unscoped concern.

Depends on torch + the project modules, so (unlike tests/fixtures.py) this
is only ever run against a specific commit's source tree -- it gets copied
verbatim into the scratch baseline build for golden capture, and imported
normally here for the current-code regression tests.
"""
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from dataset.dataloader_DynamicLoad import Dataset_ClsBased
from model.model_loader import get_model
from utils.loss import DiceLoss


MODEL_INIT_SEED = 42
MODEL_INPUT_SEED = 1


def build_metrics_tensors(channels=1, seed=0):
    """Fixed synthetic (prediction, target) pair for the loss/metrics tests.

    Lives here rather than in fixtures.py because it needs torch, and
    fixtures.py is deliberately torch-free.
    """
    shape = (2, channels, 16, 16, 16)
    generator = torch.Generator().manual_seed(seed)
    pred = torch.rand(shape, generator=generator)
    target = (torch.rand(shape, generator=generator) > 0.7).float()
    return pred, target


def build_seeded_model(args):
    """Seeded model construction, shared by capture and tests so the two can
    never drift apart."""
    torch.manual_seed(MODEL_INIT_SEED)
    model = get_model(args)
    model.eval()
    return model


def build_model_input(args):
    block = args.block_size
    return torch.rand((1, args.in_channels, block, block, block),
                       generator=torch.Generator().manual_seed(MODEL_INPUT_SEED))


def build_train_dataset(cfg, args, mode="train"):
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


def run_training_trajectory(cfg, args, n_steps=10, batch_size=2, seed=20260818, device="cpu"):
    """Run `n_steps` manual optimizer steps over a couple of fixed batches
    drawn from the train-mode dataset fixture. Returns the list of per-step
    scalar loss values.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)

    ds = build_train_dataset(cfg, args)
    n_items = min(batch_size * 2, len(ds))
    subset = Subset(ds, list(range(n_items)))
    loader = DataLoader(subset, batch_size=batch_size, shuffle=False, num_workers=0)
    batches = list(loader)

    torch.manual_seed(seed)
    model = get_model(args).to(device)
    model.train()
    loss_fn = DiceLoss(args=args)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, betas=(0.9, 0.99), weight_decay=0.01)

    losses = []
    for step in range(n_steps):
        img, label, _ = batches[step % len(batches)]
        img = img.to(device).float()
        label = label.to(device).float()

        optimizer.zero_grad()
        out = model(img)
        loss = loss_fn(out, label)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    return losses
