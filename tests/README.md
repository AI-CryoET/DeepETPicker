# Reversion test suite

Golden-master regression tests that pin down the *numeric behavior* of data
loading, loss, model forward pass, and training as it was at commit
**`7851bd0`** ("migrate to PL 2.x, working training") — the last known-good
state right after the `uv` migration and before the config-system rewrite,
CL-code removal, and device-handling refactors.

The point is not to test that the code is *correct*, but that it still does
the *same thing* it did when training was known to work.

## Running

```bash
uv sync --group test
uv run pytest tests/ -v              # everything (GPU tests auto-skip)
uv run pytest tests/ -m "not gpu"    # explicitly exclude GPU tests
uv run pytest tests/ -m gpu -v       # GPU tests only (needs a CUDA node)
```

Runs in ~7s on CPU. Run it before merging anything that touches
`dataset/`, `utils/loss.py`, `utils/metrics.py`, `model/`, `train.py`, or
`options/option.py`.

## Two fixture configs

Every test runs twice, once per config (see `fixtures.py::CONFIG_NAMES`):

| Config | Flags | Why |
|---|---|---|
| `core` | every optional architecture flag off, `num_classes=1`, 2-level `f_maps` | Pins the shared path *all* configs route through. Fast, and a failure is easy to localize |
| `production` | `use_IP=True`, `use_coord=True`, `use_softmax=True`, `num_classes=3`, 4-level `f_maps` | Mirrors `train_AuNPs.py`'s real flags, so the image-pyramid, coord-conv, softmax and multiclass-label branches the actual AuNP runs take are covered |

Only the *flags* are mirrored, not the sizes — `block_size` and `f_maps`
widths stay small because they cost CPU time without changing which branches
execute. `f_maps` **depth** is matched (4 levels), since that's what drives
the pyramid and decoder structure.

The `production` config reads its labels from `label_multicls/` (integer
class indices 0/1/2), because `num_classes>1` routes through
`multiclass_label()`, which needs discrete values.

## What's covered

| File | Covers |
|---|---|
| `test_dataset_regression.py` | `Dataset_ClsBased.__getitem__` for `mode='train'` (with augmentation) and `mode='val'` (without) — the path the mrcfile-handle fix and CL removal touched — plus the multiclass label expansion |
| `test_loss_metrics_regression.py` | `DiceLoss` vs. golden; `seg_metrics` vs. an independent NumPy oracle (1- and 3-channel) |
| `test_model_forward_regression.py` | `ResidualUNet3D` forward pass, output channel count, seeded weight init |
| `test_train_trajectory_regression.py` | 10-step manual training loop loss trajectory |

Supporting modules:
- `fixtures.py` — builds the synthetic tomogram/label/coords tree and the
  `cfg`/`args` fixtures. Deliberately **torch-free** (numpy/pandas/mrcfile
  only) so it can be copied verbatim into any commit's working tree.
- `trajectory.py` — the manual training loop, shared by capture and tests.
- `tools/capture_golden.py` — one-time golden capture script.
- `golden/` — the committed frozen reference. **Do not regenerate casually.**

## Two kinds of assertion, and why both exist

Most tests compare against golden snapshots. But
`test_volumes_are_plain_arrays_not_mrcfile_handles` is a **structural**
check, and it's the only thing that actually guards the mrcfile-handle fix:

The fix changed `self.origin = [mrcfile.open(...) for ...]` (live file
handles, indexed later via `.data`) to reading the array inside a `with`
block and storing plain ndarrays. **Both forms produce identical values** —
so the golden-value tests pass either way. This was verified empirically by
reverting the fix: only the structural test went red. If you're adding
coverage for a resource-handling or API-shape property, a value snapshot
will not catch it; assert on the shape directly.

## Why `seg_metrics` has no golden file

At the baseline commit, `seg_metrics` calls `torch.zeros(1).cuda()`
unconditionally, so on a CPU-only machine it raises
`RuntimeError: No CUDA GPUs are available` — there is **no CPU baseline
value to capture**. The current `device=y_pred.device` form is a strict fix
for that, not a behavior change worth freezing.

So `seg_metrics` is validated against an independent NumPy reference
implementation (`reference_seg_metrics`) instead. That's a stronger check
than a snapshot: it validates the metric *definitions*, so e.g. a swapped
`iou`/`f1` return order is caught. (Note `utils/metrics.py` returns
`(precision, recall, iou, f1)`, while `train.py` destructures it as
`precision, recall, f1_score, iou` — a latent mislabeling bug in `train.py`'s
logging that this suite deliberately does *not* reproduce.)

## Sensitivity: this suite has been proven to fail

A green suite means nothing unless it can go red. Each of these was
injected, confirmed to fail the expected tests, and reverted:

| Injected bug | Caught by |
|---|---|
| `point1[0] == 15` → `== 16` (oversample branch) | dataset tests |
| `1 - dice.mean()` → `dice.mean()` | loss + trajectory |
| `seg_metrics` iou/f1 return order swapped | NumPy oracle |
| `AvgPool3d` → `MaxPool3d` (encoder pooling) | model forward |
| mrcfile context-manager fix reverted | **structural test only** |
| pyramid pooling swapped (`use_IP`) | `[production]` only |
| `multiclass_label` `== i` → `>= i` | `[production]` only |
| `AddCoords` xx/yy channel order swapped | `[production]` only |

The last three fail for `[production]` and pass for `[core]` — that split is
the evidence the production config genuinely reaches those branches rather
than merely claiming to.

## Out of scope

- `cal_metrics_NMS_OneCls` / `cal_metrics_MultiCls` (`utils/misc.py`) —
  needs a full occupancy-map + gt-coords fixture. **Follow-up candidate.**
- Lightning `Trainer`-level behavior (`UNetExperiment`'s
  `training_step`/`validation_step` wiring, `persistent_workers`/`pin_memory`,
  `set_float32_matmul_precision`). The trajectory test deliberately bypasses
  `Trainer` so a Lightning 2.5→2.6 bump can't make this suite noisy. Tests
  must **never import `train.py`** — it calls
  `torch.set_float32_matmul_precision('high')` at import time, which the
  baseline doesn't, and would silently skew comparisons.
- The `mode in ('test','test_val','val_v1','test_only')` padded-tiling branch
  — unchanged between baseline and current. **Follow-up candidate.**

## Regenerating the golden files

Only do this for a **deliberate, reviewed** math change — never to make a
failing test go green. If a test fails, the default assumption is that the
code regressed, not that the reference is stale.

The capture must run against the baseline commit, which needs its own venv
(its pins differ: `numpy==1.24.0`, `scipy==1.15.3`, PyQt5 as a plain
top-level dep since the `gui` dependency-group split landed later). Work on
a disposable copy — never modify the `DeepETPicker-baseline` worktree:

```bash
SCRATCH=/tmp/baseline-venvbuild
rm -rf "$SCRATCH"
cp -r ../DeepETPicker-baseline "$SCRATCH"
rm -f "$SCRATCH"/.git                      # detach from the worktree
echo "3.11" > "$SCRATCH"/.python-version   # numpy 1.24 can't build on 3.12+
# strip "pyqtgraph==0.12.1" and "PyQt5==5.15.4" from $SCRATCH/pyproject.toml
cd "$SCRATCH" && uv sync

cp <repo>/tests/{fixtures.py,trajectory.py} "$SCRATCH"/
cp <repo>/tests/tools/capture_golden.py "$SCRATCH"/
uv run python capture_golden.py --out-dir <repo>/tests/golden

cd <repo> && rm -rf "$SCRATCH"
git status tests/golden/   # review the diff before committing
```

`capture_golden.py` uses flat imports (`import fixtures`) precisely because
the baseline commit has no `tests/` package for them to live under.
