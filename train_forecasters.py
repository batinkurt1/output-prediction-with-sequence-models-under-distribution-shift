import os
import json
import time
import hashlib
import argparse
import logging
from pathlib import Path

import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

from datasources.make_train_val_datasets import make_train_val_datasets
from learning_models.BaseForecastModel import BaseForecastModel
from learning_models.GPT2Backbone import GPT2Backbone
from learning_models.MambaBackbone import MambaBackbone


# ---------------- run-dir & logging (original-style) ----------------
def _hash_args(args: argparse.Namespace) -> str:
    payload = json.dumps(vars(args), sort_keys=True, default=str).encode("utf-8")
    return hashlib.md5(payload).hexdigest()[:6]

def make_output_dir(outputs_root: Path, model_name: str, args: argparse.Namespace) -> Path:
    ts = time.strftime("%y%m%d_%H%M%S")
    run_id = f"{ts}.{_hash_args(args)}"
    out_dir = outputs_root / model_name / run_id
    (out_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    return out_dir

def snapshot_run(out_dir: Path, args: argparse.Namespace):
    # Save CLI args
    with open(out_dir / "args.json", "w") as f:
        json.dump(vars(args), f, indent=2, default=str)
    # Save simple environment info
    env = [
        f"torch={torch.__version__}",
        f"pytorch_lightning={pl.__version__}",
        f"cuda_available={torch.cuda.is_available()}",
        f"cuda_device_count={torch.cuda.device_count()}",
        f"device_name={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}",
    ]
    (out_dir / "env.txt").write_text("\n".join(env) + "\n")

def setup_file_logger(out_dir: Path):
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    # keep stdout if present; add file handler for messages.log
    fh = logging.FileHandler(out_dir / "messages.log")
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s", "%Y-%m-%d %H:%M:%S")
    fh.setFormatter(fmt)
    root.addHandler(fh)


# ---------------- data ----------------
def build_loaders(pkl_path: str, H: int, L: int | None,
                  batch_size: int, num_workers: int,
                  train_ratio: float = 0.9, seed: int = 0):
    train_ds, val_ds = make_train_val_datasets(
        pkl_path=pkl_path, H=H, L=L, train_ratio=train_ratio, seed=seed
    )
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=False,
        persistent_workers=(num_workers > 0),
    )
    val_loader = None
    if val_ds is not None and len(val_ds) > 0:
        val_loader = DataLoader(
            val_ds, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True, drop_last=False,
            persistent_workers=(num_workers > 0),
        )

    try:
        p = int(train_ds.p)
        m = 0
        if getattr(train_ds, "u", None) is not None:
            m = int(train_ds.u.shape[-1])
    except Exception:
        sample = train_ds[0]
        p = sample["y"].shape[-1]
        m = sample.get("u", torch.empty(0)).shape[-1] if "u" in sample else 0

    return train_loader, val_loader, p, m


def build_callbacks(out_dir: Path, has_val: bool, default_monitor="val_loss_mse"):
    monitor = default_monitor if has_val else "train_loss_mse"

    # Let Lightning format {epoch:02d} and {<monitor>:.4f} at save-time.
    # NOTE: no f-string here! We inject the monitor name with % formatting.
    filename = "best-epoch={epoch:02d}-%s={%s:.4f}" % (monitor, monitor)

    ckpt_best = ModelCheckpoint(
        dirpath=str(out_dir / "checkpoints"),
        filename=filename,
        monitor=monitor,
        mode="min",
        save_top_k=1,
        save_last=True,
    )

    lr_mon = LearningRateMonitor(logging_interval="step")
    callbacks = [ckpt_best, lr_mon]
    if has_val:
        callbacks.append(EarlyStopping(monitor=monitor, mode="min", patience=10))
    return callbacks



# ---------------- training ----------------
def train_one_model(model_name: str,
                    backbone: torch.nn.Module,
                    H: int,
                    train_loader, val_loader,
                    max_epochs: int,
                    lr: float, weight_decay: float,
                    grad_clip: float | None,
                    outputs_root: Path,
                    accelerator: str,
                    devices: int,
                    args: argparse.Namespace):
    pl.seed_everything(42, workers=True)

    # Output dir, snapshots, logging
    out_dir = make_output_dir(outputs_root, model_name, args)
    snapshot_run(out_dir, args)
    setup_file_logger(out_dir)
    logging.getLogger(__name__).info(f"Output dir: {out_dir}")

    # Model
    lm = BaseForecastModel(
        backbone=backbone,
        H=H,
        lr=lr,
        weight_decay=weight_decay,
        loss_reduction="mean",
    )

    # Callbacks & TensorBoard logger (original-style)
    callbacks = build_callbacks(out_dir, has_val=(val_loader is not None))
    # This writes to: out_dir / lightning_logs / version_0 / events...
    tb_logger = TensorBoardLogger(save_dir=str(out_dir))

    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator=accelerator,
        devices=devices,
        gradient_clip_val=grad_clip if grad_clip else 0.0,
        gradient_clip_algorithm="norm",
        log_every_n_steps=10,
        callbacks=callbacks,
        deterministic=False,
        logger=tb_logger,
    )

    trainer.fit(lm, train_loader, val_loader)
    best_path = callbacks[0].best_model_path
    logging.getLogger(__name__).info(f"[{model_name}] Best checkpoint: {best_path}")
    print(f"[{model_name}] Best checkpoint: {best_path}")


def main():
    ap = argparse.ArgumentParser(description="Train GPT-2 and Mamba SSM forecasters on LTI/Drone datasets.")
    ap.add_argument("--data", type=str, default="../data/lti_dataset.pkl",
                    help="Path to dataset pickle (*.pkl or *.pkl.gz).")
    ap.add_argument("--system", type=str, choices=["lti", "drone"], default="lti",
                    help="Only used for printing/context; data path is what matters.")
    ap.add_argument("--H", type=int, default=1, help="Forecast horizon.")
    ap.add_argument("--L", type=int, default=None, help="History window length (set None for full-context).")
    ap.add_argument("--train-ratio", type=float, default=0.9, help="Train/val split by trajectory.")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=1e-2, help="Weight decay.")
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--models", type=str, choices=["gpt2", "mamba", "both"], default="both")
    ap.add_argument("--outputs-dir", type=str, default="../outputs",
                    help="Root folder for outputs/logs/checkpoints.")
    ap.add_argument("--grad-clip", type=float, default=1.0)
    args = ap.parse_args()

    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    devices = 1
    L = None if str(args.L).lower() == "none" else int(args.L)

    # Data
    train_loader, val_loader, p, m = build_loaders(
        pkl_path=args.data, H=args.H, L=L,
        batch_size=args.batch_size, num_workers=args.num_workers,
        train_ratio=args.train_ratio, seed=0,
    )
    print(f"[data] p={p}, m={m}, H={args.H}, L={L}, "
          f"train_batches={len(train_loader)}, val_batches={len(val_loader) if val_loader else 0}")

    outputs_root = Path(args.outputs_dir)

    # GPT-2
    if args.models in ("gpt2", "both"):
        gpt2_backbone = GPT2Backbone(
            p=p, m=m, H=args.H,
            d_model=256, n_layer=6, n_head=8,
            dropout=0.1, max_len=2048
        )
        train_one_model(
            model_name="gpt2",
            backbone=gpt2_backbone,
            H=args.H,
            train_loader=train_loader,
            val_loader=val_loader,
            max_epochs=args.epochs,
            lr=args.lr,
            weight_decay=args.wd,
            grad_clip=args.grad_clip,
            outputs_root=outputs_root,
            accelerator=accelerator,
            devices=devices,
            args=args,
        )

    # Mamba
    if args.models in ("mamba", "both"):
        mamba_backbone = MambaBackbone(
            p=p, m=m, H=args.H,
            d_model=256, n_layer=6, d_state=16, d_conv=4, expand=2, dropout=0.0
        )
        train_one_model(
            model_name="mamba",
            backbone=mamba_backbone,
            H=args.H,
            train_loader=train_loader,
            val_loader=val_loader,
            max_epochs=args.epochs,
            lr=args.lr,
            weight_decay=args.wd,
            grad_clip=args.grad_clip,
            outputs_root=outputs_root,
            accelerator=accelerator,
            devices=devices,
            args=args,
        )

if __name__ == "__main__":
    main()
    # examples:
    #   python train_forecasters.py --data ../data/lti_dataset.pkl --models both --H 5 --L 64
    #   python train_forecasters.py --data ../data/drone_dataset.pkl --models gpt2 --H 3 --L 128