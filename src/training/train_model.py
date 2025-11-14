import pickle
from pathlib import Path
import pytorch_lightning as pl
import yaml


from datasources.build_loaders import build_loaders
from training.build_trainer import build_trainer

from learning_models.BaseForecastModel import BaseForecastModel
from learning_models.GPT2Backbone import GPT2Backbone
from learning_models.SelectiveSSMBackbone import SelectiveSSMBackbone


def train_model(cfg, model_name):

    train_ds, val_ds, train_loader, val_loader, p, m = build_loaders(cfg)
    print(f"[{model_name}] p={p}, m={m}, H={cfg['H']}, L={cfg['L']}, "
          f"train_batches={len(train_loader)}, val_batches={len(val_loader) if val_loader else 0}")

    if model_name == "gpt2":
        backbone = GPT2Backbone(
            p=p, m=m, H=cfg["H"],
            d_model=cfg["d_model"], n_layer=cfg["n_layer"], n_head=cfg["n_head"],
            dropout=cfg["dropout"], max_len=cfg["max_len"],
        )
        arch_type = "gpt2"
        arch_cfg = {
            "d_model": cfg["d_model"],
            "n_layer": cfg["n_layer"],
            "n_head":  cfg["n_head"],
            "dropout": cfg["dropout"],
            "max_len": cfg["max_len"],
        }
    elif model_name == "ssm":
        backbone = SelectiveSSMBackbone(
            p=p, m=m, H=cfg["H"],
            d_model=cfg["d_model"], n_x=cfg["n_x"],
            s_A=cfg["s_A"], use_delta=cfg["use_delta"], fix_sA=cfg["fix_sA"],
            dropout=cfg["dropout"],
        )
        arch_type = "ssm"
        arch_cfg = {
            "d_model":  cfg["d_model"],
            "n_x":      cfg["n_x"],
            "s_A":      cfg["s_A"],
            "use_delta":cfg["use_delta"],
            "fix_sA":   cfg["fix_sA"],
            "dropout":  cfg["dropout"],
        }
    
    lm = BaseForecastModel(
        backbone=backbone, H=cfg["H"],
        lr=cfg["lr"], weight_decay=cfg["weight_decay"],
        loss_reduction=cfg["loss_reduction"],
        norm_stats=train_ds.norm_stats
    )

    u_tag = "input" if train_ds.m > 0 else "no_input"
    L_tag = f"L{cfg['L']}" if cfg["L"] is not None else "Lfull"
    lr_tag = f"lr{cfg['lr']}"
    epochs_tag = f"ep{cfg['epochs']}"
    run_dir = Path(cfg["outputs_dir"]) / train_ds.dataset_type / f"{model_name}-{u_tag}-H{cfg['H']}-{L_tag}-{lr_tag}-{epochs_tag}"

    cfg["model_name"] = model_name
    trainer, ckpt = build_trainer(run_dir, cfg, val_loader=val_loader)

    pl.seed_everything(42, workers=True)
    trainer.fit(lm, train_loader, val_loader)
    print(f"[{model_name}] Best checkpoint: {ckpt.best_model_path}")

    stats = train_ds.norm_stats
    if stats is not None:
        run_dir.mkdir(parents=True, exist_ok=True)
        with open(run_dir / "norm_stats.pkl", "wb") as f:
            pickle.dump(stats, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"[{model_name}] Saved norm stats → {run_dir / 'norm_stats.pkl'}")
    meta = {
        "type": arch_type,                 # "gpt2" or "ssm"
        "p": int(p),
        "m": int(m),
        "H": int(cfg["H"]),
        "arch": arch_cfg,                  # hyperparameters needed to rebuild the backbone
        "train": {                        
            "lr": float(cfg["lr"]),
            "weight_decay": float(cfg["weight_decay"]),
            "loss_reduction": str(cfg["loss_reduction"]),
            "epochs": int(cfg["epochs"]),
            "batch_size": int(cfg["batch_size"]),
            "L": (None if cfg["L"] is None else int(cfg["L"])),
            "seed": int(cfg.get("seed", 42)),
        },
        "data": {
            "dataset_type": train_ds.dataset_type  # "lti" or "drone"
        }
    }

    meta_path = run_dir / "model_meta.yaml"
    with open(meta_path, "w") as f:
        yaml.safe_dump(meta, f, sort_keys=False)
    print(f"[{model_name}] Saved model meta to {meta_path}")