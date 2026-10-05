import torch
import sys
import os
from dataclasses import asdict

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ai.unet import UNet
from physics.config import OpticsConfig

CHECKPOINT_FORMAT = 1

def save_checkpoint(path: str, model: torch.nn.Module, cfg: OpticsConfig, model_kwargs: dict,
                    epoch: int, val_loss: float, **extra) -> None:
    """
    Saves weights together with everything needed to rebuild the model:
    {"model_state", "config", "model_kwargs", "epoch", "val_loss", "format"} plus any extras.
    """
    torch.save({
        "format": CHECKPOINT_FORMAT,
        "model_state": model.state_dict(),
        "config": asdict(cfg),
        "model_kwargs": dict(model_kwargs),
        "epoch": int(epoch),
        "val_loss": float(val_loss),
        **extra,
    }, path)

def load_model(path: str, device=torch.device('cpu')):
    """
    Rebuilds the model from a checkpoint's saved config and loads its weights strictly.
    Raises instead of ever falling back to random weights or default geometry.

    Returns:
        (model in eval mode, OpticsConfig, checkpoint dict)
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    ckpt = torch.load(path, map_location=device, weights_only=True)
    if not isinstance(ckpt, dict) or "model_state" not in ckpt or "config" not in ckpt:
        raise RuntimeError(
            f"{path} is a legacy bare state_dict without a saved config; it cannot be rebuilt "
            f"(pre-Phase-4 checkpoints also use the old geometry and single-plane input)."
        )
    cfg = OpticsConfig(**ckpt["config"])
    model = UNet(cfg, **ckpt["model_kwargs"]).to(device)
    try:
        model.load_state_dict(ckpt["model_state"], strict=True)
    except RuntimeError as e:
        raise RuntimeError(f"Checkpoint {path} does not match the rebuilt model:\n{e}") from e
    model.eval()
    return model, cfg, ckpt
