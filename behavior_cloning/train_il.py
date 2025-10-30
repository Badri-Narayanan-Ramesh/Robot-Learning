'''
--loss mse
--loss wmse
--loss l1
--loss smoothl1
--loss huber --huber_delta 0.25
--loss mix --mix_alpha 0.7
'''

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from utilities.utils import *
from torch.utils.data import TensorDataset, DataLoader, Subset
import random


# ============================================================
#  Model Definition
# ============================================================
class BCModel(nn.Module):
    def __init__(self, in_size, out_size):
        super(BCModel, self).__init__()

        # Simple 2×128 MLP; final layer is linear (no tanh)
        self.net = nn.Sequential(
            nn.Linear(in_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, out_size),  # unbounded regression head
        )

        # Weight initialization
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=0.0, nonlinearity="relu")
                nn.init.zeros_(m.bias)

    def forward(self, x):
        # x: (B, |O|) → (B, 2)
        return self.net(x)


# ============================================================
#  Utilities
# ============================================================
def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_loss(name: str,
              w_throttle: float = 1.0,
              w_steer: float = 1.0,
              huber_delta: float = 1.0,
              mix_alpha: float = 0.5):
    """
    Returns a loss function that operates on *normalized* targets/preds.
    Supported:
      - "mse"        : Mean Squared Error
      - "wmse"       : Weighted MSE per dim [steer, throttle] weights
      - "l1"         : Mean Absolute Error
      - "smoothl1"   : PyTorch SmoothL1Loss (Huber) with delta=1.0
      - "huber"      : Explicit Huber with configurable delta
      - "mix"        : alpha * L1(steer) + (1-alpha) * MSE(throttle)
    """
    name = name.lower()

    if name == "mse":
        mse = nn.MSELoss(reduction="mean")
        return lambda y_pred, y_true: mse(y_pred, y_true)

    if name == "wmse":
        def loss_fn(y_pred, y_true):
            w = torch.tensor([w_steer, w_throttle],
                             dtype=y_true.dtype, device=y_true.device)
            return ((y_pred - y_true) ** 2 * w).mean()
        return loss_fn

    if name == "l1":
        mae = nn.L1Loss(reduction="mean")
        return lambda y_pred, y_true: mae(y_pred, y_true)

    if name == "smoothl1":
        sl1 = nn.SmoothL1Loss(reduction="mean", beta=1.0)
        return lambda y_pred, y_true: sl1(y_pred, y_true)

    if name == "huber":
        def huber(y_pred, y_true):
            diff = y_pred - y_true
            abs_diff = diff.abs()
            quadratic = torch.minimum(abs_diff,
                                      torch.tensor(huber_delta,
                                                   device=y_true.device,
                                                   dtype=y_true.dtype))
            # 0.5 * x^2 when |x| <= delta; otherwise delta*(|x| - 0.5*delta)
            linear = abs_diff - quadratic
            return (0.5 * quadratic ** 2 + huber_delta * linear).mean()
        return huber

    if name == "mix":
        def mix_loss(y_pred, y_true):
            steer_pred, throttle_pred = y_pred[:, 0], y_pred[:, 1]
            steer_true, throttle_true = y_true[:, 0], y_true[:, 1]
            l1_steer = (steer_pred - steer_true).abs().mean()
            mse_throttle = ((throttle_pred - throttle_true) ** 2).mean()
            return (w_steer * (mix_alpha * l1_steer) +
                    w_throttle * ((1.0 - mix_alpha) * mse_throttle))
        return mix_loss

    raise ValueError(f"Unknown loss: {name}")


def train_one_epoch(model, loader, optimizer, loss_fn, device, noise_std=0.0, clip_norm=1.0):
    model.train()
    running = 0.0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        if noise_std > 0:
            x = x + noise_std * torch.randn_like(x)

        optimizer.zero_grad()
        y_est = model(x)
        loss = loss_fn(y_est, y)
        loss.backward()
        if clip_norm is not None and clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
        optimizer.step()
        running += loss.item()
    return running / max(1, len(loader))


@torch.no_grad()
def evaluate(model, loader, loss_fn, device):
    model.eval()
    running = 0.0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        y_est = model(x)
        loss = loss_fn(y_est, y)
        running += loss.item()
    return running / max(1, len(loader))


# ============================================================
#  Training Loop
# ============================================================
def run_training(data, args):
    """
    Train a feedforward NN for behavior cloning using normalization,
    input noise, early stopping, and various loss options.
    """
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --------------------------------------------------------
    # Hyperparams
    # --------------------------------------------------------
    in_size = data["x_train"].shape[-1]
    out_size = data["y_train"].shape[-1]
    batch_size = args.batch_size
    weight_decay = args.weight_decay
    noise_std = args.input_noise_std
    val_split = args.val_split
    patience = args.patience
    min_delta = args.min_delta
    use_early_stop = args.early_stop
    save_best = args.save_best

    # --------------------------------------------------------
    # Train/Val Split
    # --------------------------------------------------------
    n = data["x_train"].shape[0]
    idx = np.random.permutation(n)
    n_val = int(val_split * n)
    val_idx = idx[:n_val]
    train_idx = idx[n_val:]

    x_full = data["x_train"]
    y_full = data["y_train"]

    # --------------------------------------------------------
    # Normalization (fit on train only)
    # --------------------------------------------------------
    x_mean = np.mean(x_full[train_idx], axis=0)
    x_std = np.std(x_full[train_idx], axis=0)
    x_std[x_std < 1e-6] = 1e-6

    y_mean = np.mean(y_full[train_idx], axis=0)
    y_std = np.std(y_full[train_idx], axis=0)
    y_std[y_std < 1e-6] = 1e-6

    x_norm = (x_full - x_mean) / x_std
    y_norm = (y_full - y_mean) / y_std

    # --------------------------------------------------------
    # Datasets & Loaders
    # --------------------------------------------------------
    x_t = torch.tensor(x_norm, dtype=torch.float32)
    y_t = torch.tensor(y_norm, dtype=torch.float32)
    full_ds = TensorDataset(x_t, y_t)

    train_ds = Subset(full_ds, train_idx.tolist())
    val_ds = Subset(full_ds, val_idx.tolist())

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    # --------------------------------------------------------
    # Model, Optimizer, Scheduler, Loss
    # --------------------------------------------------------
    model = BCModel(in_size, out_size).to(device)

    if args.restore:
        ckpt_path = f"./policies/{args.scenario.lower()}_{args.goal.lower()}_IL"
        state = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state)
        print(f"Restored checkpoint from {ckpt_path}")

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    loss_fn = make_loss(
        args.loss,
        w_throttle=args.w_throttle,
        w_steer=args.w_steer,
        huber_delta=args.huber_delta,
        mix_alpha=args.mix_alpha,
    )

    # --------------------------------------------------------
    # Training with Early Stopping
    # --------------------------------------------------------
    best_val = float("inf")
    best_epoch = -1
    ckpt_base = f"./policies/{args.scenario.lower()}_{args.goal.lower()}"
    maybe_makedirs("./policies")

    for epoch in range(args.epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device,
                                     noise_std=noise_std, clip_norm=args.clip_grad_norm)
        val_loss = evaluate(model, val_loader, loss_fn, device)
        scheduler.step()

        print(f"Epoch {epoch + 1:3d}, Train: {train_loss:.6f}, Val: {val_loss:.6f}, "
              f"LR: {scheduler.get_last_lr()[0]:.2e}")

        # Save best
        if val_loss + min_delta < best_val:
            best_val = val_loss
            best_epoch = epoch + 1
            if save_best:
                torch.save(model.state_dict(), ckpt_base + "_best_IL")
                np.savez(ckpt_base + "_best_scaling.npz",
                         x_mean=x_mean, x_std=x_std,
                         y_mean=y_mean, y_std=y_std)

        # Early stopping
        if use_early_stop and (epoch + 1) - best_epoch >= patience:
            print(f"\nEarly stopping at epoch {epoch + 1} "
                  f"(best val {best_val:.6f} at epoch {best_epoch})")
            break

    # --------------------------------------------------------
    # Save final model + scaling
    # --------------------------------------------------------
    torch.save(model.state_dict(), ckpt_base + "_IL")
    np.savez(ckpt_base + "_scaling.npz",
             x_mean=x_mean, x_std=x_std,
             y_mean=y_mean, y_std=y_std)

    print("\n==================== SAVE SUMMARY ====================")
    print(f"Final policy:         {ckpt_base}_IL")
    print(f"Final scaling params: {ckpt_base}_scaling.npz")
    if save_best and best_epoch > 0:
        print(f"Best policy:          {ckpt_base}_best_IL  (epoch {best_epoch})")
        print(f"Best scaling params:  {ckpt_base}_best_scaling.npz")
        print(f"Best Val Loss:        {best_val:.6f}")
    print("y_mean:", y_mean, "\ny_std:", y_std)
    print("=======================================================")


# ============================================================
#  Main Entry
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Scenario
    parser.add_argument("--scenario", type=str, default="intersection",
                        help="intersection, circularroad, lanechange")
    parser.add_argument("--goal", type=str, default="all",
                        help="left, straight, right, inner, outer, all")

    # Training
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--restore", action="store_true", default=False)

    # Loss and weights
    parser.add_argument("--loss", type=str, default="mse",
                        choices=["mse", "wmse", "l1", "smoothl1", "huber", "mix"],
                        help="Loss type")
    parser.add_argument("--w_throttle", type=float, default=1.0)
    parser.add_argument("--w_steer", type=float, default=1.0)
    parser.add_argument("--huber_delta", type=float, default=1.0)
    parser.add_argument("--mix_alpha", type=float, default=0.5)

    # Regularization / Robustness
    parser.add_argument("--input_noise_std", type=float, default=0.0)

    # Early stopping & validation split
    parser.add_argument("--early_stop", action="store_true", default=False)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--min_delta", type=float, default=1e-4)
    parser.add_argument("--val_split", type=float, default=0.1)

    # Save best checkpoint
    parser.add_argument("--save_best", action="store_true", default=True)

    # Gradient clipping
    parser.add_argument("--clip_grad_norm", type=float, default=1.0)

    args = parser.parse_args()

    maybe_makedirs("./policies")
    data = load_data(args)
    run_training(data, args)
