import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from utilities.utils import *
from torch.utils.data import TensorDataset, DataLoader
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


def make_loss(name: str, w_throttle: float = 1.0, w_steer: float = 1.0):
    """
    Create a loss function.
    - "mse": standard mean squared error
    - "wmse": weighted MSE per action dimension
    """
    if name == "mse":
        mse = nn.MSELoss(reduction="mean")

        def loss_fn(y_pred, y_true):
            return mse(y_pred, y_true)

        return loss_fn

    elif name == "wmse":
        def loss_fn(y_pred, y_true):
            # weights: [steer, throttle]
            w = torch.tensor([w_steer, w_throttle],
                             dtype=y_true.dtype,
                             device=y_true.device)
            return ((y_pred - y_true) ** 2 * w).mean()

        return loss_fn
    else:
        raise ValueError(f"Unknown loss: {name}")


# ============================================================
#  Training Loop
# ============================================================
def run_training(data, args):
    """
    Train a feedforward NN for behavior cloning using
    input/output normalization. Saves model + scaling params.
    """
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    params = {"train_batch_size": 4096}
    in_size = data["x_train"].shape[-1]
    out_size = data["y_train"].shape[-1]  # (steering, throttle)

    # --------------------------------------------------------
    # 1. Compute normalization constants for X and Y
    # --------------------------------------------------------
    x_mean = np.mean(data["x_train"], axis=0)
    x_std = np.std(data["x_train"], axis=0)
    x_std[x_std < 1e-6] = 1e-6
    x_train_norm = (data["x_train"] - x_mean) / x_std

    y_mean = np.mean(data["y_train"], axis=0)
    y_std = np.std(data["y_train"], axis=0)
    y_std[y_std < 1e-6] = 1e-6
    y_train_norm = (data["y_train"] - y_mean) / y_std

    # --------------------------------------------------------
    # 2. Prepare model and optimizer
    # --------------------------------------------------------
    bc_model = BCModel(in_size, out_size).to(device)

    if args.restore:
        ckpt_path = f"./policies/{args.scenario.lower()}_{args.goal.lower()}_IL"
        state = torch.load(ckpt_path, map_location=device)
        bc_model.load_state_dict(state)
        print(f"Restored checkpoint from {ckpt_path}")

    optimizer = optim.Adam(bc_model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                           T_max=args.epochs)

    # --------------------------------------------------------
    # 3. DataLoader (normalized)
    # --------------------------------------------------------
    x_train = torch.tensor(x_train_norm, dtype=torch.float32)
    y_train = torch.tensor(y_train_norm, dtype=torch.float32)
    dataset = TensorDataset(x_train, y_train)
    dataloader = DataLoader(dataset,
                            batch_size=params["train_batch_size"],
                            shuffle=True)

    # --------------------------------------------------------
    # 4. Loss
    # --------------------------------------------------------
    loss_fn = make_loss(args.loss,
                        w_throttle=args.w_throttle,
                        w_steer=args.w_steer)

    # --------------------------------------------------------
    # 5. Training
    # --------------------------------------------------------
    bc_model.train()
    for epoch in range(args.epochs):
        epoch_loss = 0.0
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            y_est = bc_model(x)
            loss = loss_fn(y_est, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(bc_model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()

        scheduler.step()
        epoch_loss /= len(dataloader)
        print(f"Epoch {epoch + 1:3d}, Loss: {epoch_loss:.6f}, LR: {scheduler.get_last_lr()[0]:.2e}")

    # --------------------------------------------------------
    # 6. Save model + normalization constants
    # --------------------------------------------------------
    maybe_makedirs("./policies")
    ckpt_base = f"./policies/{args.scenario.lower()}_{args.goal.lower()}"
    torch.save(bc_model.state_dict(), ckpt_base + "_IL")
    np.savez(ckpt_base + "_scaling.npz",
             x_mean=x_mean, x_std=x_std,
             y_mean=y_mean, y_std=y_std)

    print(f"\n✅ Saved policy to: {ckpt_base}_IL")
    print(f"✅ Saved scaling params to: {ckpt_base}_scaling.npz")
    print("y_mean:", y_mean, "\ny_std:", y_std)


# ============================================================
#  Main Entry
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--scenario",
                        type=str,
                        default="intersection",
                        help="intersection, circularroad, lanechange")
    parser.add_argument("--goal",
                        type=str,
                        default="all",
                        help="left, straight, right, inner, outer, all")
    parser.add_argument("--epochs",
                        type=int,
                        default=400,
                        help="number of epochs for training")
    parser.add_argument("--lr",
                        type=float,
                        default=1e-3,
                        help="learning rate for Adam optimizer")
    parser.add_argument("--restore",
                        action="store_true",
                        default=False)

    # Loss and weights
    parser.add_argument("--loss",
                        type=str,
                        default="mse",
                        choices=["mse", "wmse"],
                        help="Loss type")
    parser.add_argument("--w_throttle",
                        type=float,
                        default=1.0,
                        help="Throttle weight (wmse only)")
    parser.add_argument("--w_steer",
                        type=float,
                        default=1.0,
                        help="Steering weight (wmse only)")

    args = parser.parse_args()

    maybe_makedirs("./policies")
    data = load_data(args)
    run_training(data, args)
