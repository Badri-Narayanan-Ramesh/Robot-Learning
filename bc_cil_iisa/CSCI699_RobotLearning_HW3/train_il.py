import numpy as np
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from utils import *
# import torch.nn.functional as F  # (unused)
from torch.utils.data import TensorDataset, DataLoader
import random

class BCModel(nn.Module):
    def __init__(self, in_size, out_size):
        super(BCModel, self).__init__()

        # 2×128 MLP; tanh keeps outputs in [-1, 1]
        self.net = nn.Sequential(
            nn.Linear(in_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, out_size),
            nn.Tanh(),
        )

        # Kaiming init for ReLU layers, small init for final head
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=0.0, nonlinearity="relu")
                nn.init.zeros_(m.bias)
        final = self.net[-2]  # last Linear before Tanh
        if isinstance(final, nn.Linear):
            nn.init.xavier_uniform_(final.weight, gain=0.01)
            nn.init.zeros_(final.bias)

    def forward(self, x):
        # x: (B, |O|) -> (B, 2)
        return self.net(x)


def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_training(data, args):
    """
    Trains a feedforward NN for behavior cloning.
    """
    set_seed(42)

    params = {
        "train_batch_size": 4096,
    }

    in_size = data["x_train"].shape[-1]
    out_size = data["y_train"].shape[-1]  # should be 2

    bc_model = BCModel(in_size, out_size)

    if args.restore:
        ckpt_path = (
            "./policies/" + args.scenario.lower() + "_" + args.goal.lower() + "_IL"
        )
        bc_model.load_state_dict(torch.load(ckpt_path))

    optimizer = optim.Adam(bc_model.parameters(), lr=args.lr)
    # optional: small weight decay can help, but keeping exactly your args
    # optimizer = optim.Adam(bc_model.parameters(), lr=args.lr, weight_decay=1e-5)

    # Prepare dataloader
    dataset = TensorDataset(
        torch.tensor(data["x_train"], dtype=torch.float32),
        torch.tensor(data["y_train"], dtype=torch.float32),
    )
    dataloader = DataLoader(dataset, batch_size=params["train_batch_size"], shuffle=True)

    # One-batch train step (has access to bc_model/optimizer via closure)
    def train_step(x, y):
        """
        We want to compute the loss between y_est and y where:
        - y_est is the output of the network for a batch of observations,
        - y is the actions the expert took for the corresponding observations.

        NOTE on action order:
        utils.load_data() comment says: y = [throttle, steering].
        If you later discover it's actually [steering, throttle], either swap here
        or in test script. Start with unweighted MSE.
        """
        y_est = bc_model(x)  # (B, 2) in [-1,1] due to tanh

        # Optional unequal penalty (start with 1.0, 1.0)
        w_throttle = 1.0
        w_steer = 1.0
        w = torch.tensor([w_throttle, w_steer], dtype=y.dtype, device=y.device)  # (2,)

        mse = (y_est - y) ** 2  # (B, 2)
        loss = (mse * w).mean()  # scalar
        return loss

    # Train
    bc_model.train()
    for epoch in range(args.epochs):
        epoch_loss = 0.0
        for x, y in dataloader:
            optimizer.zero_grad()
            loss = train_step(x, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(bc_model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()
        epoch_loss /= len(dataloader)
        print(f"Epoch {epoch + 1}, Loss: {epoch_loss:.6f}")

    # Save checkpoint
    ckpt_path = "./policies/" + args.scenario.lower() + "_" + args.goal.lower() + "_IL"
    torch.save(bc_model.state_dict(), ckpt_path)
    print(f"Saved policy to: {ckpt_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--scenario",
        type=str,
        help="intersection, circularroad, lanechange",
        default="intersection",
    )
    parser.add_argument(
        "--goal",
        type=str,
        help="left, straight, right, inner, outer, all",
        default="all",
    )
    parser.add_argument(
        "--epochs", type=int, help="number of epochs for training", default=1000
    )
    parser.add_argument(
        "--lr", type=float, help="learning rate for Adam optimizer", default=5e-3
    )
    parser.add_argument("--restore", action="store_true", default=False)
    args = parser.parse_args()

    maybe_makedirs("./policies")
    data = load_data(args)
    run_training(data, args)
