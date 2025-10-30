import numpy as np
from behavior_cloning.train_il import BCModel
import gym_carlo
import gym
import time
import argparse
import torch
from utilities.utils import *

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
    parser.add_argument("--visualize", action="store_true", default=False)
    args = parser.parse_args()

    # --------------------------------------------------------
    # 1. Prepare environment
    # --------------------------------------------------------
    scenario_name = args.scenario.lower()
    assert scenario_name in scenario_names, "--scenario argument is invalid!"

    if args.goal.lower() == "all":
        env = gym.make(scenario_name + "Scenario-v0", goal=len(goals[scenario_name]))
    else:
        env = gym.make(
            scenario_name + "Scenario-v0",
            goal=np.argwhere(np.array(goals[scenario_name]) == args.goal.lower())[0, 0],
        )

    # --------------------------------------------------------
    # 2. Load model and scaling parameters
    # --------------------------------------------------------
    ckpt_base = f"./policies/{scenario_name}_{args.goal.lower()}"
    model_path = ckpt_base + "_IL"
    scale_path = ckpt_base + "_scaling.npz"

    bc_model = BCModel(obs_sizes[scenario_name], 2)
    bc_model.load_state_dict(torch.load(model_path, map_location="cpu"))
    bc_model.eval()

    scale = np.load(scale_path)
    x_mean, x_std = scale["x_mean"], scale["x_std"]
    y_mean, y_std = scale["y_mean"], scale["y_std"]

    print(f"Loaded model: {model_path}")
    print(f"Loaded scaling: {scale_path}")
    print("y_mean:", y_mean, "y_std:", y_std)

    # --------------------------------------------------------
    # 3. Run evaluation
    # --------------------------------------------------------
    episode_number = 10 if args.visualize else 100
    success_counter = 0
    env.T = 200 * env.dt - env.dt / 2.0  # Max 20 seconds per episode

    for ep in range(episode_number):
        env.seed(int(np.random.rand() * 1e6))
        obs, done = env.reset(), False
        if args.visualize:
            env.render()

        step_counter = 0
        while not done:
            t = time.time()

            # ------------------------------------------------
            #  a) Normalize observation using saved stats
            # ------------------------------------------------
            obs = np.array(obs, dtype=np.float32).reshape(1, -1)
            obs_norm = (obs - x_mean) / (x_std + 1e-8)

            # ------------------------------------------------
            #  b) Predict normalized actions
            # ------------------------------------------------
            with torch.no_grad():
                a_norm = bc_model(torch.from_numpy(obs_norm)).cpu().numpy().reshape(-1)

            # ------------------------------------------------
            #  c) Un-normalize to physical (steering, throttle)
            # ------------------------------------------------
            action = a_norm * y_std + y_mean

            # ------------------------------------------------
            #  d) Clamp to valid ranges
            # ------------------------------------------------
            lo, hi = steering_lims[scenario_name]
            action[0] = float(np.clip(action[0], lo, hi))      # steering
            action[1] = float(np.clip(action[1], 0.0, 1.0))    # throttle

            # Debug (first few steps of first episode)
            if ep == 0 and step_counter < 5:
                print(f"Step {step_counter:02d}: norm_act={a_norm}, phys_act={action}")

            # ------------------------------------------------
            #  e) Step the environment
            # ------------------------------------------------
            obs, _, done, _ = env.step(action.astype(np.float32))
            step_counter += 1

            if args.visualize:
                env.render()
                while time.time() - t < env.dt / 2:
                    pass  # run ~2× realtime

            if done:
                env.close()
                if args.visualize:
                    time.sleep(1)
                if hasattr(env, "target_reached") and env.target_reached:
                    success_counter += 1

    # --------------------------------------------------------
    # 4. Print summary
    # --------------------------------------------------------
    if not args.visualize:
        print("\n--------------------------------")
        print(f"Success Rate = {float(success_counter) / episode_number:.2f}")
        print("--------------------------------")
