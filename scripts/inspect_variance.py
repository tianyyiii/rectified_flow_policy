import os
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["OMP_NUM_THREADS"] = "1"

import sys
from pathlib import Path
import argparse
import pickle
import csv

import numpy as np
import jax
import jax.numpy as jnp
from jax import random
from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt

from relax.env import create_env
from relax.utils.persistence import PersistFunction

def evaluate(key, env, policy_fn, policy_params, num_episodes):
    ep_len_list = []
    ep_ret_list = []
    action_std_list = []
    for _ in range(num_episodes):
        obs, _ = env.reset()
        ep_len = 0
        ep_ret = 0.0
        while True:
            if ep_len % 10 == 0:
                acts = []
                for _ in range(100):
                    key, std_key = jax.random.split(key)
                    acts.append(policy_fn_rnd(std_key, policy_params, obs))
                acts = np.stack(acts, axis=0)
                mean_action = np.mean(acts, axis=0)   
                diffs = acts - mean_action  
                sq_norms = np.sum(diffs**2, axis=1)   
                std = np.sqrt(np.mean(sq_norms))
                action_std_list.append(std)
            act = policy_fn(policy_params, obs)
            obs, reward, terminated, truncated, _ = env.step(act)
            ep_len += 1
            ep_ret += reward
            if terminated or truncated:
                break
        ep_len_list.append(ep_len)
        ep_ret_list.append(ep_ret)
    return ep_len_list, ep_ret_list, action_std_list, key


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy_root", type=str)
    parser.add_argument("--env", type=str, required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()

    master_rng = np.random.default_rng(args.seed)
    env_seed, env_action_seed, policy_seed, iter_seed = map(int, master_rng.integers(0, 2**32 - 1, 4))
    env, _, _ = create_env(args.env, env_seed, env_action_seed)
    policy_root = Path(args.policy_root)

    policy = PersistFunction.load(policy_root / "deterministic.pkl")
    @jax.jit
    def policy_fn(policy_params, obs):
        return policy(policy_params, obs).clip(-1, 1)
    
    policy_rnd = PersistFunction.load(Path("/localscratch/tchen667/soft_q_sdac/DACER-Diffusion-with-Online-RL/logs/HalfCheetah-v4/sdac_2025-05-21_22-33-38_s100_test") / "vanilla.pkl")
    iter_key = random.PRNGKey(iter_seed)
    @jax.jit
    def policy_fn_rnd(key, policy_params, obs):
        return policy_rnd(key, policy_params, obs).clip(-1, 1)
    
    for step in range(4, 50, 4):
        policy_path = policy_root / f"policy-{step*25000}-{step*5000}.pkl"
        with open(policy_path, "rb") as f:
            policy_params = pickle.load(f)
        ep_len, ep_ret, action_std, iter_key = evaluate(iter_key, env, policy_fn, policy_params, 10)
        print("ep_len", sum(ep_len)/len(ep_len))
        print("ep_ret", sum(ep_ret)/len(ep_ret))
        print("std", sum(action_std)/len(action_std))
        plt.figure()
        plt.hist(action_std, bins=100)
        plt.xlabel("Standard Deviation of Action Distribution per State")
        plt.ylabel("Count of States")
        plt.title(f"HalfCheetah-v4 {step*25}k steps")
        plt.savefig(f"figures/std_figures/action_std_hist_{step*25}k.png")