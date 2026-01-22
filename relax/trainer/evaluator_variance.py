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
from jax import random
from tensorboardX import SummaryWriter

from relax.env import create_env
from relax.utils.persistence import PersistFunction

def evaluate(key, env, policy_fn, policy_fn_rnd, policy_params, num_episodes):
    ep_len_list = []
    ep_ret_list = []
    action_std_list = []
    num_episodes = 5
    for _ in range(num_episodes):
        obs, _ = env.reset()
        ep_len = 0
        ep_ret = 0.0
        while True:
            if ep_len % 20 == 0:
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

class Logger(object):

	def __init__(self, log_dir):
		self.path = os.path.join(log_dir, 'log.csv')
		with open(self.path, mode='w', newline='') as f:
			writer = csv.writer(f)
			writer.writerow(['step', 'avg_ret', 'std_ret', 'avg_action_std'])

	def log(self, step, avg_ret, std_ret, action_std):
		with open(self.path, mode='a', newline='') as f:
			writer = csv.writer(f)
			writer.writerow([step, avg_ret, std_ret, action_std])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("policy_root", type=Path)
    parser.add_argument("--env", type=str, required=True)
    parser.add_argument("--num_episodes", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()

    master_rng = np.random.default_rng(args.seed)
    env_seed, env_action_seed, policy_seed, iter_seed = map(int, master_rng.integers(0, 2**32 - 1, 4))
    env, _, _ = create_env(args.env, env_seed, env_action_seed)

    policy = PersistFunction.load(args.policy_root / "deterministic.pkl")
    @jax.jit
    def policy_fn(policy_params, obs):
        return policy(policy_params, obs).clip(-1, 1)
    
    policy_rnd = PersistFunction.load(args.policy_root / "vanilla.pkl")
    iter_key = random.PRNGKey(iter_seed)
    @jax.jit
    def policy_fn_rnd(key, policy_params, obs):
        return policy_rnd(key, policy_params, obs).clip(-1, 1)

    logger = Logger(args.policy_root)

    while payload := sys.stdin.readline():
        step, policy_path = payload.strip().split(",", maxsplit=1)
        step = int(step)
        with open(policy_path, "rb") as f:
            policy_params = pickle.load(f)

        ep_len_list, ep_ret_list, action_std_list, iter_key = evaluate(iter_key, env, policy_fn, policy_fn_rnd, policy_params, args.num_episodes)

        ep_len = np.array(ep_len_list)
        ep_ret = np.array(ep_ret_list)
        action_std = np.array(action_std_list)
        # logger.add_scalar("evaluate/episode_length", ep_len_mean.mean(), step)
        # logger.add_scalar("evaluate/episode_return", ep_ret_mean.mean(), step)
        # # logger.add_histogram("evaluate/episode_length", ep_len_mean, step)
        # # logger.add_histogram("evaluate/episode_return", ep_ret_mean, step)
        # logger.flush()
        logger.log(step, ep_ret.mean(), ep_ret.std(), action_std.mean())
