"""
Shared infrastructure for offline RL training on .hdf5 datasets.
Compatible with CORL algorithm classes and gymnasium environments.
"""
import os
import random
import numpy as np
import h5py
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Union

TensorBatch = List[torch.Tensor]

# ── D4RL Reference Scores (for normalized score computation) ──
# These are the standard min/max episode returns from D4RL
REF_MIN_SCORE = {
    "halfcheetah": -280.178953,
    "hopper": -20.272305,
    "walker2d": 1.629008,
}
REF_MAX_SCORE = {
    "halfcheetah": 12135.0,
    "hopper": 3234.3,
    "walker2d": 4592.3,
}

# ── Env name mapping: dataset prefix → gymnasium env name ──
ENV_NAME_MAP = {
    "halfcheetah": "HalfCheetah-v4",
    "hopper": "Hopper-v4",
    "walker2d": "Walker2d-v4",
}


def get_normalized_score(env_name: str, score: float) -> float:
    """Compute D4RL-style normalized score (0-100+ scale)."""
    for prefix in REF_MIN_SCORE:
        if prefix in env_name.lower():
            min_score = REF_MIN_SCORE[prefix]
            max_score = REF_MAX_SCORE[prefix]
            return (score - min_score) / (max_score - min_score)
    return score


def get_gymnasium_env_name(dataset_name: str) -> str:
    """Map dataset prefix to gymnasium environment name."""
    for prefix, env_name in ENV_NAME_MAP.items():
        if prefix in dataset_name.lower():
            return env_name
    raise ValueError(f"Unknown dataset: {dataset_name}")


def compute_mean_std(states: np.ndarray, eps: float = 1e-3) -> Tuple[np.ndarray, np.ndarray]:
    mean = states.mean(0)
    std = states.std(0) + eps
    return mean, std


def normalize_states(states: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (states - mean) / std


def set_seed(seed: int, env: Optional = None, deterministic_torch: bool = False):
    if env is not None:
        env.reset(seed=seed)
        try:
            env.action_space.seed(seed)
        except AttributeError:
            pass  # gymnasium action_space may not have seed()
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(deterministic_torch)


class ReplayBuffer:
    """Replay buffer for offline RL, loads .hdf5 datasets in d4rl format."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        buffer_size: int,
        device: str = "cpu",
    ):
        self._buffer_size = buffer_size
        self._pointer = 0
        self._size = 0

        self._states = torch.zeros(
            (buffer_size, state_dim), dtype=torch.float32, device=device
        )
        self._actions = torch.zeros(
            (buffer_size, action_dim), dtype=torch.float32, device=device
        )
        self._rewards = torch.zeros((buffer_size, 1), dtype=torch.float32, device=device)
        self._next_states = torch.zeros(
            (buffer_size, state_dim), dtype=torch.float32, device=device
        )
        self._dones = torch.zeros((buffer_size, 1), dtype=torch.float32, device=device)
        self._device = device

    def _to_tensor(self, data: np.ndarray) -> torch.Tensor:
        return torch.tensor(data, dtype=torch.float32, device=self._device)

    def load_hdf5_dataset(self, filepath: str):
        """Load dataset from .hdf5 file in d4rl format."""
        if self._size != 0:
            raise ValueError("Trying to load data into non-empty replay buffer")

        with h5py.File(filepath, "r") as f:
            n_transitions = f["observations"].shape[0]
            if n_transitions > self._buffer_size:
                raise ValueError(
                    "Replay buffer is smaller than the dataset you are trying to load!"
                )
            self._states[:n_transitions] = self._to_tensor(f["observations"][:])
            self._actions[:n_transitions] = self._to_tensor(f["actions"][:])
            self._rewards[:n_transitions] = self._to_tensor(f["rewards"][:][..., None])
            self._next_states[:n_transitions] = self._to_tensor(f["next_observations"][:])
            self._dones[:n_transitions] = self._to_tensor(f["terminals"][:][..., None])

        self._size += n_transitions
        self._pointer = min(self._size, n_transitions)
        print(f"Dataset size: {n_transitions}")

    def load_d4rl_dataset(self, data: Dict[str, np.ndarray]):
        """Load data from a d4rl-format dict (compatibility with CORL)."""
        if self._size != 0:
            raise ValueError("Trying to load data into non-empty replay buffer")
        n_transitions = data["observations"].shape[0]
        if n_transitions > self._buffer_size:
            raise ValueError(
                "Replay buffer is smaller than the dataset you are trying to load!"
            )
        self._states[:n_transitions] = self._to_tensor(data["observations"])
        self._actions[:n_transitions] = self._to_tensor(data["actions"])
        self._rewards[:n_transitions] = self._to_tensor(data["rewards"][..., None])
        self._next_states[:n_transitions] = self._to_tensor(data["next_observations"])
        self._dones[:n_transitions] = self._to_tensor(data["terminals"][..., None])
        self._size += n_transitions
        self._pointer = min(self._size, n_transitions)
        print(f"Dataset size: {n_transitions}")

    def sample(self, batch_size: int) -> TensorBatch:
        indices = np.random.randint(0, min(self._size, self._pointer), size=batch_size)
        states = self._states[indices]
        actions = self._actions[indices]
        rewards = self._rewards[indices]
        next_states = self._next_states[indices]
        dones = self._dones[indices]
        return [states, actions, rewards, next_states, dones]


def load_hdf5_as_dict(filepath: str) -> Dict[str, np.ndarray]:
    """Load .hdf5 dataset into a d4rl-compatible dict."""
    with h5py.File(filepath, "r") as f:
        data = {
            "observations": f["observations"][:],
            "actions": f["actions"][:],
            "rewards": f["rewards"][:],
            "next_observations": f["next_observations"][:],
            "terminals": f["terminals"][:],
        }
    return data


def return_reward_range(dataset: Dict, max_episode_steps: int = 1000) -> Tuple[float, float]:
    """Compute min and max episode returns from a dataset."""
    returns, lengths = [], []
    ep_ret, ep_len = 0.0, 0
    for r, d in zip(dataset["rewards"], dataset["terminals"]):
        ep_ret += float(r)
        ep_len += 1
        if d or ep_len == max_episode_steps:
            returns.append(ep_ret)
            lengths.append(ep_len)
            ep_ret, ep_len = 0.0, 0
    lengths.append(ep_len)
    assert sum(lengths) == len(dataset["rewards"])
    return min(returns), max(returns)


def modify_reward(
    dataset: Dict,
    env_name: str,
    max_episode_steps: int = 1000,
    reward_scale: float = 1.0,
    reward_bias: float = 0.0,
):
    """Modify rewards: normalize by return range then apply scale/bias."""
    if any(s in env_name.lower() for s in ("halfcheetah", "hopper", "walker2d")):
        min_ret, max_ret = return_reward_range(dataset, max_episode_steps)
        dataset["rewards"] /= max_ret - min_ret
        dataset["rewards"] *= max_episode_steps
    dataset["rewards"] = dataset["rewards"] * reward_scale + reward_bias


@torch.no_grad()
def eval_actor(
    env,
    actor: nn.Module,
    device: str,
    n_episodes: int = 10,
    seed: int = 0,
) -> np.ndarray:
    """Evaluate an actor in the environment, returning episode returns."""
    actor.eval()
    episode_rewards = []
    for ep_idx in range(n_episodes):
        state, _ = env.reset(seed=seed + ep_idx)
        done = False
        episode_reward = 0.0
        while not done:
            if isinstance(state, tuple):
                state = state[0]
            action = actor.act(state.reshape(1, -1), device)
            step_result = env.step(action)
            if len(step_result) == 5:
                # gymnasium API: (obs, reward, terminated, truncated, info)
                state, reward, terminated, truncated, _ = step_result
                done = terminated or truncated
            elif len(step_result) == 4:
                # old gym API: (obs, reward, done, info)
                state, reward, done, _ = step_result
            else:
                state, reward, done = step_result[0], step_result[1], step_result[2]
            episode_reward += float(reward)
        episode_rewards.append(episode_reward)
    actor.train()
    return np.asarray(episode_rewards)
