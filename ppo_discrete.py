import numpy as np
from dataclasses import dataclass
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
torch.set_float32_matmul_precision('high')

# ---------- Actor-Critic (single discrete action) ----------

class ActorCritic(nn.Module):
    def __init__(self, state_dim: int, n_actions: int, hidden_dim: int = 512):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.policy = nn.Linear(hidden_dim, n_actions)  # logits for Categorical
        self.value = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.body(x)
        logits = self.policy(h)           # [B, n_actions]
        value = self.value(h).squeeze(-1) # [B]
        return logits, value

    @torch.no_grad()
    def act(self, x: torch.Tensor) -> Tuple[int, float, float]:
        logits, value = self.forward(x)
        dist = Categorical(logits=logits)
        action = dist.sample()
        logprob = dist.log_prob(action)
        return int(action.item()), float(logprob.item()), float(value.item())

    def evaluate(self, x: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns (logprob, entropy, value) for given states/actions.
        actions: [B] long
        """
        logits, value = self.forward(x)
        dist = Categorical(logits=logits)
        logprob = dist.log_prob(actions)
        entropy = dist.entropy()
        return logprob, entropy, value


# ---------- Rollout Buffer with GAE(λ) ----------

@dataclass
class Transition:
    state: np.ndarray
    action: int
    reward: float
    done: bool
    logprob: float
    value: float
    next_value: float

class RolloutBuffer:
    def __init__(self, gamma: float, gae_lambda: float):
        self.gamma = gamma
        self.lmbda = gae_lambda
        self.data: List[Transition] = []

    def add(self, *args, **kwargs):
        self.data.append(Transition(*args, **kwargs))

    def __len__(self):
        return len(self.data)

    def clear(self):
        self.data.clear()

    def compute(self):
        """
        Returns tensors: states [N,S], actions [N], old_logprobs [N], returns [N], advantages [N]
        """
        states = np.array([t.state for t in self.data], dtype=np.float32)
        actions = np.array([t.action for t in self.data], dtype=np.int64)
        rewards = np.array([t.reward for t in self.data], dtype=np.float32)
        dones = np.array([t.done for t in self.data], dtype=np.float32)
        values = np.array([t.value for t in self.data], dtype=np.float32)
        next_values = np.array([t.next_value for t in self.data], dtype=np.float32)
        old_logprobs = np.array([t.logprob for t in self.data], dtype=np.float32)

        deltas = rewards + self.gamma * (1.0 - dones) * next_values - values
        advantages = np.zeros_like(rewards, dtype=np.float32)
        gae = 0.0
        for t in reversed(range(len(rewards))):
            gae = deltas[t] + self.gamma * self.lmbda * (1.0 - dones[t]) * gae
            advantages[t] = gae
        returns = advantages + values

        # torchify
        states_t = torch.from_numpy(states)
        actions_t = torch.from_numpy(actions)
        old_logprobs_t = torch.from_numpy(old_logprobs)
        returns_t = torch.from_numpy(returns)
        advantages_t = torch.from_numpy(advantages)
        return states_t, actions_t, old_logprobs_t, returns_t, advantages_t


# ---------- PPO Agent ----------

class PPOAgent:
    """
    PPO-Clip for single discrete action space defined by a custom action_list.
    Paper hyperparams fixed: lr=1e-5, batch_size=64, gamma=0.99, gae_lambda=0.95.
    Other PPO knobs exposed with safe defaults.
    """
    def __init__(
        self,
        state_dim: int,
        n_actions: int,
        learning_rate: float = 1e-5,     # from paper
        batch_size: int = 512,            # from paper
        gamma: float = 0.99,             # from paper
        gae_lambda: float = 0.95,        # from paper
        clip_range: float = 0.2,
        n_epochs: int = 10,
        ent_coef: float = 0.0,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        device: str | None = None,
    ):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.net = ActorCritic(state_dim, n_actions).to(self.device)
        self.optim = optim.Adam(self.net.parameters(), lr=learning_rate)

        self.buffer = RolloutBuffer(gamma=gamma, gae_lambda=gae_lambda)
        self.batch_size = batch_size
        self.gamma = gamma
        self.lmbda = gae_lambda
        self.clip_range = clip_range
        self.n_epochs = n_epochs
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm

        self._last_state = None
        self._last_action = None
        self._last_logprob = None
        self._last_value = None

    def _to_tensor(self, x: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(x, dtype=torch.float32, device=self.device)

    @torch.no_grad()
    def select_action(self, state: np.ndarray) -> int:
        s = self._to_tensor(state).unsqueeze(0)  # [1,S]
        action, logp, value = self.net.act(s)
        self._last_state = state.copy()
        self._last_action = action
        self._last_logprob = logp
        self._last_value = value
        return action

    @torch.no_grad()
    def _estimate_value(self, next_state: np.ndarray, done: bool) -> float:
        if done:
            return 0.0
        s = self._to_tensor(next_state).unsqueeze(0)
        _, v = self.net.forward(s)
        return float(v.item())

    def store(self, reward: float, done: bool, next_state: np.ndarray):
        next_val = self._estimate_value(next_state, done)
        self.buffer.add(
            state=self._last_state,
            action=self._last_action,
            reward=float(reward),
            done=bool(done),
            logprob=float(self._last_logprob),
            value=float(self._last_value),
            next_value=float(next_val),
        )

    def update(self) -> dict:
        if len(self.buffer) == 0:
            return {}

        states, actions, old_logprobs, returns, advantages = self.buffer.compute()
        # to device
        states = states.to(self.device)
        actions = actions.to(self.device)
        old_logprobs = old_logprobs.to(self.device)
        returns = returns.to(self.device)
        advantages = advantages.to(self.device)

        # normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)

        n = states.size(0)
        idx = np.arange(n)

        policy_losses, value_losses, entropies, clip_fracs = [], [], [], []

        for _ in range(self.n_epochs):
            np.random.shuffle(idx)
            for start in range(0, n, self.batch_size):
                mb = torch.as_tensor(idx[start:start + self.batch_size], dtype=torch.long, device=self.device)
                s_mb = states[mb]
                a_mb = actions[mb]
                old_lp_mb = old_logprobs[mb]
                ret_mb = returns[mb]
                adv_mb = advantages[mb]

                new_logp, entropy, values = self.net.evaluate(s_mb, a_mb)
                ratio = (new_logp - old_lp_mb).exp()

                # PPO-Clip objective
                unclipped = ratio * adv_mb
                clipped = torch.clamp(ratio, 1.0 - self.clip_range, 1.0 + self.clip_range) * adv_mb
                policy_loss = -torch.min(unclipped, clipped).mean()

                value_loss = nn.MSELoss()(values, ret_mb)
                loss = policy_loss + self.vf_coef * value_loss - self.ent_coef * entropy.mean()

                self.optim.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), self.max_grad_norm)
                self.optim.step()

                with torch.no_grad():
                    clip_frac = (torch.abs(ratio - 1.0) > self.clip_range).float().mean().item()
                policy_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
                entropies.append(float(entropy.mean().item()))
                clip_fracs.append(clip_frac)

        self.buffer.clear()
        return {
            "policy_loss": float(np.mean(policy_losses)) if policy_losses else 0.0,
            "value_loss": float(np.mean(value_losses)) if value_losses else 0.0,
            "entropy": float(np.mean(entropies)) if entropies else 0.0,
            "clip_frac": float(np.mean(clip_fracs)) if clip_fracs else 0.0,
        }
