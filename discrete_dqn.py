import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
torch.set_float32_matmul_precision('high')
# --- Q-Network for custom discrete action space ---
class QNetwork(nn.Module):
    def __init__(self, state_dim, n_actions, hidden_dim=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_actions)
        )

    def forward(self, x):
        return self.net(x)  # shape: [batch_size, n_actions]

# --- Replay Buffer ---
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
        self.obs_shape = None

    def push(self, state, action_index, reward, next_state, done):
        if self.obs_shape is None:
            self.obs_shape = state.shape
        elif state.shape != self.obs_shape:
            raise ValueError(f"New observation shape {state.shape} does not match expected shape {self.obs_shape}")
        self.buffer.append((state, action_index, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        s, a, r, ns, d = zip(*batch)
        return (
            np.array(s),
            np.array(a),
            np.array(r),
            np.array(ns),
            np.array(d)
        )

    def __len__(self):
        return len(self.buffer)

# --- Double DQN Agent with custom discrete action list ---
class DQNAgent:
    def __init__(self, state_dim, action_list, lr=1e-3, gamma=0.99,
                 epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=500):

        self.action_list = action_list  # e.g. [0, 1, 2, ..., 112] sparse list
        self.n_actions = len(self.action_list)

        self.q_net = QNetwork(state_dim, self.n_actions)
        self.target_net = QNetwork(state_dim, self.n_actions)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.replay_buffer = ReplayBuffer(capacity=100000)
        self.batch_size = 512
        self.gamma = gamma
        self.update_target_every = 100
        self.steps = 0

        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.q_net.to(self.device)
        self.target_net.to(self.device)

    def select_action(self, state):
        epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
                  np.exp(-1. * self.steps / self.epsilon_decay)
        self.steps += 1

        if random.random() < epsilon:
            action_index = random.randint(0, self.n_actions - 1)
        else:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                q_values = self.q_net(state_tensor)
                action_index = q_values.argmax(dim=1).item()

        return action_index  # index into self.action_list

    def decode_action(self, action_index):
        return self.action_list[action_index]

    def update(self):
        if len(self.replay_buffer) < self.batch_size:
            return

        s, a, r, ns, d = self.replay_buffer.sample(self.batch_size)

        s = torch.FloatTensor(s).to(self.device)
        ns = torch.FloatTensor(ns).to(self.device)
        r = torch.FloatTensor(r).unsqueeze(1).to(self.device)
        d = torch.FloatTensor(d).unsqueeze(1).to(self.device)
        a = torch.LongTensor(a).unsqueeze(1).to(self.device)

        q_curr = self.q_net(s)
        q_val = q_curr.gather(1, a)

        with torch.no_grad():
            q_next_online = self.q_net(ns)
            q_next_target = self.target_net(ns)
            next_actions = q_next_online.argmax(dim=1, keepdim=True)
            q_target_val = q_next_target.gather(1, next_actions)
            target = r + self.gamma * q_target_val * (1 - d)

        loss = nn.MSELoss()(q_val, target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.steps % self.update_target_every == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())
