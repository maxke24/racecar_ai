import numpy as np
import json
import os


class QNetwork:
    """Feedforward Q-network: 8 -> 64 -> 64 -> 9 with ReLU hidden layers."""

    def __init__(self, layer_sizes=None):
        if layer_sizes is None:
            layer_sizes = [8, 64, 64, 9]
        self.layer_sizes = layer_sizes
        self.weights = []
        self.biases = []

        # Xavier initialization
        for i in range(len(layer_sizes) - 1):
            scale = np.sqrt(2.0 / layer_sizes[i])
            w = np.random.randn(layer_sizes[i], layer_sizes[i + 1]) * scale
            b = np.zeros(layer_sizes[i + 1])
            self.weights.append(w)
            self.biases.append(b)

        # Adam optimizer state
        self.m_w = [np.zeros_like(w) for w in self.weights]
        self.v_w = [np.zeros_like(w) for w in self.weights]
        self.m_b = [np.zeros_like(b) for b in self.biases]
        self.v_b = [np.zeros_like(b) for b in self.biases]
        self.adam_t = 0

    def forward(self, x, training=False):
        """Forward pass. If training, cache activations for backprop."""
        x = np.array(x, dtype=np.float64)
        if training:
            self._activations = [x]
            self._pre_activations = []

        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            z = x @ w + b
            if training:
                self._pre_activations.append(z)
            if i < len(self.weights) - 1:
                x = np.maximum(0, z)  # ReLU
            else:
                x = z  # Linear output for Q-values
            if training:
                self._activations.append(x)
        return x

    def backward(self, d_output, lr):
        """Backprop with Adam optimizer and gradient clipping."""
        self.adam_t += 1
        beta1, beta2, eps = 0.9, 0.999, 1e-8

        delta = d_output
        for i in reversed(range(len(self.weights))):
            a_prev = self._activations[i]

            # Gradients
            if a_prev.ndim == 1:
                dw = np.outer(a_prev, delta)
            else:
                dw = a_prev.T @ delta
            db = delta.sum(axis=0) if delta.ndim > 1 else delta

            # Gradient clipping
            dw = np.clip(dw, -1.0, 1.0)
            db = np.clip(db, -1.0, 1.0)

            # Adam update
            self.m_w[i] = beta1 * self.m_w[i] + (1 - beta1) * dw
            self.v_w[i] = beta2 * self.v_w[i] + (1 - beta2) * (dw ** 2)
            self.m_b[i] = beta1 * self.m_b[i] + (1 - beta1) * db
            self.v_b[i] = beta2 * self.v_b[i] + (1 - beta2) * (db ** 2)

            m_w_hat = self.m_w[i] / (1 - beta1 ** self.adam_t)
            v_w_hat = self.v_w[i] / (1 - beta2 ** self.adam_t)
            m_b_hat = self.m_b[i] / (1 - beta1 ** self.adam_t)
            v_b_hat = self.v_b[i] / (1 - beta2 ** self.adam_t)

            self.weights[i] -= lr * m_w_hat / (np.sqrt(v_w_hat) + eps)
            self.biases[i] -= lr * m_b_hat / (np.sqrt(v_b_hat) + eps)

            # Propagate delta to previous layer
            if i > 0:
                delta = delta @ self.weights[i].T
                # ReLU derivative
                delta = delta * (self._pre_activations[i - 1] > 0)

    def copy_weights_from(self, other):
        for i in range(len(self.weights)):
            self.weights[i] = other.weights[i].copy()
            self.biases[i] = other.biases[i].copy()

    def save(self, filepath):
        data = {
            "layer_sizes": self.layer_sizes,
            "weights": [w.tolist() for w in self.weights],
            "biases": [b.tolist() for b in self.biases],
            "adam_t": self.adam_t,
            "m_w": [m.tolist() for m in self.m_w],
            "v_w": [v.tolist() for v in self.v_w],
            "m_b": [m.tolist() for m in self.m_b],
            "v_b": [v.tolist() for v in self.v_b],
        }
        with open(filepath, "w") as f:
            json.dump(data, f)

    @staticmethod
    def load(filepath):
        with open(filepath) as f:
            data = json.load(f)
        net = QNetwork(data["layer_sizes"])
        net.weights = [np.array(w) for w in data["weights"]]
        net.biases = [np.array(b) for b in data["biases"]]
        if "adam_t" in data:
            net.adam_t = data["adam_t"]
            net.m_w = [np.array(m) for m in data["m_w"]]
            net.v_w = [np.array(v) for v in data["v_w"]]
            net.m_b = [np.array(m) for m in data["m_b"]]
            net.v_b = [np.array(v) for v in data["v_b"]]
        return net


class ReplayBuffer:
    """Circular experience replay buffer."""

    def __init__(self, capacity=50000):
        self.capacity = capacity
        self.buffer = []
        self.pos = 0

    def push(self, state, action, reward, next_state, done):
        transition = (state, action, reward, next_state, done)
        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self.pos] = transition
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, next_states, dones = [], [], [], [], []
        for i in indices:
            s, a, r, ns, d = self.buffer[i]
            states.append(s)
            actions.append(a)
            rewards.append(r)
            next_states.append(ns)
            dones.append(d)
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards),
            np.array(next_states),
            np.array(dones, dtype=np.float64),
        )

    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    """Deep Q-Network agent with experience replay and target network."""

    # 9 discrete actions: 3 throttle x 3 steering
    THROTTLE_VALUES = [0.0, 0.5, 1.0]
    STEERING_VALUES = [-1.0, 0.0, 1.0]
    NUM_ACTIONS = 9

    def __init__(self):
        self.q_network = QNetwork()
        self.target_network = QNetwork()
        self.target_network.copy_weights_from(self.q_network)
        self.replay_buffer = ReplayBuffer(50000)

        # Hyperparameters
        self.gamma = 0.99
        self.batch_size = 64
        self.lr = 0.001
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.9995
        self.target_update_freq = 10  # episodes
        self.min_replay_size = 1000
        self.train_every = 4  # train every N steps

        # Tracking
        self.episode = 0
        self.step_count = 0
        self.total_loss = 0.0
        self.loss_count = 0
        self.best_reward = float('-inf')

    def select_action(self, state):
        if np.random.random() < self.epsilon:
            return np.random.randint(self.NUM_ACTIONS)
        q_values = self.q_network.forward(state)
        return int(np.argmax(q_values))

    def action_to_controls(self, action_idx):
        t_idx = action_idx // 3
        s_idx = action_idx % 3
        return self.THROTTLE_VALUES[t_idx], self.STEERING_VALUES[s_idx]

    def store_transition(self, state, action, reward, next_state, done):
        self.replay_buffer.push(state, action, reward, next_state, done)

    def train_step(self):
        if len(self.replay_buffer) < self.min_replay_size:
            return
        if self.step_count % self.train_every != 0:
            return

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(
            self.batch_size
        )

        # Compute target Q-values
        # Forward pass on batch for current Q
        current_q = self.q_network.forward(states, training=True)
        next_q = self.target_network.forward(next_states)

        # TD targets
        max_next_q = np.max(next_q, axis=1)
        targets = rewards + self.gamma * max_next_q * (1.0 - dones)

        # Build gradient: only update Q-value for taken action
        d_output = np.zeros_like(current_q)
        for i in range(self.batch_size):
            a = actions[i]
            td_error = current_q[i, a] - targets[i]
            d_output[i, a] = td_error / self.batch_size

        self.q_network.backward(d_output, self.lr)

        # Track loss
        loss = np.mean(
            [(current_q[i, actions[i]] - targets[i]) ** 2 for i in range(self.batch_size)]
        )
        self.total_loss += loss
        self.loss_count += 1

    def end_episode(self, episode_reward):
        self.episode += 1
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        if episode_reward > self.best_reward:
            self.best_reward = episode_reward
        if self.episode % self.target_update_freq == 0:
            self.target_network.copy_weights_from(self.q_network)

    def get_avg_loss(self):
        if self.loss_count == 0:
            return 0.0
        return self.total_loss / self.loss_count

    def compute_reward(self, car, prev_field_dist, curr_field_dist, checkpoint_hit):
        """Per-step reward signal using BFS distance field and checkpoints."""
        if not car.alive:
            return -5.0

        reward = 0.0

        # Time penalty: forces the car to do something
        reward -= 0.01

        # Speed bonus: encourage forward movement
        reward += (car.vel_forward / car.max_speed) * 0.3

        # Checkpoint bonus: big reward for reaching next checkpoint
        if checkpoint_hit:
            reward += 10.0

        # BFS distance field progress: continuous signal toward next checkpoint
        field_delta = curr_field_dist - prev_field_dist
        reward += field_delta * 0.02

        # Center ray bonus: ray index 3 is the center (forward) ray
        rays_norm = car.get_normalized_rays()
        reward += rays_norm[3] * 0.05

        # Wall proximity penalty
        min_ray = np.min(rays_norm)
        if min_ray < 0.15:
            reward -= 0.1

        # Stall penalty
        if car.stall_timer > 30:
            reward -= 0.5

        return reward

    def save(self, filepath):
        data = {
            "epsilon": self.epsilon,
            "episode": self.episode,
            "best_reward": self.best_reward,
            "step_count": self.step_count,
        }
        # Save Q-network
        self.q_network.save(filepath)
        # Append agent metadata into the same file
        with open(filepath) as f:
            net_data = json.load(f)
        net_data["agent"] = data
        with open(filepath, "w") as f:
            json.dump(net_data, f)

    def load(self, filepath):
        if not os.path.exists(filepath):
            return False
        try:
            self.q_network = QNetwork.load(filepath)
            self.target_network = QNetwork()
            self.target_network.copy_weights_from(self.q_network)
            with open(filepath) as f:
                data = json.load(f)
            if "agent" in data:
                agent_data = data["agent"]
                self.epsilon = agent_data.get("epsilon", self.epsilon)
                self.episode = agent_data.get("episode", 0)
                self.best_reward = agent_data.get("best_reward", float('-inf'))
                self.step_count = agent_data.get("step_count", 0)
            return True
        except Exception:
            return False
