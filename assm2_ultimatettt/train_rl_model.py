# Requirements: torch, numpy, tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from collections import deque
from tqdm import tqdm
from model import UltimateTicTacToeModel


def preprocess_state(board, macroboard, current_player, valid_moves, move_number, agent_level, game_result):
    board = np.array(board).reshape(9, 9)
    macro = np.array(macroboard).reshape(3, 3)
    macro_full = np.repeat(np.repeat(macro, 3, axis=0), 3, axis=1)
    player = np.full((9, 9), current_player)
    valid = np.zeros((9, 9))
    for move_id in valid_moves:
        i, j = divmod(move_id, 9)
        valid[i][j] = 1

    board_tensor = np.stack([board, macro_full, player, valid], axis=0).astype(np.float32)
    move_number = move_number / 81.0
    mlp_features = np.array([move_number, agent_level, game_result], dtype=np.float32)
    mask = np.full(81, -np.inf, dtype=np.float32)
    mask[valid_moves] = 0.0

    return (
        torch.tensor(board_tensor).unsqueeze(0),
        torch.tensor(mlp_features).unsqueeze(0),
        torch.tensor(mask).unsqueeze(0)
    )


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, *args):
        self.buffer.append(tuple(args))

    def sample(self, batch_size):
        samples = random.sample(self.buffer, batch_size)
        return map(torch.stack, zip(*samples))

    def __len__(self):
        return len(self.buffer)


class Agent:
    def __init__(self, model, lr=1e-4, gamma=0.99, epsilon_start=1.0, epsilon_end=0.05, decay=5000):
        self.model = model
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_end
        self.decay = decay
        self.step_count = 0

    def select_action(self, board_tensor, mlp_features, valid_mask):
        if random.random() < self.epsilon:
            valid_indices = (valid_mask[0] == 0.0).nonzero(as_tuple=False).squeeze(1)
            return random.choice(valid_indices).item()
        else:
            with torch.no_grad():
                logits, _ = self.model(board_tensor, mlp_features)
                logits = logits + valid_mask  # Apply mask
                return torch.argmax(logits, dim=1).item()

    def train_step(self, batch):
        boards, features, actions, rewards, next_boards, next_features, dones, next_masks = batch

        logits, _ = self.model(boards, features)
        action_q = logits.gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_q_logits, _ = self.model(next_boards, next_features)
            next_q_logits += next_masks
            max_next_q = next_q_logits.max(1)[0]
            target_q = rewards + self.gamma * max_next_q * (1 - dones)

        loss = F.mse_loss(action_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.step_count += 1
        self.epsilon = max(self.epsilon_min, self.epsilon * np.exp(-1. * self.step_count / self.decay))


# Placeholder game env (you should replace this with your actual UTTT game logic)
class DummyEnv:
    def reset(self):
        self.board = [0] * 81
        self.macroboard = [0] * 9
        self.valid_moves = [i for i in range(81)]
        self.move_number = 0
        self.player = 1
        return self.board, self.macroboard, self.player, self.valid_moves

    def step(self, action):
        reward = random.choice([-1, 0, 1])
        done = random.random() < 0.1
        self.board = [random.choice([0, 1, 2]) for _ in range(81)]
        self.valid_moves = random.sample(range(81), 10)
        self.move_number += 1
        return self.board, reward, done, {}


# Training loop
if __name__ == '__main__':
    model = UltimateTicTacToeModel()
    model.load_state_dict(torch.load('model.pt'))
    agent = Agent(model)
    buffer = ReplayBuffer(capacity=10000)
    env = DummyEnv()
    episodes = 1000
    batch_size = 32

    for ep in tqdm(range(episodes)):
        board, macroboard, player, valid_moves = env.reset()
        done = False
        while not done:
            bt, mf, mask = preprocess_state(board, macroboard, player, valid_moves, move_number=0, agent_level=1.0, game_result=0.0)
            action = agent.select_action(bt, mf, mask)
            next_board, reward, done, _ = env.step(action)

            nbt, nmf, nmask = preprocess_state(next_board, macroboard, player, valid_moves, move_number=1, agent_level=1.0, game_result=0.0)

            buffer.push(bt.squeeze(0), mf.squeeze(0), torch.tensor(action), torch.tensor(reward, dtype=torch.float32),
                        nbt.squeeze(0), nmf.squeeze(0), torch.tensor(done, dtype=torch.float32), nmask.squeeze(0))

            if len(buffer) > batch_size:
                batch = buffer.sample(batch_size)
                agent.train_step(batch)

            board = next_board
