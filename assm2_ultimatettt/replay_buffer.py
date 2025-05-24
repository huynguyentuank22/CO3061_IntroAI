import numpy as np
import random
from collections import deque

class ReplayBuffer:
    """Store and sample experiences for reinforcement learning"""
    
    def __init__(self, capacity=10000):
        """Initialize a replay buffer with fixed capacity"""
        self.buffer = deque(maxlen=capacity)
        
    def add(self, state, mlp_features, policy, reward):
        """Add an experience to the buffer"""
        self.buffer.append((state, mlp_features, policy, reward))
        
    def sample(self, batch_size):
        """Sample a batch of experiences"""
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        states, mlp_features, policies, rewards = zip(*batch)
        
        # Convert to numpy arrays for easier processing
        states = np.array(states)
        mlp_features = np.array(mlp_features)
        policies = np.array(policies)
        rewards = np.array(rewards)
        
        return states, mlp_features, policies, rewards
    
    def __len__(self):
        return len(self.buffer)
    
    def clear(self):
        """Clear the buffer"""
        self.buffer.clear()
