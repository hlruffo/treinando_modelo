from __future__ import annotations

import pickle
import random 
from collections import defaultdict


class QLearningAgent:
    def __init__(
        self,
        alpha: float = 0.3,
        gamma: float = 0.9,
        epsilon: float = 0.3,
        epsilon_min: float = 0.05,
    ) -> None:
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.q: defaultdict[tuple, float] = defaultdict(float)
        
    def choose_action(self, state: tuple, available_actions: list[int]) -> int:
        if random.random() < self.epsilon:
            return random.choice(available_actions)
        q_values = {a: self.q[(state, a)] for a in available_actions}
        return max(q_values, key=q_values.get)
    
    def update(self, state: tuple, action: int, reward: float, next_state: tuple, next_available: list[int], done: bool) -> None:
        current_q = self.q[(state, action)]
        if done or not next_available:
            target = reward
        else:
            max_next_q = max(self.q[(next_state, a)] for a in next_available)
            target = reward + self.gamma * max_next_q
        self.q[(state, action)] += self.alpha * (target - current_q)
        
    def decay_epsilon(self, factor: float = 0.95) -> None:
        self.epsilon = max(self.epsilon_min, self.epsilon * factor)
        
    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump(self, f)
            
    @staticmethod
    def load(path: str) -> "QLearningAgent":
        with open(path, "rb") as f:
            return pickle.load(f)