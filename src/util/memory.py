from collections import deque
import random


class Memory:
    def __init__(self, maxlen=10000):
        self.maxlen = maxlen
        self.memory = deque(maxlen=self.maxlen)

    def add(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def sample(self, batch_size=64):
        return random.sample(self.memory, batch_size)
