from collections import deque
import numpy as np
import random

import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

import gym


class DQNAgent:
    def __init__(self, state_size, action_size):
        self.render = True
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=10000)
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.005
        self.epsilon_decay = 0.99
        self.learning_rate = 0.001
        self.train_start = 1000
        self.model = self._build_model()

    def _build_model(self):
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        else:
            action_values = self.model.predict(state)
            return np.argmax(action_values[0])

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(self.model.predict(next_state)[0])
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save_model(self, path='models/dqn-cartpole.h5'):
        self.model.save(path)

    def load_model(self, path):
        self.model = keras.models.load_model(path)


if __name__ == '__main__':
    # make environment
    env = gym.make('CartPole-v0')

    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = DQNAgent(state_size, action_size)

    episodes = 1000
    batch_size = 32
    done = False

    for e in range(episodes):
        # reset the state at the beginning of each episode
        total_reward = 0
        state = env.reset()
        state = np.reshape(state, [1, 4])

        # time_t represents each frame of the game, our goal is to keep the pole upright as long as possible
        for time_t in range(500):
            # decide action
            action = agent.act(state)

            # execute action and get next state and reward
            next_state, reward, done, info = env.step(action)
            next_state = np.reshape(next_state, [1, 4])
            total_reward += reward
            reward = reward if not done else -100

            # remember the previous state, action, reward, next_state and done
            agent.remember(state, action, reward, next_state, done)

            # make next_state the current one
            state = next_state

            # render environment
            if agent.render:
                env.render()

            # if the agent drops the pole
            if done:
                print("episode: {}/{}, score: {}, epsilon: {:.2}".format(e, episodes, total_reward, agent.epsilon))
                break

            # train the agent with the experience of the episodes
            if len(agent.memory) > agent.train_start:
                agent.replay(batch_size)

        if e > 0 & e % 50 == 0:
            agent.save_model()
