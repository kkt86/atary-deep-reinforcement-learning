from collections import deque
import numpy as np
import random

from keras.models import Sequential
from keras.layers import Activation, Conv2D, Dense, Flatten
from keras.optimizers import Adam


class DeepQNetwork:
    def __init__(self, state_size, action_space, learning_rate=0.001):
        self.state_size = state_size
        self.input_shape = self.state_size + (4,)
        self.action_space = action_space
        self.learning_rate = learning_rate

        self.epsilon = 1
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.gamma = 0.95

        self.memory = deque(maxlen=2000)
        self.model = self.generate_model()

    def act(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.action_space - 1)
        else:
            action_values = self.model.predict(self._to_keras(state))
            return np.argmax(action_values)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self, batch_size=64):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(self.model.predict(self._to_keras(state))[0])
            target_f = self.model.predict(self._to_keras(state))
            target_f[0][action] = target
            self.model.fit(self._to_keras(state), target_f, epochs=1, verbose=0)
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay

    def generate_model(self):
        model = Sequential()
        model.add(Conv2D(32, kernel_size=8, padding='valid', strides=4, input_shape=self.input_shape))
        model.add(Activation('elu'))
        model.add(Conv2D(64, kernel_size=4, strides=2, padding='valid'))
        model.add(Activation('elu'))
        model.add(Conv2D(64, kernel_size=3, strides=2, padding='valid'))
        model.add(Activation('elu'))
        model.add(Flatten())
        model.add(Dense(512))
        model.add(Activation('elu'))
        model.add(Dense(self.action_space))
        model.add(Activation('linear'))

        model.compile(optimizer=Adam(lr=self.learning_rate), loss='mse')

        return model

    def _to_keras(self, state):
        return np.expand_dims(np.stack(list(state), axis=2), axis=0)

    def save_model(self, filepath):
        self.model.save_weights(filepath)

    def load_model(self, filepath):
        self.model.load_weights(filepath)
