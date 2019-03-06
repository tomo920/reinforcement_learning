import numpy as np
import tensorflow as tf
from actor import Actor
from critic import Critic

gamma = 1

class Agent:
    def __init__(self, state_size, action_size):
        self.actor = Actor(state_size, action_size)
        self.critic = Critic(state_size, action_size)
        self.state_batch = []
        self.action_batch = []
        self.next_state_batch = []
        self.reward_batch = []
        self.done_bach = []

    def choose_action(self, state):
        state = np.array([state])
        return self.actor.action(state)[0][0]

    def store(self, state, action, next_state, reward, done):
        self.state_batch.append(state)
        self.action_batch.append(action)
        self.next_state_batch.append(next_state)
        self.reward_batch.append(reward)
        self.done_bach.append(done)

    def train(self):
        state_batch = np.vstack(self.state_batch)
        action_batch = np.vstack(self.action_batch)
        next_state_batch = np.vstack(self.next_state_batch)
        reward_batch = np.vstack(self.reward_batch)
        done_bach = np.vstack(self.done_bach)
        next_action_batch = self.actor.action(next_state_batch)

        self.state_batch = []
        self.action_batch = []
        self.next_state_batch = []
        self.reward_batch = []
        self.done_bach = []
