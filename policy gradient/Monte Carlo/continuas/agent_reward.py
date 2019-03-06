import numpy as np
import tensorflow as tf
from policy import Policy

gamma = 1

class Agent:
    def __init__(self, state_size, action_size, action_high, action_low, sample_num):
        sess = tf.Session()
        self.policy = Policy(sess, state_size, action_size, action_high, action_low, sample_num)
        self.state_batch = []
        self.action_batch = []
        self.reward_list = []
        self.step_list = []
        self.weight_bach = []
        self.sample_num = sample_num
        sess.run(tf.global_variables_initializer())

    def choose_action(self, state):
        return self.policy.choose_action(state)

    def store(self, state, action, reward):
        self.state_batch.append(state)
        self.action_batch.append(action)
        self.reward_list.append(reward)

    def train(self):
        state_batch = np.vstack(self.state_batch)
        action_batch = np.vstack(self.action_batch)
        t = 0
        for i in range(self.sample_num):
            tlast = t+self.step_list[i]
            for _ in range(self.step_list[i]):
                weight = 0.0
                for n in range(t, tlast):
                    weight+=self.reward_list[n]*np.power(gamma, (n-t))
                self.weight_bach.append(weight)
                t+=1
        weight_bach = np.vstack(self.weight_bach)
        self.policy.train(state_batch, action_batch, weight_bach)
        self.state_batch = []
        self.action_batch = []
        self.reward_list = []
        self.step_list = []
        self.weight_bach = []
