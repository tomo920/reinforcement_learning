import numpy as np
import random
import tensorflow as tf
from actor import Actor
from critic import Critic
from ounoise import OUnoise
from buffer import Buffer

gamma = 0.9

class Agent:
    def __init__(self, state_size, action_size, buffer_size, minibatch_size, action_high, action_low):
        sess = tf.Session()
        self.actor = Actor(sess, state_size, action_size, action_high, action_low)
        self.critic = Critic(sess, state_size, action_size)
        self.noiser = OUnoise(action_size, action_high, action_low)
        self.buffer = Buffer(buffer_size)
        self.minibatch_size = minibatch_size
        self.action_high = action_high
        self.action_low = action_low
        self.training = False
        sess.run(tf.global_variables_initializer())

    def choose_action(self, state):
        state = np.array([state])
        action = self.actor.action(state)[0]
        action = action+self.noiser._noise()
        #clip
        for i, (high, low) in enumerate(zip(self.action_high, self.action_low)):
            if action[i] > high:
                action[i] = high
            elif action[i] < low:
                action[i] = low
        return action

    def train(self, transition):
        self.buffer.store(transition)
        if not self.training and len(self.buffer.transitions) == self.minibatch_size:
            self.training = True
        if self.training:
            minibatch = np.array(random.sample(self.buffer.transitions, self.minibatch_size))
            state_batch = np.vstack(minibatch[:,0])
            action_batch = np.vstack(minibatch[:,1])
            next_state_batch = np.vstack(minibatch[:,2])
            reward_batch = np.vstack(minibatch[:,3])
            done_batch = np.vstack(minibatch[:,4])
            next_action_batch = self.actor.next_action(next_state_batch)
            q_target = reward_batch+(1-done_batch)*gamma*self.critic.next_q_value(next_state_batch, next_action_batch)
            self.critic.train(state_batch, action_batch, q_target)
            action_grad_batch = self.critic.action_grad(state_batch, self.actor.action(state_batch))
            self.actor.train(state_batch, action_grad_batch)
            self.critic.update_target()
            self.actor.update_target()
