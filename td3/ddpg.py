import numpy as np
import random
import tensorflow as tf
from actor import Actor
from critic import Critic
from ounoise import OUnoise
from buffer import Buffer

gamma = 0.9
c = [1.0]
sigma = 1.0

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
        self.action_size = action_size
        self.training = False
        self.policy_update = False
        sess.run(tf.global_variables_initializer())

    def clip_num(self, num, num_high, num_low):
        for i, (high, low) in enumerate(zip(num_high, num_low)):
            if num[i] > high:
                num[i] = high
            elif num[i] < low:
                num[i] = low
        return num

    def choose_action(self, state):
        state = np.array([state])
        action = self.actor.action(state)[0]
        action = action+self.noiser._noise()
        return self.clip_num(action, self.action_high, self.action_low)

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
            next_noise = []
            for _ in range(self.minibatch_size):
                noise = sigma*np.random.randn(self.action_size) #noise for next action
                noise = self.clip_num(noise, np.array(c*self.action_size), -np.array(c*self.action_size))
                next_noise.append(noise)
            next_noise = np.vstack(next_noise)
            next_action_batch = self.clip_num(next_action_batch+next_noise, self.action_high, self.action_low)
            next_q1 = self.critic.next_q_value(self.critic.q1, next_state_batch, next_action_batch)
            next_q2 = self.critic.next_q_value(self.critic.q2, next_state_batch, next_action_batch)
            next_q = np.vstack(np.amin(np.concatenate([next_q1, next_q2], axis=1), axis=1))
            q_target = reward_batch+(1-done_batch)*gamma*next_q
            self.critic.train(state_batch, action_batch, q_target)
            if self.policy_update:
                action_grad_batch = self.critic.action_grad(state_batch, self.actor.action(state_batch))
                self.actor.train(state_batch, action_grad_batch)
                self.critic.update_target()
                self.actor.update_target()
                self.policy_update = False
            else:
                self.policy_update = True
