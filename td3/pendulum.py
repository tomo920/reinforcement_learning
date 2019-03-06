import numpy as np
import gym
from ddpg import Agent

def main():
    action_high = 2
    action_low = -2
    action_high = np.array([action_high])
    action_low = np.array([action_low])
    buffer_size = 100000
    minibatch_size = 256
    num_episode = 500

    env = gym.make("Pendulum-v0")
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.shape[0]
    agent = Agent(state_size, action_size, buffer_size, minibatch_size, action_high, action_low)
    reward_list = []
    for i_episode in range(num_episode):
        print("episode: %d" % i_episode)
        state = env.reset()
        total_reward = 0
        for t_timesteps in range(env.spec.timestep_limit):
            env.render()
            action = agent.choose_action(state)
            next_state, reward, done, info = env.step(action)
            total_reward += reward
            transition = [state, action, next_state, reward, done]
            agent.train(transition)
            state = next_state
            if (done or t_timesteps == env.spec.timestep_limit - 1):
                print("Episode finish---time steps: %d" % t_timesteps)
                print("total reward: %d" % total_reward)
                reward_list.append(total_reward)
                break
    np.save('reward', reward_list)

if __name__ == '__main__':
    main()
