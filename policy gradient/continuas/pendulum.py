import numpy as np
import gym
from agent_reward import Agent

def main():
    env = gym.make("Pendulum-v0")
    action_high = 2
    action_low = -2
    action_high = np.array([action_high])
    action_low = np.array([action_low])
    step_num = env.spec.timestep_limit
    sample_num = 50
    num_episode = 300

    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    print(action_size)
    agent = Agent(state_size, action_size, action_high, action_low, sample_num)
    reward_list = []
    for i_episode in range(num_episode):
        print("episode: %d" % i_episode)
        total_reward = 0.0
        for i_sample in range(sample_num):
            state = env.reset()
            for t_timesteps in range(step_num):
                env.render()
                action = agent.choose_action(state)
                next_state, reward, done, info = env.step(action)
                total_reward+=reward
                agent.store(state, action, reward)
                if t_timesteps == step_num-1:
                    agent.step_list.append(t_timesteps+1)
                    break
                state = next_state
        print("total reward: %d" % total_reward)
        reward_list.append(total_reward)
        agent.train()
    np.save('reward', reward_list)

if __name__ == '__main__':
    main()
