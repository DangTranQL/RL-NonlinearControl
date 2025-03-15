import gym
import numpy as np
import matplotlib.pyplot as plt
from ppo import Agent

env = gym.make('CartPole-v1')

n_actions = env.action_space.n
input_dims = env.observation_space.shape
alpha = 0.0003
batch_size = 5
n_epochs = 4

agent_base = Agent(n_actions=n_actions, input_dims=input_dims, gamma=0.99, alpha=alpha,
                   gae_lambda=0.95, policy_clip=0.2, batch_size=batch_size, n_epochs=n_epochs, chkpt_dir='model_base')
agent_base.load_models()

agent_episode_20 = Agent(n_actions=n_actions, input_dims=input_dims, gamma=0.99, alpha=alpha,
                         gae_lambda=0.95, policy_clip=0.2, batch_size=batch_size, n_epochs=n_epochs, chkpt_dir='model_episode=20')
agent_episode_20.load_models()

def collect_pole_angles(agent, num_frames=500):
    pole_angles = []
    observation, _ = env.reset()
    
    for frame in range(num_frames):
        action, _, _ = agent.choose_action(observation)
        observation_, _, done, _, _ = env.step(action)
        
        _, _, pole_angle, _ = observation_
        pole_angles.append(pole_angle)
        
        if done:
            break
        
        observation = observation_

    return pole_angles

pole_angles_base = collect_pole_angles(agent_base, num_frames=500)
pole_angles_episode_20 = collect_pole_angles(agent_episode_20, num_frames=500)

plt.figure()
plt.plot(pole_angles_base, label='model_base', color='blue')
plt.plot(pole_angles_episode_20, label='model_episode=20', color='red')
plt.xlabel('Frames')
plt.ylabel('Pole Angle (radians)')
plt.title('Pole Angles of Two PPO Models over 500 Frames')
plt.legend()
plt.grid(True)
plt.show()
