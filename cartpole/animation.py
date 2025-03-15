import gym
from ppo import Agent 
import matplotlib.pyplot as plt

env = gym.make('CartPole-v1', render_mode="human")

n_actions = env.action_space.n
input_dims = env.observation_space.shape
alpha = 0.0003
batch_size = 5
n_epochs = 4

agent = Agent(n_actions=n_actions, input_dims=input_dims, gamma=0.99, alpha=alpha,
              gae_lambda=0.95, policy_clip=0.2, batch_size=batch_size, n_epochs=n_epochs, chkpt_dir="model_episode=2000")
agent.load_models()  

observation, _ = env.reset(seed=3)
observation[2] = 0.0

angles = []

for _ in range(500):
    action, _, _ = agent.choose_action(observation)
    observation, _, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        observation, _ = env.reset(seed=3)
        observation[2] = 0.0

    cart_pos, cart_vel, pole_angle, pole_vel = observation
    angles.append(pole_angle)

    env.render()
        
env.close()

plt.figure()
plt.plot(angles, color='blue')
plt.xlabel('Frames')
plt.ylabel('Pole Angle (radians)')
# plt.title('Pole Angles of Two PPO Models over 500 Frames')
plt.grid(True)
plt.show()