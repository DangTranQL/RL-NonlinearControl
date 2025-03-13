import gym
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from ppo import Agent

env = gym.make('Acrobot-v1')
n_actions = env.action_space.n
input_dims = env.observation_space.shape
alpha = 0.0003
batch_size = 5
n_epochs = 4

agent = Agent(n_actions=n_actions, input_dims=input_dims, gamma=0.99, alpha=alpha,
              gae_lambda=0.95, policy_clip=0.2, batch_size=batch_size, n_epochs=n_epochs)
agent.load_models()

fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(111, xlim=(-2, 2), ylim=(-2, 2))

link1, = ax.plot([], [], 'b-', lw=2)  
link2, = ax.plot([], [], 'r-', lw=2)  
joints, = ax.plot([], [], 'ko', markersize=5)  
time_text = ax.text(-1.8, 1.8, '', fontsize=12)

def init():
    link1.set_data([], [])
    link2.set_data([], [])
    joints.set_data([], [])
    time_text.set_text('')
    return link1, link2, joints, time_text

def update(frame):
    observation, _ = env.reset()
    done = False
    while not done:
        action, _, _ = agent.choose_action(observation)

        observation_, _, done, _, _ = env.step(action)

        cos_theta1, sin_theta1, cos_theta2, sin_theta2, _, _ = observation_
        theta1 = np.arctan2(sin_theta1, cos_theta1)
        theta2 = np.arctan2(sin_theta2, cos_theta2)

        l1, l2 = 1.0, 1.0  

        x1 = l1 * np.sin(theta1)
        y1 = -l1 * np.cos(theta1)

        x2 = x1 + l2 * np.sin(theta2)
        y2 = y1 - l2 * np.cos(theta2)

      
        link1.set_data([0, x1], [0, y1]) 
        link2.set_data([x1, x2], [y1, y2])  
        joints.set_data([x1, x2], [y1, y2])  
        time_text.set_text(f"Frame: {frame}")

        return link1, link2, joints, time_text

num_frames = 500
anim = animation.FuncAnimation(fig, update, frames=num_frames, init_func=init, interval=90, blit=True)

plt.show()
