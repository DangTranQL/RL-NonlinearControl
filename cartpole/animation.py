import gym
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from ppo import Agent 

env = gym.make('CartPole-v1')

n_actions = env.action_space.n
input_dims = env.observation_space.shape
alpha = 0.0003
batch_size = 5
n_epochs = 4

agent = Agent(n_actions=n_actions, input_dims=input_dims, gamma=0.99, alpha=alpha,
              gae_lambda=0.95, policy_clip=0.2, batch_size=batch_size, n_epochs=n_epochs)
agent.load_models()  


fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(111, xlim=(-2.5, 2.5), ylim=(-1, 2))


cart, = ax.plot([], [], 'bo', markersize=12)  
pole, = ax.plot([], [], 'r-', lw=2)  
time_text = ax.text(-2, 1.8, '', fontsize=15)


def init():
    cart.set_data([], [])
    pole.set_data([], [])
    time_text.set_text('')
    return cart, pole, time_text

def update(frame):
    observation, _ = env.reset()

    action, _, _ = agent.choose_action(observation)
    
    observation_, _, _, _, _ = env.step(action)

    cart_position, _, pole_angle, _ = observation_

    cart.set_data([cart_position], [0])

    pole_length = 0.5
    pole_x = cart_position + np.sin(pole_angle) * pole_length
    pole_y = np.cos(pole_angle) * pole_length

    pole.set_data([cart_position, pole_x], [0, pole_y])

    time_text.set_text(f"Frame: {frame}")

    return cart, pole, time_text

num_frames = 500
anim = animation.FuncAnimation(fig, update, frames=num_frames, init_func=init, interval=90, blit=True)

# anim.save('anim/cartpole_animation.gif', writer='imagemagick', fps=30)

plt.show()
