import gym
from ppo import Agent 
import matplotlib.pyplot as plt
from matplotlib import animation

def save_frames_as_gif(frames, path='anim/', filename='model_layer=16-64.gif'):

    #Mess with this to change frame size
    plt.figure(figsize=(frames[0].shape[1] / 72.0, frames[0].shape[0] / 72.0), dpi=72)

    patch = plt.imshow(frames[0])
    plt.axis('off')

    def animate(i):
        patch.set_data(frames[i])

    anim = animation.FuncAnimation(plt.gcf(), animate, frames = len(frames), interval=50)
    anim.save(path + filename, writer='imagemagick', fps=60)

env = gym.make('CartPole-v1', render_mode="rgb_array")

n_actions = env.action_space.n
input_dims = env.observation_space.shape
alpha = 0.0003
batch_size = 5
n_epochs = 4

agent = Agent(n_actions=n_actions, input_dims=input_dims, gamma=0.99, alpha=alpha,
              gae_lambda=0.95, policy_clip=0.2, batch_size=batch_size, n_epochs=n_epochs, chkpt_dir="model_layer=16-64")
agent.load_models()  

observation, _ = env.reset()

angles = []
frames = []

for _ in range(300):
    action, _, _ = agent.choose_action(observation)
    observation, _, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        observation, _ = env.reset()

    cart_pos, cart_vel, pole_angle, pole_vel = observation
    angles.append(pole_angle)

    frames.append(env.render())
        
env.close()
save_frames_as_gif(frames)

plt.figure()
plt.plot(angles, color='blue')
plt.xlabel('Frames')
plt.ylabel('Pole Angle (radians)')
plt.grid(True)
plt.show()