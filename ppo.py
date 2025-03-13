import os
import numpy as np
import torch as T
import torch.nn as nn
import torch.optim as optim
import time
from torch.distributions.categorical import Categorical

class PPOMemory:
    """
        Data batch generation and storage
        
        Attr:
            states (array): State space
            probs (array): Probabilities of actions
            vals (array): Values after sequence of actions
            actions (array): Action space
            rewards (array): Rewards of actions
            dones (array): Actions done
    """
    
    def __init__(self, batch_size):
        self.states = []
        self.probs = []
        self.vals = []
        self.actions = []
        self.rewards = []
        self.dones = []

        self.batch_size = batch_size

    def generate_batches(self):
        n_states = len(self.states)
        batch_start = np.arange(0, n_states, self.batch_size)
        indices = np.arange(n_states, dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i:i+self.batch_size] for i in batch_start]

        return np.array(self.states), np.array(self.actions), np.array(self.probs), np.array(self.vals), np.array(self.rewards), np.array(self.dones), batches

    def store_memory(self, state, action, probs, vals, reward, done):
        self.states.append(state)
        self.actions.append(action)
        self.probs.append(probs)
        self.vals.append(vals)
        self.rewards.append(reward)
        self.dones.append(done)

    def clear_memory(self):
        self.states = []
        self.probs = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.vals = []

class Actor(nn.Module):
    """ Policy-based model that maps state-action space to learn optimal policy
    
        Attr:
            n_actions (int): Action space
            input_dims (int): Observation space
            alpha (float): Learning rate
            fc1_dims (int): NN layer dimension
            fc2_dim2 (int): NN layer dimension
            chkpt_dir (string): Model's weights storage path        
    """
    def __init__(self, n_actions, input_dims, alpha,
            fc1_dims=256, fc2_dims=256, chkpt_dir='model'):
        super(Actor, self).__init__()

        self.checkpoint_file = os.path.join(chkpt_dir, 'actor')
        os.makedirs(chkpt_dir, exist_ok=True)
        
        self.actor = nn.Sequential(
                nn.Linear(*input_dims, fc1_dims),
                nn.ReLU(),
                nn.Linear(fc1_dims, fc2_dims),
                nn.ReLU(),
                nn.Linear(fc2_dims, n_actions),
                nn.Softmax(dim=-1)
        )

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        dist = self.actor(state)
        dist = Categorical(dist)
        
        return dist

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        device = T.device('cuda' if T.cuda.is_available() else 'cpu')
        self.load_state_dict(T.load(self.checkpoint_file, map_location=device))

class Critic(nn.Module):
    """ Value-based model that performs action at a state given by Actor to get the corresponding reward
    
        Attr:
            input_dims (int): Observation space
            alpha (float): Learning rate
            fc1_dims (int): NN layer dimension
            fc2_dims (int): NN layer dimension
            chkpt_dir (string): Model's weights storage path
    """
    def __init__(self, input_dims, alpha, fc1_dims=256, fc2_dims=256,
            chkpt_dir='model'):
        super(Critic, self).__init__()

        self.checkpoint_file = os.path.join(chkpt_dir, 'critic')
        os.makedirs(chkpt_dir, exist_ok=True)
        
        self.critic = nn.Sequential(
                nn.Linear(*input_dims, fc1_dims),
                nn.ReLU(),
                nn.Linear(fc1_dims, fc2_dims),
                nn.ReLU(),
                nn.Linear(fc2_dims, 1)
        )

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        value = self.critic(state)

        return value

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        device = T.device('cuda' if T.cuda.is_available() else 'cpu')
        self.load_state_dict(T.load(self.checkpoint_file, map_location=device))

class Agent:
    """ RL model-free off-policy model that combines policy-based Actor and value-based Critic
    
        Attr:
            n_actions (int): Action space
            input_dims (int): Observation space
            gamma (float): Discount factor
            alpha (float): Learning rate
            gae_lambda (float): Advantage's variance and bias control
            policy_clip (float): Policy update's deviation
            batch_size (int): Batch size
            n_epochs (int): Number of epochs for each episode
            max_episode_time (int): Maximum time an epoch is allowed to train
    """
    
    def __init__(self, n_actions, input_dims, gamma=0.99, alpha=0.0003, gae_lambda=0.95,
            policy_clip=0.2, batch_size=64, n_epochs=10, max_episode_time=300):
        self.gamma = gamma
        self.policy_clip = policy_clip
        self.n_epochs = n_epochs
        self.gae_lambda = gae_lambda
        self.max_episode_time = max_episode_time

        self.actor = Actor(n_actions, input_dims, alpha)
        self.critic = Critic(input_dims, alpha)
        self.memory = PPOMemory(batch_size)
       
    def remember(self, state, action, probs, vals, reward, done):
        self.memory.store_memory(state, action, probs, vals, reward, done)

    def save_models(self):
        print('... saving models ...')
        self.actor.save_checkpoint()
        self.critic.save_checkpoint()

    def load_models(self):
        print('... loading models ...')
        self.actor.load_checkpoint()
        self.critic.load_checkpoint()

    def choose_action(self, observation):
        """ Choose action based on observation

        Args:
            observation (tensor): Current observation

        Returns:
            action (tensor): Action at the current state
            probs (tensor): Log probability of the action
            value (tensor): Value after performing the action
        """
        
        state = T.tensor([observation], dtype=T.float).to(self.actor.device)

        dist = self.actor(state)
        value = self.critic(state)
        action = dist.sample()

        probs = T.squeeze(dist.log_prob(action)).item()
        action = T.squeeze(action).item()
        value = T.squeeze(value).item()

        return action, probs, value

    def learn(self):
        """
            Model's learning progress
        """
        
        for _ in range(self.n_epochs):
            start_time = time.time()
            state_arr, action_arr, old_prob_arr, vals_arr, reward_arr, dones_arr, batches = self.memory.generate_batches()

            values = vals_arr
            advantage = np.zeros(len(reward_arr), dtype=np.float32)

            """ Compute advantage function
                A_t = r(s_t) + gamma * gae_lambda * V(s_{t+1}) - V(s_t)
            """
            for t in range(len(reward_arr)-1):
                discount = 1
                a_t = 0
                for k in range(t, len(reward_arr)-1):
                    a_t += discount*(reward_arr[k] + self.gamma*values[k+1]*(1-int(dones_arr[k])) - values[k])
                    discount *= self.gamma*self.gae_lambda
                advantage[t] = a_t
            advantage = T.tensor(advantage).to(self.actor.device)

            values = T.tensor(values).to(self.actor.device)
            for batch in batches:
                states = T.tensor(state_arr[batch], dtype=T.float).to(self.actor.device)
                old_probs = T.tensor(old_prob_arr[batch]).to(self.actor.device)
                actions = T.tensor(action_arr[batch]).to(self.actor.device)

                dist = self.actor(states)
                critic_value = self.critic(states)
                critic_value = T.squeeze(critic_value)

                """ Compute clipped surrogate objective
                    r_t(theta) = pi_theta(a_t|s_t) / pi_old(a_t|s_t)
                    L^CPI(theta) = r_t(theta) * A_t
                    L^CLIP(theta) = min[E[L^CPI(theta), clip(r_t(theta), (1-clip, 1+clip)) * A_t]]
                """
                new_probs = dist.log_prob(actions)
                prob_ratio = new_probs.exp() / old_probs.exp()

                weighted_probs = advantage[batch] * prob_ratio
                weighted_clipped_probs = T.clamp(prob_ratio, 1-self.policy_clip,
                        1+self.policy_clip)*advantage[batch]
                actor_loss = -T.min(weighted_probs, weighted_clipped_probs).mean()
                
                """ Compute objective loss
                    L^VF_t = (A_t + V_theta(s_t) - V^targ_t)^2
                """

                returns = advantage[batch] + values[batch]
                critic_loss = (returns-critic_value)**2
                critic_loss = critic_loss.mean()

                total_loss = actor_loss + 0.5*critic_loss
                self.actor.optimizer.zero_grad()
                self.critic.optimizer.zero_grad()
                total_loss.backward()
                self.actor.optimizer.step()
                self.critic.optimizer.step()
            
            elapsed_time = time.time() - start_time

            if elapsed_time > self.max_episode_time:
                print(f"Took {elapsed_time:.2f} seconds, exceeding the time limit of {self.max_episode_time} seconds.")
                print("Stopping training.")
                break

        self.memory.clear_memory()               

