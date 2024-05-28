import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import matplotlib.pyplot as plt
from robotic_warehouse.warehouse import RewardType, Warehouse  # Ensure the package is correctly imported
from tqdm import tqdm

# Check if a GPU is available and use it
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define the Actor-Critic network with 5 hidden layers
class ActorCritic(nn.Module):
    def __init__(self, state_size, action_size):
        super(ActorCritic, self).__init__()
        self.fc1 = nn.Linear(state_size, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        self.fc4 = nn.Linear(256, 256)
        self.fc5 = nn.Linear(256, 256)
        self.actor = nn.Linear(256, action_size)
        self.critic = nn.Linear(256, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        x = torch.relu(self.fc5(x))
        action_probs = torch.softmax(self.actor(x), dim=-1)
        state_values = self.critic(x)
        return action_probs, state_values

# PPO algorithm
class PPOAgent:
    def __init__(self, state_size, action_size, lr=0.0003, gamma=0.99, eps_clip=0.2):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.policy = ActorCritic(state_size, action_size).to(device).float()
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.policy_old = ActorCritic(state_size, action_size).to(device).float()
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.MseLoss = nn.MSELoss()

    def select_action(self, state):
        state = torch.FloatTensor(state).to(device).unsqueeze(0)
        with torch.no_grad():
            action_probs, _ = self.policy_old(state)
        dist = Categorical(action_probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action)

    def update(self, memory):
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(memory.rewards[::-1], memory.is_terminals[::-1]):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        rewards = torch.tensor(rewards, dtype=torch.float).to(device)
        old_states = torch.tensor(np.array(memory.states), dtype=torch.float).to(device)
        old_actions = torch.tensor(memory.actions).to(device)
        old_logprobs = torch.tensor(memory.logprobs).to(device)

        for _ in range(5):  # PPO update loop
            action_probs, state_values = self.policy(old_states)
            dist = Categorical(action_probs)
            entropy = dist.entropy().mean()
            new_logprobs = dist.log_prob(old_actions)
            state_values = state_values.squeeze()
            advantages = rewards - state_values.detach()

            ratio = torch.exp(new_logprobs - old_logprobs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            loss = -torch.min(surr1, surr2).mean() + 0.5 * self.MseLoss(state_values, rewards) - 0.01 * entropy

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        self.policy_old.load_state_dict(self.policy.state_dict())

class Memory:
    def __init__(self):
        self.states = []
        self.actions = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []

    def clear_memory(self):
        del self.states[:]
        del self.actions[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]

def load_env():
    env = Warehouse(
        shelf_columns=3,
        column_height=8,
        shelf_rows=1,
        n_agents=2,
        msg_bits=0,
        sensor_range=1,
        request_queue_size=2,
        max_inactivity_steps=None,
        max_steps=500,
        reward_type=RewardType.INDIVIDUAL
    )
    num_agents = env.n_agents
    state_size = env.observation_space[0].shape[0]
    action_size = env.action_space[0].n
    return env, num_agents, state_size, action_size

def plot_stats(reward_history, steps_history, filename_prefix, window_size=50):
    for i in reward_history:
        episodes = list(range(window_size, len(reward_history[i]) + 1, window_size))
        smoothed_rewards = [
            np.mean(reward_history[i][j-window_size:j])
            for j in episodes
        ]
        plt.figure(figsize=(10, 5))
        plt.scatter(episodes, smoothed_rewards, label=f"Agent {i} (Smoothed)", color='darkred')
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        plt.title(f'Reward vs Episode for Agent {i}')
        plt.legend()
        plt.savefig(f'{filename_prefix}_rewards_vs_episode_agent_{i}.png')
        plt.show()

    for i in steps_history:
        episodes = list(range(window_size, len(steps_history[i]) + 1, window_size))
        smoothed_steps = [
            np.mean(steps_history[i][j-window_size:j])
            for j in episodes
        ]
        plt.figure(figsize=(10, 5))
        plt.scatter(episodes, smoothed_steps, label=f"Agent {i} (Steps)", color='darkgreen')
        plt.xlabel('Episode')
        plt.ylabel('Steps per Episode')
        plt.title(f'Steps per Episode vs Episode for Agent {i}')
        plt.legend()
        plt.savefig(f'{filename_prefix}_steps_vs_episode_agent_{i}.png')
        plt.show()

def main():
    env, num_agents, state_size, action_size = load_env()
    agents = [PPOAgent(state_size, action_size) for _ in range(num_agents)]
    memory = Memory()
    reward_history = {i: [] for i in range(num_agents)}
    steps_history = {i: [] for i in range(num_agents)}
    episode_lengths = []

    for episode in tqdm(range(15000), desc="Training Progress"):
        state = env.reset()
        total_rewards = np.zeros(num_agents)
        episode_length = 0

        for t in range(500):
            actions = []
            logprobs = []
            for i in range(num_agents):
                action, logprob = agents[i].select_action(state[i])
                actions.append(action)
                logprobs.append(logprob)
            new_state, reward, done, _ = env.step(actions)

            for i in range(num_agents):
                memory.states.append(state[i])
                memory.actions.append(actions[i])
                memory.logprobs.append(logprobs[i])
                memory.rewards.append(reward[i])
                memory.is_terminals.append(done[i])

            state = new_state
            total_rewards += reward
            episode_length += 1

            if any(done):
                break

        for i in range(num_agents):
            reward_history[i].append(total_rewards[i])
            steps_history[i].append(episode_length)

        for agent in agents:
            agent.update(memory)
        memory.clear_memory()

        if episode % 10 == 0:
            tqdm.write(f"Episode {episode} completed")

    for i in range(num_agents):
        torch.save(agents[i].policy.state_dict(), f'ppo_agent_12000_{i}.pth')

    plot_stats(reward_history, steps_history, 'training')

if __name__ == "__main__":
    main()
