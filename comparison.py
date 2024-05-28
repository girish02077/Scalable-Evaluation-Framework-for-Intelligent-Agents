import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import matplotlib.pyplot as plt
from robotic_warehouse.warehouse import RewardType, Warehouse
from tqdm import tqdm
import itertools
import os

# Check if a GPU is available and use it
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define the Actor-Critic network
class ActorCritic(nn.Module):
    def __init__(self, state_size, action_size):
        super(ActorCritic, self).__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.actor = nn.Linear(128, action_size)
        self.critic = nn.Linear(128, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        action_probs = torch.softmax(self.actor(x), dim=-1)
        state_values = self.critic(x)
        return action_probs, state_values

# PPO algorithm
class PPOAgent:
    def __init__(self, state_size, action_size, lr=0.00078, gamma=0.9621, eps_clip=0.270):
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

def main():
    learning_rates = [0.0008, 0.0009, 0.00095]
    discount_factors = [0.85, 0.9, 0.95]
    epsilons = [0.05, 0.1, 0.15]

    results_dir = "results_hyperparameters"
    os.makedirs(results_dir, exist_ok=True)

    for eps in epsilons:
        for lr, gamma in itertools.product(learning_rates, discount_factors):
            env, num_agents, state_size, action_size = load_env()
            agents = [PPOAgent(state_size, action_size, lr=lr, gamma=gamma, eps_clip=eps) for _ in range(num_agents)]
            memory = Memory()
            reward_history = {i: [] for i in range(num_agents)}
            episode_lengths = []

            for episode in tqdm(range(3000), desc=f"Training Progress (lr={lr}, gamma={gamma}, eps={eps})"):
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

                episode_lengths.append(episode_length)

                for agent in agents:
                    agent.update(memory)
                memory.clear_memory()

            avg_reward = np.mean([reward_history[i] for i in range(num_agents)], axis=0)

            plt.figure()
            plt.plot(avg_reward)
            plt.xlabel('Episode')
            plt.ylabel('Average Reward')
            plt.title(f'Average Reward vs Episode (lr={lr}, gamma={gamma}, eps={eps})')
            plt.savefig(os.path.join(results_dir, f'average_reward_vs_episode_lr_{lr}_gamma_{gamma}_eps_{eps}.png'))
            plt.close()

            for i in range(num_agents):
                torch.save(agents[i].policy.state_dict(), f'{results_dir}/ppo_agent_lr_{lr}_gamma_{gamma}_eps_{eps}_agent_{i}.pth')

if __name__ == "__main__":
    main()
