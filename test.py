import gym
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from robotic_warehouse.warehouse import RewardType, Warehouse  # Ensure the package is correctly imported
from tqdm import tqdm

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

# PPO Agent
class PPOAgent:
    def __init__(self, state_size, action_size, model_path):
        self.state_size = state_size
        self.action_size = action_size
        self.policy = ActorCritic(state_size, action_size).float()
        self.policy.load_state_dict(torch.load(model_path))
        self.policy.eval()  # Set the policy to evaluation mode

    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            action_probs, _ = self.policy(state)
        action = torch.argmax(action_probs).item()
        return action

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

def plot_stats(reward_history, episode_lengths, filename_prefix):
    for i in reward_history:
        plt.figure(figsize=(10, 5))
        plt.plot(reward_history[i], label=f"Agent {i}")
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        plt.title(f'Reward vs Episode for Agent {i}')
        plt.legend()
        plt.savefig(f'{filename_prefix}_rewards_vs_episode_agent_{i}.png')
        plt.show()

    plt.figure(figsize=(10, 5))
    plt.plot(episode_lengths)
    plt.xlabel('Episode')
    plt.ylabel('Episode Length')
    plt.title('Episode Length vs Episode')
    plt.savefig(f'{filename_prefix}_episode_length_vs_episode.png')
    plt.show()


def test_agents(env, agents, num_episodes=10):
    reward_history = {i: [] for i in range(len(agents))}
    episode_lengths = []

    for episode in tqdm(range(num_episodes), desc="Testing Progress"):
        state = env.reset()
        total_rewards = np.zeros(len(agents))
        episode_length = 0

        for t in range(500):
            actions = []
            for i in range(len(agents)):
                action = agents[i].select_action(state[i])
                actions.append(action)
            state, reward, done, _ = env.step(actions)
            total_rewards += reward
            episode_length += 1
            env.render()
            if any(done):
                break

        for i in range(len(agents)):
            reward_history[i].append(total_rewards[i])

        episode_lengths.append(episode_length)

    plot_stats(reward_history, episode_lengths, 'testing')

def main():
    env, num_agents, state_size, action_size = load_env()
    agents = [PPOAgent(state_size, action_size, f'ppo_agent_15000_{i}.pth') for i in range(num_agents)]

    test_agents(env, agents, num_episodes=100)

if __name__ == "__main__":
    main()
