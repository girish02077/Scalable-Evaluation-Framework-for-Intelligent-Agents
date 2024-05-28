import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from robotic_warehouse.warehouse import RewardType, Warehouse
import optuna

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
    def __init__(self, state_size, action_size, lr, gamma, eps_clip):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.policy = ActorCritic(state_size, action_size).float()
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.policy_old = ActorCritic(state_size, action_size).float()
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.MseLoss = nn.MSELoss()

    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
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

        rewards = torch.tensor(rewards, dtype=torch.float)
        old_states = torch.FloatTensor(memory.states)
        old_actions = torch.tensor(memory.actions, dtype=torch.long)
        old_logprobs = torch.tensor(memory.logprobs, dtype=torch.float)

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

def objective(trial):
    env, num_agents, state_size, action_size = load_env()
    lr = trial.suggest_loguniform('lr', 1e-5, 1e-2)
    gamma = trial.suggest_uniform('gamma', 0.9, 0.999)
    eps_clip = trial.suggest_uniform('eps_clip', 0.1, 0.3)
    
    agents = [PPOAgent(state_size, action_size, lr, gamma, eps_clip) for _ in range(num_agents)]
    memory = Memory()
    reward_history = []

    for episode in range(100):
        state = env.reset()
        total_rewards = np.zeros(num_agents)
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

            if any(done):
                break

        for agent in agents:
            agent.update(memory)
        memory.clear_memory()

        reward_history.append(total_rewards.sum())

    return np.mean(reward_history)

if __name__ == "__main__":
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=50)

    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
