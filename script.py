import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
import yfinance as yf

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense


from datetime import datetime
from collections import deque
import random
import itertools
import argparse
import re
import os
import pickle


"""
Potentially use: interval="1m", period="7d"
for day trading data
"""


class TradingEnvironment:
    def __init__(self, prices, initial_balance=10000, max_shares=1000):
        """
        prices: A list or array of stock prices (e.g., daily closing prices).
        initial_balance: How much cash we start with.
        max_shares: Maximum shares you can hold (to avoid unrealistic large positions).
        """
        self.prices = prices
        self.n_steps = len(prices)
        self.initial_balance = initial_balance
        self.max_shares = max_shares

        self.reset()

    def reset(self):
        """ Reset the environment to the initial state. """
        self.current_step = 0
        self.balance = self.initial_balance
        self.shares_held = 0
        self.net_worth = self.initial_balance

        return self._get_state()

    def _get_state(self):
        """
        Return the state as a NumPy array (can be expanded with indicators).
        Example features:
          - current price
          - balance
          - shares held
          - Volume
          - Change in value ($ or %)
          - 
        """
        current_price = self.prices[self.current_step].item()
        
        if isinstance(self.balance, np.ndarray): ## Band-Aid
            self.balance = int(self.balance.item())
        
        return np.array([
            f"{current_price:.2f}",
            self.balance, 
            self.shares_held
        ], dtype=np.float32)

    def step(self, action):
        """
        Take an action: 0 = Buy, 1 = Sell, 2 = Hold
        Returns:
          next_state, reward, done
        """
        current_price = self.prices[self.current_step]
        prev_net_worth = self.net_worth

        # Execute action
        if action == 0:  # Buy
            # Number of shares we can buy
            max_possible = int(self.balance // current_price)
            shares_to_buy = min(max_possible, self.max_shares - self.shares_held)
            cost = shares_to_buy * current_price
            self.balance -= cost
            self.shares_held += shares_to_buy

        elif action == 1:  # Sell
            # Sell all shares held (or you can choose partial selling)
            shares_to_sell = self.shares_held
            self.balance += shares_to_sell * current_price
            self.shares_held = 0

        # If action == 2 (Hold), do nothing

        # Update net worth
        self.net_worth = self.balance + self.shares_held * current_price
        reward = self.net_worth - prev_net_worth  # change in net worth

        self.current_step += 1
        done = (self.current_step >= self.n_steps - 1)

        next_state = self._get_state()
        return next_state, reward, done
    
class ReplayBuffer:
    def __init__(self, max_size=100):
        self.buffer = deque(maxlen=max_size)

    def store(self, experience):
        self.buffer.append(experience)

    def sample_batch(self, batch_size=32):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards),
            np.array(next_states),
            np.array(dones)
        )
        # 100 200 40
        # 102 200 40 (sell) +1
        # 104 4200 0

    def __len__(self):
        return len(self.buffer)
    
    
class DQNAgent:
    def __init__(self, state_size, action_size, gamma=1, lr=0.001,
                 epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995):
        """
        state_size: Dimension of the state (e.g., 3 in our environment).
        action_size: Number of actions (Buy, Sell, Hold = 3).
        gamma: Discount factor for future rewards.
        lr: Learning rate for optimizer.
        epsilon: Exploration rate (start).
        epsilon_min: Minimum exploration rate.
        epsilon_decay: Decay factor for epsilon per episode.
        """
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.lr = lr
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        self.replay_buffer = ReplayBuffer()
        self.model = self._build_model()
        
    def _build_model(self):
        model = Sequential()
        model.add(Dense(64, input_shape=(self.state_size,), activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=self.lr))
        return model

    def act(self, state):
        """
        Epsilon-greedy policy:
          - With probability epsilon, pick a random action
          - Otherwise, pick the action with the highest Q-value
        """
        if np.random.rand() <= self.epsilon:
            return np.random.randint(0, self.action_size)
        q_values = self.model.predict(np.array([state]), verbose=0)
        return np.argmax(q_values[0])

    def train(self, batch_size=32):
        """
        Sample a batch from replay memory and perform one step of gradient descent on the Q-network.
        """
        if len(self.replay_buffer) < batch_size:
            return  # Not enough data to train

        states, actions, rewards, next_states, dones = self.replay_buffer.sample_batch(batch_size)

        # Predict Q-values for current states
        q_values = self.model.predict(states, verbose=0)

        # Predict Q-values for next states
        q_next = self.model.predict(next_states, verbose=0)

        for i in range(batch_size):
            target = q_values[i]
            if dones[i]:
                # If done, the target is just the reward
                target[actions[i]] = rewards[i]
            else:
                # Update rule:Q(s, a) =  r + gamma * max(Q_next)
                target[actions[i]] = rewards[i] + self.gamma * np.max(q_next[i])

        self.model.fit(states, q_values, epochs=1, verbose=0)

    def update_epsilon(self):
        """
        Decay epsilon after each episode (reduce exploration over time).
        """
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

def train_dqn(env, agent, n_episodes=100, batch_size=32):
    for episode in range(n_episodes):
        state = env.reset()
        total_reward = 0

        while True:
            action = agent.act(state)
            next_state, reward, done = env.step(action)
            
            # Store in replay buffer
            agent.replay_buffer.store((state, action, reward, next_state, done))

            # Train the agent
            agent.train(batch_size)

            state = next_state
            total_reward += reward

            if done:
                break

        # Decay exploration rate
        agent.update_epsilon()
        
        if isinstance(total_reward, np.ndarray):    ## Band-Aid
            total_reward = int(total_reward.item())
        
        print(f"Episode {episode+1}/{n_episodes}, Total Reward: {total_reward:.2f}, Epsilon: {agent.epsilon:.3f}")



intel_data = yf.download(
    "INTC",
)

intel_data

intel_data = intel_data[["Open", "Close", "High", "Low", "Volume"]]

intel_data.columns = intel_data.columns.droplevel('Ticker') # Drops Ticker row

data = yf.download('AAPL', start='2020-01-01', end='2022-01-01')
prices = data['Close'].values

# Create environment
env = TradingEnvironment(prices, initial_balance=10000)

# DQN agent
state_size = 3  # (price, balance, shares_held)
action_size = 3  # (Buy, Sell, Hold)
agent = DQNAgent(state_size, action_size)

# Train
train_dqn(env, agent, n_episodes=50, batch_size=32)