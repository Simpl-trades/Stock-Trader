import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense


from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetAssetsRequest
from alpaca.data.requests import StockLatestTradeRequest, StockLatestBarRequest
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.data import StockHistoricalDataClient, StockTradesRequest

from datetime import datetime
from collections import deque
import random
import itertools
import argparse
import re
import os
from dotenv import load_dotenv
from dotenv import dotenv_values
import pickle


class TradingEnvironment:
    def __init__(self, trading_client, data_client,symbol, initial_balance=10000):
        """
        Trading Client
        Data Client
        Symbol
        Current Step
        Sharesh Held
        Balance
        Net Worth
        """
        self.trading_client = trading_client
        self.data_client = data_client
        self.symbol = symbol
        self.initial_balance = initial_balance

        self.current_step = 0
        self.shares_held = 0
        self.balance = self.initial_balance
        self.net_worth = self.initial_balance

    def reset(self):
        """ 
        Reset the environment to the initial state. 
        Note: Needs revision when going live
        """
        self.current_step = 0
        self.shares_held = 0
        self.balance = self.initial_balance
        self.net_worth = self.initial_balance
        return self._get_state()

    def _get_latest_price(self):
        """
        Fetches the latest trade or bar for the given symbol from Alpaca.
        """
        request = StockLatestTradeRequest(symbol_or_symbols=[self.symbol])
        latest_trade_dict = self.data_client.get_stock_latest_trade(request)
        latest_trade = latest_trade_dict[self.symbol]
        current_price = float(latest_trade.price)
        return current_price



    def _place_market_order(self, side, qty):
        """
        Places a market order (paper) via Alpaca.
        `qty` can be fractional
        """

        order_data = MarketOrderRequest(
            symbol=self.symbol,
            qty=qty,
            side=side,
            time_in_force=TimeInForce.DAY
        )
        submitted_order = self.trading_client.submit_order(order_data=order_data)
        return submitted_order


    def _get_state(self):
        """
        Return the state as a NumPy array (can be expanded with indicators).
        Example features:
        """
        if isinstance(self.balance, np.ndarray): ## Band-Aid
            self.balance = float(self.balance.item())
        
        return np.array([
            self.balance, 
            self.shares_held
        ], dtype=np.float32)

    def step(self, action):
        """
        Take an action: Buy = 0, Sell 25% = 1, sell 50% = 2, sell 75% = 3, sell 100% = 4, Hold = 5
        Returns:
          next_state, reward, done
        """
        prev_net_worth = self.net_worth
        self.current_step += 1
        current_price = self._get_latest_price()
        shares_to_sell = 0


        # Execute action
        if action == 0:  # Buy all we can with local 'balance'
            max_shares_can_buy = float(self.balance) // float(current_price)
            if max_shares_can_buy > 0:
                self._place_market_order(OrderSide.BUY, max_shares_can_buy)
                # Update local tracking
                cost = max_shares_can_buy * current_price
                self.shares_held += max_shares_can_buy
                self.balance -= float(cost)

        elif action == 1:  # Sell 25%
            if self.shares_held > 0:
                shares_to_sell = float(self.shares_held * 0.25)
                if shares_to_sell > 0:
                    self._place_market_order(OrderSide.SELL, shares_to_sell)
                    revenue = shares_to_sell * current_price
                    self.shares_held -= shares_to_sell
                    self.balance += revenue

        elif action == 2:  # Sell 50%
            if self.shares_held > 0:
                shares_to_sell = float(self.shares_held * 0.5)
                if shares_to_sell > 0:
                    self._place_market_order(OrderSide.SELL, shares_to_sell)
                    revenue = shares_to_sell * current_price
                    self.shares_held -= shares_to_sell
                    self.balance += revenue

        elif action == 3:  # Sell 75%
            if self.shares_held > 0:
                shares_to_sell = float(self.shares_held * 0.75)
                if shares_to_sell > 0:
                    self._place_market_order(OrderSide.SELL, shares_to_sell)
                    revenue = shares_to_sell * current_price
                    self.shares_held -= shares_to_sell
                    self.balance += revenue

        elif action == 4:  # Sell 100%
            if self.shares_held > 0:
                shares_to_sell = self.shares_held
                self._place_market_order(OrderSide.SELL, shares_to_sell)
                revenue = shares_to_sell * current_price
                self.shares_held = 0
                self.balance += revenue    
        

        # If action == 5 (Hold), do nothing

        # Update net worth
        self.net_worth = float(self.balance) + float(self.shares_held * current_price)
        reward = float(self.net_worth) - float(prev_net_worth)  # change in net worth
        
        done = (self.current_step >= 100)

        next_state = self._get_state()
        return next_state, reward, done
    
class ReplayBuffer:
    def __init__(self, max_size=120):
        self.buffer = deque(maxlen=max_size)
    
    def sample_batch(self, batch_size=45):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards),
            np.array(next_states),
            np.array(dones)
        )
    
    def store(self, experience):
        self.buffer.append(experience)

    def __len__(self):
        return len(self.buffer)
    

class DQNAgent:
    def __init__(self, state_size, action_size, gamma=1, lr=0.001,
                 epsilon=0.75, epsilon_min=0.01, epsilon_decay=0.935):
        """
        state_size: Dimension of the state (e.g., 7 in our environment).
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

    def train(self, batch_size=45):
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
            
    def save_model(self, filename='dqn_model.h5'):
        self.model.save(filename)

def train_dqn(env, agent, n_episodes=100, batch_size=45):
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
            total_reward = float(total_reward.item())
        
        print(f"Episode {episode+1}/{n_episodes}, Total Reward: {total_reward:.2f}, Epsilon: {agent.epsilon:.3f}")
        episode_rewards.append(round(total_reward, 2))
        
    agent.save_model('dqn_model.keras')


load_dotenv()
ALPACA_API_KEY = os.getenv('ALPACA_API_KEY')
ALPACA_API_SECRET = os.getenv('ALPACA_API_SECRET')

trading_client = TradingClient(ALPACA_API_KEY, ALPACA_API_SECRET, paper=True)
data_client = StockHistoricalDataClient(ALPACA_API_KEY, ALPACA_API_SECRET)
account = trading_client.get_account()

# Stores rewards of the model per episode
episode_rewards = []

# Create environment
env = TradingEnvironment(
    trading_client=trading_client, 
    data_client=data_client,
    symbol="NVDA", 
    initial_balance=account.buying_power
)

# DQN agent
state_size = 2
action_size = 6  # (Buy = 0, Sell 25% = 1, sell 50% = 2, sell 75% = 3, sell 100% = 4, Hold = 5)
agent = DQNAgent(state_size, action_size)

# Train
train_dqn(env, agent, n_episodes=50, batch_size=45)

# Plot rewards per episode
plt.figure(figsize=(8, 4))
plt.plot(episode_rewards, label='Episode Reward')
plt.title('DQN Training Rewards')
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.legend()
plt.savefig("my_plot.png")