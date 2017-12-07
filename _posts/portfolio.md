

```python

'''

### Cryptocurrency Trader Agent
### UCB MIDS 2017 Winter Capstone Project
### Ramsey Aweti, Shuang Chan, GuangZhi(Frank) Xie, Jason Xie

### Class: 
###        Portfolio
### Purpose: 
###        This is utility class used to maintain action, reward and internal state definitions.
###        
### Sample Usage:

from env import *
env = Environment(coin_name="ethereum")
p = Portfolio(env)

print env.getStates() ## initial step
p._buy(10)
print env.step()
print p.getCurrentHoldings()

'''

import numpy as np
from enum import Enum

# action list
class Action(Enum):
    HOLD=0
    BUY=1
    SELL=2

# internal state
state_list = ["coin", "cash", "total_value", "is_holding_coin", "return_since_entry"]

class Portfolio:
    
    # initialize the portfolio variables
    def __init__(self, portfolio_cash=1000.0, num_coins_per_order=10.0, states=state_list):
        self.starting_cash = portfolio_cash
        self.portfolio_coin = 0.0
        self.portfolio_cash = portfolio_cash
        self.num_coins_per_order = num_coins_per_order
        self.states = states
        
        ### Mapping states to their names
        self.state_dict = {}
        self.state_dict["coin"] = self.portfolio_coin
        self.state_dict["cash"] = self.portfolio_cash
        self.state_dict["total_value"] = self.portfolio_cash
        self.state_dict["is_holding_coin"] = 0
        self.state_dict["return_since_entry"] = 0
        
        self.daily_return_percentage = 0

    def __buy(self, current_price):
        if not current_price:
            return 0
        
        if self.num_coins_per_order == 0:
            amount_to_buy = self.portfolio_cash / current_price
        else:
            amount_to_buy = min(self.portfolio_cash / current_price, self.num_coins_per_order)
            
        self.portfolio_coin += amount_to_buy
        self.portfolio_cash -= amount_to_buy * current_price
        return amount_to_buy
    
    def __sell(self, current_price):
        if not current_price:
            return 0
        
        if self.num_coins_per_order == 0:
            coin_to_sell = self.portfolio_coin
        else:
            coin_to_sell = min(self.num_coins_per_order, self.portfolio_coin)
        
        self.portfolio_coin -= coin_to_sell
        self.portfolio_cash += coin_to_sell * current_price
        return coin_to_sell
    
    # reset portfolio
    def reset(self):
        self.__init__(portfolio_cash=self.starting_cash, num_coins_per_order=self.num_coins_per_order, states=self.states)
        
    # return internal state    
    def getStates(self, states=None):
        if not states:
            states = self.states
        return [self.state_dict[state] for state in states]
    
    # reward defintion
    # Is Daily Return a good reward function?
    def getReward(self):
        return self.daily_return_percentage

    # apply action (buy, sell or hold) to the portfolio
    # update the internal state after the action
    def apply_action(self, current_price, action):
        if action == Action.BUY:
            self.__buy(current_price)
        elif action == Action.SELL:
            self.__sell(current_price)
        
        # Update daily return
        self.daily_return_percentage = (self.getCurrentValue(current_price) * 100.0 / self.state_dict["total_value"]) - 100
        
        # Update states
        self.state_dict["coin"] = self.portfolio_coin
        self.state_dict["cash"] = self.portfolio_cash
        self.state_dict["total_value"] = self.getCurrentValue(current_price)
        self.state_dict["is_holding_coin"] = (self.portfolio_coin > 0)*1
        self.state_dict["return_since_entry"] = self.getReturnsPercent(current_price)
        

    def getCurrentValue(self, current_price):
        return self.portfolio_coin * current_price + self.portfolio_cash

    def getReturnsPercent(self, current_price):
        return 100 * (self.getCurrentValue(current_price) - self.starting_cash) / self.starting_cash

    def getCurrentHoldings(self, current_price):
        return "%.2f coins, %.2f cash, %.2f current value, %.2f percent returns" \
                % (self.portfolio_coin, self.portfolio_cash, self.getCurrentValue(current_price), self.getReturnsPercent())
        
    def getActionSpaceSize(self):
        return len(list(Action))
    
    def getStateSpaceSize(self):
        return len(self.states)




```
