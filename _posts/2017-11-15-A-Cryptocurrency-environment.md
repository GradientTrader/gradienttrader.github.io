---
layout: post
title: A CryptoCurrency Environment
---

How do we create a programmatic environment to explore the Cryptocurrenct environment?

We obtained versions of crytocurrency price histories [online](https://github.com/GradientTrader/simulator/tree/master/v2/cryptocurrencypricehistory). Our data consists of the opening and closing prices of 20+ cryptocurrencies. 

![bitcoin prices](https://www.buybitcoinworldwide.com/img/kb/bitcoinprice.png)

We knew that we wanted something that adheres to the [OpenAI gym](https://github.com/openai/gym) api, which is a MDP environment useful for reinforcement learning.

![mdp env](https://cdn-images-1.medium.com/max/1600/1*hfuWJ7CeLlA57KMXDQaJrw.jpeg)

Normally in an OpenAI Gym, the state of the system is the graphics of the game. However, in our case, it's the prices of the cryptocurrencies. 

Our state includes various technical trading features, such as ```current_price, rolling_mean, rolling_std, cross_upper_band, cross_lower_band, upper_band, lower_band, price_over_sma```.

Our environment allows users to step through each day of the cryptocurrency.

```
### Sample Usage:

env = Environment(coin_name="ethereum")
print env.step()
print env.step()
print env.step()
env.plot()

```


For more, see our implementation [here](https://github.com/GradientTrader/simulator/blob/master/v2/env.py). 
