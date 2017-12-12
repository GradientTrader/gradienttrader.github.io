---
layout: post
title: Test and Evaluation
---

And finally, how are we evaluating our results?

We have run the model against 4 different cryptocurrencies, namely Bitcoin, Ethereum, Ripple, and Numeraire. Then we evaluate the total percentage of returns on investment, and comparing with the baseline model that is a simple agent trades based on Bollinger Bands.

The data used in the evaluation is shown below.

Bitcoin

![TE1](https://github.com/GradientTrader/gradienttrader.github.io/blob/master/images/btcPrice.png?raw=true)

Ethereum

![TE1](https://github.com/GradientTrader/gradienttrader.github.io/blob/master/images/ethPrice.png?raw=true)

Ripple

![TE1](https://github.com/GradientTrader/gradienttrader.github.io/blob/master/images/ripplePrice.png?raw=true)

Numeraire

![TE1](https://github.com/GradientTrader/gradienttrader.github.io/blob/master/images/numerairePrice.png?raw=true)

One of the key parts in the test & evaluation process is to explore the effective states and the reward function to use. We have developed 2 types of states: external and internal. External states are the states derived from the market, including current coin price, rolling mean, rolling standard deviation, whether price crosses upper Bollinger band, whether price crosses lower Bollinger band, the value of upper Bollinger band, the value of lower Bollinger band, and ratio of price over moving average. Internal states are those related to the portfolio, including number of coins currently holding, the value of cash currently holding, the total value of the portfolio, whether hodling any coins, and the return since entry. From the experiment, we observed that it is more effective to have lesser states input to the model, as more states can bring more noises. Among all the external states, current coin price is proven to be the most effective indicator, as all the other states are derived from it. In terms of internal states, whether a portfolio is holding any coin is proven to be the most effective one.

We have also explored several reward functions. 