---
layout: post
title: Simulating a Portfolio
---

We need to keep track of the performances of our trader. And more importantly, understand how to reward our trader for making good trade decisions. 

Any good reinforcement learning system depends on how good the reward function is. 

![reward fn](https://leonardoaraujosantos.gitbooks.io/artificial-inteligence/content/image_folder_5/conv_agent.png)

In the Mario example, the rewards are very clear: do eat coins, don't jump off cliffs; avoid monsters, make it to the finish line. 

However, in cryptocurrency trading, the reward function requires a lot more thought. Most previous ML-based methods use instantaneous reward, such as daily profit. Daily profit is usually calculated in additive ![additive](http://rogercortesi.com/eqn/tempimagedir/eqn5109.png) or productive form ![productive](http://rogercortesi.com/eqn/tempimagedir/eqn5156.png). 

The instantaneous reward is often too noisy for learning, and it is not consistent with our goal of maximizing long term profit. We ended up basing our reward based on accumulated wealth over n days in the past. 

![final reward](http://rogercortesi.com/eqn/tempimagedir/eqn6667.png)

Our problem formulation is heavily influenced by [Deep-Q-Trading](http://cslt.riit.tsinghua.edu.cn/mediawiki/images/5/5f/Dtq.pdf) by an algorithmic group at TsingHua University. 

Our portfolio class also keeps track of the number of coins and cash in our portfolio, as well as how well we are doing compared with the past. 

For more, see our implementation [here](https://github.com/GradientTrader/simulator/blob/master/v2/portfolio.py). 
