---
layout: post
title: Deep Q Learning Agent
---

So how exactly does an agent trade? Here we explain our DQN agent.

Deep Q Learning Algorithm (E-Greedy)

For each step of the episode from the simulation

a. Retrieve the feature vector that defines the state, i.e.: Cross Bollinger Bands, SMA and etc.  
b. Use the deep learning neural network to estimate the Q values for each action.  The output is the Target Q vector  
c. Generate a number randomly between 0 and 1.  If the number is greater or equal to the Epsilon, the exploration factor, choose an action randomly. If the number is smaller than the Epsilon, choose the action with the largest Q value from the Target Q vector.  

        if np.random.rand() < self.epsilon:
            return random.choice(list(Action))
        act_values = self.model.predict(state)
        return Action(np.argmax(act_values[0]))

d. Apply the action from the previous step to the portfolio and retrieve the reward  
e. Move one step forward through the environment  
f. Retrieve the feature vector that defines the state  
g. Estimate the Q values from the feature vector from the previous step  

h. Apply a future value discount to the maximum Q value from the previous step and add it to the reward  
i. Update the Q value that corresponds to the action taken from the Target Q vector with the reward from previous step  

        # Bellman Equation
        # -0.60 + gamma * -0.50
        target[0][action.value] = reward + self.gamma * t[np.argmax(a)]
                
j. Train the deep neural network with the new Target Q vector and State incrementally  

Decrease the Epilon by a decay function  
Repeat Step #1 until convergence  


For more, see our implementation [here](https://github.com/GradientTrader/simulator/blob/master/v2/ddqn_agent.py). 

