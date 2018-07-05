# MULTI-ARM BANDIT PROBLEM
##### ***Refer to bandit.ipynb file for code***

Multi-arm bandit is a classic example of reinforcement learning where there are multiple options, in this case slot machines, and the player needs to identify the slot with the maximum payout so that the profit is maximized.

However, the caveat is that resource is limited and there may not be enough time or slot pulls available, but the player still needs to maximize the payout. If the player spends more time figuring out which slot has the highest return, the opportunity cost, or in this case exploit, increases. Conversely, the effect of quickly deciding on a slot without a thorough exploration might lead to choosing a sub-optimal slot machine. As such, there is always a tradeoff between exploit and explore.

For this experiment, we free ourselves from the aforementioned tradeoff and let the experiment run for say 100,000 times so that we can analyze all potential sub-optimal choices and at the end find out the optimal choice.

We are using epsilon greedy algorithm in this experiment because it is effective and easy to understand and implement. We start with three slot machines and assign them epsilon values (0.1, 0.05 and 0.01). How this works is we first randomly choose a probability value and compare it against our set epsilon value. If it is less than our epsilon, we randomly select a slot but if it is higher, we take the maximum mean value for the slot machine, and this is where the greedy epsilon algorithm comes into play.

We are running the experiment for 100,000 times so that we can clearly see the trend in payouts for all bandits and calculate their means. If we just use a linear scale for the plots, we cannot see the initial volatility in the trending so we use a log scale.

Just to experiment, I added another slot machine with epsilon value 0.001 and ran the experiment. We can now see the results in the graph for all four slot machines. In this case, looks like the last slot with epsilon value 0.001, shown in red, gives the maximum payout.

![1](https://user-images.githubusercontent.com/7417075/42305525-8d2319d4-7fe8-11e8-888a-8a0a8ed024d8.png)

The result we got from epsilon greedy for solving the explore/exploit dilemma is not bad, but we can do better. We do this by implementing the concept of upper bound or optimistic initial value. The concept behind this is that if we set an unnaturally high mean value to begin with so there is only going down. This is at odds with our initial method where we sent the mean as 0. If the true mean is say, 3 but we have the optimistic mean/ upper bound at 10, the estimate mean will eventually converge and get closer to 1 as we collect more data. 

![2](https://user-images.githubusercontent.com/7417075/42305526-8d369c2a-7fe8-11e8-864d-7c1e92c9e387.png)

We then want to compare the results from our optimistic method to when epsilon is 0.001 because 0.001 gave us the best result on our initial run. As shown by the graph below, in both linear and log scale, the optimistic method outperforms the one with epsilon.

![3](https://user-images.githubusercontent.com/7417075/42305524-8d0fa91c-7fe8-11e8-9b33-6660d86b5d52.png)


# TIC TAC TOE

##### ***Refer to tictactoe.ipynb file for code***

There are several ways Tic Tac Tor can be coded. The obvious way is to list down all conditions and states and generate a sequence of if statements what will help the model determine what the best move is while playing it against the opponent. Tic Tac Toc is not a complex game, it is a 3*3 board only with nine available options, but all possible permutations in this board is 9! which is large. If the board were to be 4*4 with 16 permutations, you can easily image how long the code would be if we were to do this through is statements.

As such, we do this via Reinforcement Learning. The goal here is to set the environment with all possible states, agents and rewards so that the game can be played over a period of episodes so that the game learns from itself in each episode. This way, we still reach the goal of generating a model that is able to play the game without hard coding all possible states. 

The way we teach our model how to make every subsequent move is by the concept of value function and reward. Moving to each step or state has an immediate reward. However, we are more concerned about the future prospect of that state rather than the immediate reward. A state can have a better reward but it may not have the best value that will eventually help us win. As such, we calculate the value function of each possible states and move to that state rather than a state with just a better reward. 
The value function can be summarized as the average of all future rewards for a state. The value function is notated as below. 

V(s) = V(s) + alpha*(V(s’) – V(s)) 

At high level, our goal is to have to agents play against each other for 1000 iterations and learn from each other. Player 1 will be X while Player 2 will be 0. After 1000 iterations, we will consider the model as trained and we will swap Player 2 with a human player. This is when we can play against the trained model and see it how well the model plays.

We can see our value function in action in the format we had specified earlier.

![1](https://user-images.githubusercontent.com/7417075/42296199-f9be9c64-7faf-11e8-8184-3d58d892ef8b.png)

We have also developed a verbose function that visualizes the value function in play. It prints the 3*3 grid and gives a value function for each available state. So in the following case, (2,1) has the highest value because for agent 0, it will help win the game and for agent X, it will help defend the game.

![2](https://user-images.githubusercontent.com/7417075/42296200-f9f590a2-7faf-11e8-8c62-45c4017dcae2.png)

When playing the game, you can see how the model is playing the game. Because of our verbose function, P1 makes a move according to the value function and lets us make the move. 

![3](https://user-images.githubusercontent.com/7417075/42296202-fa1892b4-7faf-11e8-82ea-62c443b347ca.png)

As the game progresses, P1 uses the value function to make its move and makes the action based on the highest value from the grid.

![4](https://user-images.githubusercontent.com/7417075/42296203-fa436f98-7faf-11e8-9121-5a60a374a779.png)


There are a few variables that we can tweak here. Epsilon dictates how much exploration to do. Alpha dictates the learning rate and T dictates the number of episodes. The model was so good that I could not beat Player 1 so I tweaked these variables so that the learning for P1 is limited hoping I can beat it. Here is what happened. Although, it was good that it was trying to make a sequence, it was not necessarily looking at my moves and deterring me from making a game. 

![5](https://user-images.githubusercontent.com/7417075/42296198-f9a8dcd0-7faf-11e8-8d5c-77626bfe4b1e.png)

With all machine learning models, there is always a tradeoff between making a model robust vs the time it takes to train the model. Although time is not of concern here in this simple game, a less robust model may be quicker to train but the results may not be great, as shown in the final example.


# ITERATIVE POLICY EVALUATION

Here we try to discuss the Markov Decision Process, otherwise known as MDP. Any reinforcement learning problem with these five components (set of states, actions, rewards, action probabilities and discount factor) is a MDP.

To work on a MDP, there are two steps. First one is to calculate a value function, that determines what actions to take based on the maximum value of available states and the other is to find a policy, otherwise known as a set of actions that lead to the terminal state. However, we do not need any random policy but an optimum policy with the highest reward. Value is denoted by V(s) at state s and policy is noted as π. Optimal value is noted as V*(s) and optimal policy as π*.

For iterative policy evaluation, we will use the Bellman’s equation repeatedly until it converges. When we say converges, we mean that the difference between the maximum value and the calculated value so that we break the loop when we are sure the current maximum value we have is the maximum.  Here is how we denote the equation.

![1](https://user-images.githubusercontent.com/7417075/42296998-cd88451a-7fb8-11e8-9021-42f990ef694b.png)

We will use the help of the grid world problem to delve into iterative policy evaluation. The grid is basically a 3*4 box with a defined starting place, with two possible terminal states, one with a positive value and the other with a negative value.  There is also a state in the gird that the player cannot enter, so we will exclude this as a possible state. Here is what the grid looks like.

![2](https://user-images.githubusercontent.com/7417075/42296996-cd6106da-7fb8-11e8-95f3-26caf1ba6705.png)

A player may eventually reach the terminal state with the highest reward but it may also never reach the terminal state because the player can in fact just revolve around the black wall. This can be done in a normal grid. We now introduce a negative grid where there is cost for each action. As such, the player sets to reach the best terminal state in the shortest time possible. There is also a concept of a random policy and a fixed policy. We will use them both in this example to look at the differences.

### Uniformly Random Actions
In the case of uniformly random actions, each action has an equal probability of being selected. This is noted as 1.0 / len(grid.actions[s]). We then get into the loop, calculate the value function and store the value as new_v to eventually compare if the difference in values is smaller than our set small threshold to get the best value.

![3](https://user-images.githubusercontent.com/7417075/42296997-cd752ee4-7fb8-11e8-9caf-2faf0ec44e33.png)

When we look at values for uniformly random actions, here is what the values look like. Most of them are negative because if they are mobbing randomly, they can also get to the loosing state and there are two ways you can get to the loosing state and only one way to the winning state.

![4](https://user-images.githubusercontent.com/7417075/42297002-d7186524-7fb8-11e8-935e-d38023265cf6.png)

### Fixed Policy

For a fixed policy with a discount factor, the closer we are to the terminal state, the higher our values are. This is because of the discount factor, which we set as 0.9 which is why the value function decreases from 1 to 0.9 to 0.81 and so on. 

![5](https://user-images.githubusercontent.com/7417075/42297003-d72af7e8-7fb8-11e8-87e3-dc3f617cb694.png)

I tried this with a discount factor of 1 and the discounting stopped. All the states within the policy had  a value of 1 because there is no discounting and eventually you will reach the terminal state because of the fixed policy.

![6](https://user-images.githubusercontent.com/7417075/42297001-d7040930-7fb8-11e8-952f-b52685a46e32.png)



# POLICY ITERATION AND VALUE ITERATION

In iterative policy evaluation, we looked at calculating value function but we did not look into optimizing the policy. Now we will look at two different ways improving our policy.

### Policy Iteration

Policy iteration includes policy evaluation and policy improvement in order to finally converge the policy. Here you start with a random policy and then find the value function of this random policy and make improvements to the policy based on the value function. In subsequent steps, the policy goes through improvements until the policy converges.

We will use grid world example to elaborate on this. We will also use the negative grid on this and have -0.1 reward for each non terminal state. 

As mentioned earlier, we will begin with a random policy, calculate the value function and make improvement on the policy.

![1](https://user-images.githubusercontent.com/7417075/42298259-38b7569c-7fc2-11e8-8413-fd8103997c89.png)

We now move the calculating the value function capturing max change. If the change is very small, we proceed to policy improvement phase.
 
![2](https://user-images.githubusercontent.com/7417075/42298260-38c9beae-7fc2-11e8-81dd-f73dac6f9f19.png)

For policy improvement, we assign the old values to old_a and look for all possible actions to find the best value. We choose the action that gives us the best value. At the end we check if new_a = old_a. If they are equal, the policy is said to be converged and we have our final policy. If not, the process continues.

![3](https://user-images.githubusercontent.com/7417075/42298261-38dcc918-7fc2-11e8-8fb4-68c4e4f8f83f.png)

Here is the final policy we get through policy iteration.

![4](https://user-images.githubusercontent.com/7417075/42298262-38efddaa-7fc2-11e8-863c-18d2440fde46.png)

### Value Iteration
Value iteration does not calculate the policy but rather the optimal value function and selects any policy that meets the criteria because a policy with an optimal value function should also be optimal.
Policy iteration step may not be the best option because there is on iterative step inside another so value iteration may be a better option. The value iteration function combines policy evaluation and policy improvement. This equation takes the maximum of all possible actions. The codes for policy iteration and value iteration may look similar.
 
![5](https://user-images.githubusercontent.com/7417075/42298263-39035bdc-7fc2-11e8-888f-29d6e3cf8c91.png)

Here is the result from value iteration. The result from both options yield the same values.

![6](https://user-images.githubusercontent.com/7417075/42298264-3917e98a-7fc2-11e8-993c-e0c86c95a1b6.png)

# MONTE CARLO

##### ***Refer to Monte Carlo Combined.ipynb file for code***

In previous evaluations, we assumed that we were fully aware of the environment and all possible states. We played a simple board game and tic tac toe where it was valid. However, in real life reinforcement learning, this is not always possible—we become more aware of the environment and the different states within in as we play the game or gain experience. In this experiment, we will go through both options and see how they compare.

Another variation here is starting from a fixed position vs being able to start from any position. The later one is called exploring starts or god mode. In this condition, we use epsilon greedy policy instead of a greedy policy to allow for exploration. While in some cases, there is a fixed starting position, like when playing a game with a predetermined start like Mario vs another game where all states are available as a starting position like tic tac toe. 

In Monte Carlo, the event is let to run for an entire episode, expected mean (we take stochastic approach than deterministic) is calculated and then the value is updated, rather than updating the value after each action. To do this, a game is played for multiple times to collect the sample and at the end, sample mean of all returns is calculated.

There are two aspects in Monte Carlo. One is policy evaluation (predicting problem) and policy optimization (control problem).

### POLICY EVALUATION STEP
First, we look at the policy evaluation part with a deterministic model where we list out all the possible actions on the grid as 'U', 'D', 'L', 'R'. In this case, the policy reaches the terminal state but does not factor in the rewards. As such, although the optimal policy should lead to the terminal state of 0,3, which has a reward of 1, it does not. There are three states underlined in blue that lead to a terminal state of 1,3 with a reward of -1.

![1](https://user-images.githubusercontent.com/7417075/42256786-5cce3846-7f10-11e8-89e3-1b200f732f01.png)

But this stage is just about calculating the value, which is shown as following.

![2](https://user-images.githubusercontent.com/7417075/42256787-5ce1b4fc-7f10-11e8-8fdf-f8361931c8c0.png)

The second variation here is to make the policy stochastic thus adding an element of probability and making the policy win, unlike our last policy.

As shown below, now the policy aims to win by making 0,3 its terminal state which has a reward of 1.

![3](https://user-images.githubusercontent.com/7417075/42256788-5d319116-7f10-11e8-9512-29e065b9aa70.png)

Notice that in this policy, which is designed to win, the value increases as you get closer to the terminal state with the highest return(0,3).

![4](https://user-images.githubusercontent.com/7417075/42256789-5d44cc2c-7f10-11e8-909d-163e62712d6c.png)

### POLICY OPTIMIZATION STEP

Now we try to find an optimal policy from all possibilities. For this, we do not define a policy like we did for policy evaluation--we select a random policy. However, until now, we were only finding expected value, V(s) based on a state but now we need to find expected value, V(s, a) based on a state and the action taken. This way we can identify which set of actions(policy) has the maximum value to identify the optimum policy.
This is calculated by taking argmax of Q(s,a).
Here is what the outcome looks like with exploring start method where all possible states have the likelihood of being the initial step—or the first action is uniformly random with a greedy policy.

![5](https://user-images.githubusercontent.com/7417075/42256790-5d5859e0-7f10-11e8-84ef-126221b56f47.png)

However, it is not always possible to select a random initial step. Sometimes, the initial step is already set. We will do the same here. We will remove the randomness on the initial action and set a fixed starting point. We will also change our policy to epsilon greedy from greedy so that we do a little exploration while finding our optimal policy.

![6](https://user-images.githubusercontent.com/7417075/42256791-5d6aadd4-7f10-11e8-9310-bcc46cbb0db6.png)


# TEMPORAL DIFFERENCE

##### ***Refer to TD Learning.ipynb file for code***

Temporal Difference Learning combines features from both Dynamic Programming and Monte Carlo to solve the Markov Decision Process. In fact, it tries to overcome the drawbacks from both methods. For instance, Dynamic Programming requires all states to be listed, which isn’t always possible, so it learns from experience, like in Monte Carlo. Similarly, Monte Carlo requires an episode to be complete before updating the estimates, but TD Learning can improve its estimates based on its existing estimates, like in Dynamic Programming.

As in Monte Carlo, we have two steps to solving the MDP through TD Learning. First is to calculate the value function and then find the optimal policy. In this case, we use TD(0) for calculating value function and Q learning for optimization.

### Value Function through TD(0)

This is the step in TD Learning where we overcome the drawback of Monte Carlo to wait for an episode to complete calculate the return. This is because TD(0) algorithm just needs to reach to the next state to get the value for the present state as r + γV(s’). As such, we can improve the performance within an episode itself, which is very helpful if an episode is very long. We also don’t need the full environment because we only calculate returns for the state we visit.

As we only calculate value for the state we visit, our model may not calculate values for all states, especially if the model is deterministic. As such, we use epsilon soft to do a bit of exploration. However, I wanted to experiment what would happen if there was no exploration so I made the value of epsilon negligible and the model just followed one policy and calculated values for states within this policy. Values for other states were unexplored and their values were 0, as shown below.

![1](https://user-images.githubusercontent.com/7417075/42291981-6e8ff552-7f8c-11e8-866d-6373dffdc9ae.png)

We use a standard grid and our initial policy. Here is how we calculate the value for each state once we reach s’, or in this case, s2.

![2](https://user-images.githubusercontent.com/7417075/42291982-6ea1d18c-7f8c-11e8-9454-a65c9ef9f7e5.png)

Notice that this is value calculation step, so the policy is not optimal.

![3](https://user-images.githubusercontent.com/7417075/42291983-6eb25a3e-7f8c-11e8-81ad-64af01228fb7.png)


### Optimizing through Q Learning

Q Learning is unique because it takes a very different approach than we have been taking so far for optimizing. We started off with evaluating a given policy and then moved to improving the policy using a greedy method, which is also called on-policy method because we a following a policy. However, Q learning is different as we do not have to do these steps and there is no set policy we are using or improving, which is why it is off-policy. It assumes that we can take any random action and still be able to calculate the optimum value and the optimum policy.

But there is a cost to taking random actions and not following a policy. It will take longer to arrive at the same result and thus an episode will take longer to finish resulting in a suboptimal model.

In the code, rather than taking the next action based on max value, a random action is taken every time with just epsilon greedy.

![4](https://user-images.githubusercontent.com/7417075/42291984-6ecad14a-7f8c-11e8-9b68-7fa3820462cd.png)

Here are the values and policy through Q Learning. Unlike in T(0), now the policy is optimal.

![5](https://user-images.githubusercontent.com/7417075/42291980-6e7b9a4e-7f8c-11e8-8dee-757b5949e26f.png)
