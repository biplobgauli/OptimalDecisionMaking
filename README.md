# OptimalDecisionMaking

### Multi-Arm Bandit Problem

Multi-arm bandit is a classic example of reinforcement learning where there are multiple options, in this case slot machines, and the player needs to identify the slot with the maximum payout so that the profit is maximized.

However, the caveat is that resource is limited and there may not be enough time or slot pulls available, but the player still needs to maximize the payout. If the player spends more time figuring out which slot has the highest return, the opportunity cost, or in this case exploit, increases. Conversely, the effect of quickly deciding on a slot without a thorough exploration might lead to choosing a sub-optimal slot machine. As such, there is always a tradeoff between exploit and explore.

For this experiment, we free ourselves from the aforementioned tradeoff and let the experiment run for say 100,000 times so that we can analyze all potential sub-optimal choices and at the end find out the optimal choice.

We are using epsilon greedy algorithm in this experiment because it is effective and easy to understand and implement. We start with three slot machines and assign them epsilon values (0.1, 0.05 and 0.01). How this works is we first randomly choose a probability value and compare it against our set epsilon value. If it is less than our epsilon, we randomly select a slot but if it is higher, we take the maximum mean value for the slot machine, and this is where the greedy epsilon algorithm comes into play.

We are running the experiment for 100,000 times so that we can clearly see the trend in payouts for all bandits and calculate their means. If we just use a linear scale for the plots, we cannot see the initial volatility in the trending so we use a log scale.

Just to experiment, I added another slot machine with epsilon value 0.001 and ran the experiment. We can now see the results in the graph for all four slot machines. In this case, looks like the second slot with epsilon value 0.05 gives a maximum payout.

Refer to the notebook file bandit.ipynb for details.

![bandit](https://user-images.githubusercontent.com/7417075/40161526-4ad94f6c-596e-11e8-9138-00f2eac10e68.PNG)



## Monte Carlo

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

