# OptimalDecisionMaking

Week 1: Multi-Arm Bandit Problem


Multi-arm bandit is a classic example of reinforcement learning where there are multiple options, in this case slot machines, and the player needs to identify the slot with the maximum payout so that the profit is maximized.

However, the caveat is that resource is limited and there may not be enough time or slot pulls available, but the player still needs to maximize the payout. If the player spends more time figuring out which slot has the highest return, the opportunity cost, or in this case exploit, increases. Conversely, the effect of quickly deciding on a slot without a thorough exploration might lead to choosing a sub-optimal slot machine. As such, there is always a tradeoff between exploit and explore.

For this experiment, we free ourselves from the aforementioned tradeoff and let the experiment run for say 100,000 times so that we can analyze all potential sub-optimal choices and at the end find out the optimal choice.

We are using epsilon greedy algorithm in this experiment because it is effective and easy to understand and implement. We start with three slot machines and assign them epsilon values (0.1, 0.05 and 0.01). How this works is we first randomly choose a probability value and compare it against our set epsilon value. If it is less than our epsilon, we randomly select a slot but if it is higher, we take the maximum mean value for the slot machine, and this is where the greedy epsilon algorithm comes into play.

We are running the experiment for 100,000 times so that we can clearly see the trend in payouts for all bandits and calculate their means. If we just use a linear scale for the plots, we cannot see the initial volatility in the trending so we use a log scale.

Just to experiment, I added another slot machine with epsilon value 0.001 and ran the experiment. We can now see the results in the graph for all four slot machines. In this case, looks like the second slot with epsilon value 0.05 gives a maximum payout.

![bandit](https://user-images.githubusercontent.com/7417075/40161526-4ad94f6c-596e-11e8-9138-00f2eac10e68.PNG)
