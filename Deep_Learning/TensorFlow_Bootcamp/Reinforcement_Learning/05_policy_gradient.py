"""Policy gradient.

The network in 04 did not work very well, because we aren't considering the history of our actions,
but only considering a single previous action.

This is often called an __assignment of credit__ problem:
Which actions should be credited when the agent gets rewarded at time t, only actions at t-1, or the series of historical actions?

We solve this problem by applying a discount rate. We evaluate an action based off all the rewards that come after the action,
not just the immediate first reward.

We choose a discount rate D, typically 0.95 to 0.99:
Score :R(0) + R(1) * D + R(2)*D² ... = sum(R(n)*D^n)

The closer D is to 1, the more weight future rewards have. Closer to 0, future rewards don't count as much as immediate rewards.
Choosing a discount rate often depends on the specific environment and whether actions have short or long term effects.

Due to this delayed effect, good actions may sometimes receive bad scores due to bad actions that follow,
unrelated to their initial action. To counter this, we train over many episodes.

We must also then normalize the action scores by substracting the mean and dividing by the standard deviation.
These extra steps can significantly increase training time for complex environments.

Steps:
- Neural Networks play several episodes.
- The optimizer will calculate the gradients (instead of calling minimize).
- Compute each aaction's discounted and normalized score.
- Then multiply the gradient vector by the action's score.
- Negative scores will create opposite gradients when multiplied.
- Calculate mean of the resulting gradient vector for Gradient Descent.
"""
