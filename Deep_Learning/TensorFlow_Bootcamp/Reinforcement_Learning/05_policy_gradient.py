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

import gym
import tensorflow as tf
import numpy as np

num_inputs = 4
num_hidden = 4
num_outputs = 1

learning_rate = 0.01

initialize = tf.contrib.layers.variance_scaling_initializer()

X = tf.placeholder(tf.float32, shape=[None, num_inputs])
hidden_layer =tf.layers.dense(X, num_hidden, activation=tf.nn.relu, kernel_initializer=initializer)

logit = tf.layers.dense(hidden_layer, num_outputs)
outputs = tf.nn.sigmoid(logits)

probabilities = tf.concat(axis=1, values = [outputs, 1 - outputs])
actions = tf.multinomial(probabilities, num_samples=1)

y = 1.0 - tf.to_float(action)

# gradients off cross_entropy
cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=logits)
optimizer = tf.train.AdamOptimizer(learning_rate)

gradients_and_variables = optimizer.compute_gradients(cross_entropy)


# Compute gradients
gradients = []
gradient_placeholders = []
grads_and_vars_feed = []

for gradient, variable in gradients_and_variables:
    gradients.append(gradient)
    gradient_placeholder = tf.placeholder(tf.float32, shape=gradient.get_shape())
    gradient_placeholders.append(gradient_placeholder)
    grads_and_vars_feed.append(gradient_placeholder, variable)

training_optimizer = optimizer.apply_gradients(grads_and_vars_feed)

init =tf.global_variables_initializer()
saver = tf.saver.Saver()

def helper_discount_reward(rewards, discount_rate):
    """"Takes in rewards and applies discount rate."""
    discounted_rewards = np.zeros(len(rewards))
    cumulative_rewards = 0
    for step in reversed(range(len(rewards))):
        cumulative_rewards = rewards[step] + cumulative_rewards*discount_rate
        discounter_rewards[step] = cumulative_rewards
    return discounted_rewards

def discount_and_normalize_rewards(all_rewards, discount_rate):
    """Takes in all rewards, applies helper_discount function and then normalizes using mean and std."""
    all_discounted_rewards = [helper_discount_rewards(rewards, discount_rate) for rewards in all_rewards]

    flat_rewards = np.concatenate(all_discounted_rewards)
    reward_mean = flat_rewards.mean()
    reward_std = flat_rewards.std()

    return [(d_w - reward_mean) / reward_std for d_w in all_discounted_rewards]
