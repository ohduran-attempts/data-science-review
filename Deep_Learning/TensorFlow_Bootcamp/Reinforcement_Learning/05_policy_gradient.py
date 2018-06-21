"""Policy gradient.

The network in 04 did not work very well, because we aren't considering the history of our actions,
but only considering a single previous action.

This is often called an __assignment of credit__ problem:
Which actions should be credited when the agent gets rewarded at time t, only actions at t-1, or the series of historical actions?

We solve this problem by applying a discount rate. We evaluate an action based off all the rewards that come after the action,
not just the immediate first reward.

We choose a discount rate D, typically 0.95 to 0.99, applied to each step n times, in reverse.

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

initializer = tf.contrib.layers.variance_scaling_initializer()

X = tf.placeholder(tf.float32, shape=[None, num_inputs])
hidden_layer =tf.layers.dense(X, num_hidden, activation=tf.nn.relu, kernel_initializer=initializer)

logits = tf.layers.dense(hidden_layer, num_outputs)
outputs = tf.nn.sigmoid(logits)

probabilities = tf.concat(axis=1, values = [outputs, 1 - outputs])
actions = tf.multinomial(probabilities, num_samples=1)

y = 1.0 - tf.to_float(actions)

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
    grads_and_vars_feed.append((gradient_placeholder, variable))

training_op = optimizer.apply_gradients(grads_and_vars_feed)



init =tf.global_variables_initializer()
saver = tf.train.Saver()

def helper_discount_reward(rewards, discount_rate):
    """"Takes in rewards and applies discount rate."""
    discounted_rewards = np.zeros(len(rewards))
    cumulative_rewards = 0
    for step in reversed(range(len(rewards))):
        cumulative_rewards = rewards[step] + cumulative_rewards*discount_rate
        discounted_rewards[step] = cumulative_rewards
    return discounted_rewards

def discount_and_normalize_reward(all_rewards, discount_rate):
    """Takes in all rewards, applies helper_discount function and then normalizes using mean and std."""
    all_discounted_rewards = [helper_discount_reward(rewards, discount_rate) for rewards in all_rewards]

    flat_rewards = np.concatenate(all_discounted_rewards)
    reward_mean = flat_rewards.mean()
    reward_std = flat_rewards.std()

    return [(d_w - reward_mean) / reward_std for d_w in all_discounted_rewards

    ]


# Session
env = gym.make('CartPole-v0')

num_game_rounds = 10
max_game_steps = 1000
num_iterations = 650

discount_rate = 0.9

with tf.Session() as sess:

    sess.run(init)

    for iteration in range(num_iterations):
        print('On iteration: {}'.format(iteration))

        all_rewards = []
        all_gradients = []

        for game in range(num_game_rounds):

            current_rewards = []
            current_gradients = []

            observations = env.reset()

            for step in range(max_game_steps):

                action_val, gradients_val = sess.run([actions, gradients], feed_dict={X:observations.reshape(1, num_inputs)})

                observations, reward, done, info = env.step(action_val[0][0])

                current_rewards.append(reward)
                current_gradients.append(gradients_val)

                if done:
                    break

            all_rewards.append(current_rewards)
            all_gradients.append(current_gradients)

        all_rewards = discount_and_normalize_reward(all_rewards, discount_rate)
        feed_dict = {}

        for var_index, gradient_placeholder in enumerate(gradient_placeholders):
            mean_gradients = np.mean([reward * all_gradients[game_index][step][var_index]
                                                                    for game_index, rewards in enumerate(all_rewards)
                                                                    for step, reward in enumerate(rewards)], axis=0)
            feed_dict[gradient_placeholder] = mean_gradients

        sess.run(training_op, feed_dict=feed_dict)

        print('Saving graph and session')

        save_path = saver.save(sess, "./tmp/model.ckpt")
        print("Model saved in file: {}".format(save_path))


##############################

### Run trained Model On Environment ##

##############################

env = gym.make('CartPole-v0')

observations = env.reset()

with tf.Session() as sess:
    saver.restore(sess, "./tmp/model.ckpt")
    print("Model restored.")

    for x in range(500):
        env.render()
        action_val, gradients_val = sess.run([actions, gradients], feed_dict={X:observations.reshape(1, num_inputs)})
        observations, reward, done, info = env.step(action_val[0][0])
