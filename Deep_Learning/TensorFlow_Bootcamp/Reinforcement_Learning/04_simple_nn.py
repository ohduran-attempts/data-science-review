#!env/bin/python3
"""Simple Neural Network.

let's design a simple Neural Network that takes in the observation array,
passes it through a hidden layer and outputs 2 probabilities,
one for the left and one for the right.

We will then choose a random action, weighted by the probabilities.

Notice how we don't choose just automatically the highest probability for our decision. This is
to balance trying out new actions versus constantly choosing well known ones.
"""
import tensorflow as tf
import gym
import numpy as np

num_inputs = 4
num_hidden = 4
num_outputs = 1 # Probability to go left

initializer = tf.contrib.layers.variance_scaling_intializer()

X = tf.placeholder(tf.float32, shape=[None, num_inputs])

hidden_layer_one = tf.layers.dense(X, num_hidden, activation=tf.nn.relu, kernel_initializer = initializer)

hidden_layer_two = tf.layers.dense(hidden_layer_one, num_hidden, activation=tf.nn.relu, kernel_initializer = initializer)

output_layer = tf.layers.dense(hidden_layer_two, num_outputs, activation=tf.nn.sigmoid, kernel_initializer = initializer)

probabilities = tf.concat(axis=1, values=[output_layer, 1 - output_layer])

action = tf.multinomial(probabilities, num_samples= 1)

initializer = tf.global_variables_initializer()

epi = 50
step_limit = 500
env = gym.make('CartPole-v0')
avg_steps = []

with tf.Session() as sess:
    init.run()

    for i_episode in range(epi):
        observation.reset()
        for step in range(step_limit):
            action_value = action.eval(feed_dict={X:observation.reshape(1, num_inputs)})

            observation, reward, done, info = env.step(action_value[0][0]) # 0 or 1 passed onto env.step()

            if done:
                avg_steps.append(step)
                print('Done after', step, 'steps')
                break
print('after', epi, 'episodes, the average steps per game was', np.mean(avg_steps))
