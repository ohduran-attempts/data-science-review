import gym
import tensorflow as tf
import numpy as np


env = gym.make('CartPole-v0')

observations = env.reset()

env = gym.make('CartPole-v0')

observations = env.reset()

saver = tf.train.Saver()

with tf.Session() as sess:
    saver.restore(sess, "./tmp/model.ckpt")
    print("Model restored.")

    for x in range(500):
        env.render()
        action_val, gradients_val = sess.run([actions, gradients], feed_dict={X:observations.reshape(1, num_inputs)})
        observations, reward, done, info = env.step(action_val[0][0])
