import gym
import tensorflow as tf
import numpy as np

init =tf.global_variables_initializer()
env = gym.make('CartPole-v0')

observations = env.reset()

with tf.Session() as sess:
    sess.run(init)
    saver = tf.train.Saver()
    saver.restore(sess, "./tmp/model.ckpt")
    print("Model restored.")

    for x in range(500):
        env.render()
        action_val, gradients_val = sess.run([actions, gradients], feed_dict={X:observations.reshape(1, num_inputs)})
        observations, reward, done, info = env.step(action_val[0][0])
