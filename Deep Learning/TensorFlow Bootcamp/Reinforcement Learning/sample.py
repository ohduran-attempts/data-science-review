"""Let's explore how to create an example environment with Open AI Gym with CartPole.

Goal of CartPole environment is to balance the pole on the cart. 0 moves to the left, 1 to the right.
Environment is a numpy array with 4 floating point numbers: Horizontal position, Horizontal velocity, Angle of Pole, Angular velocity.
"""

import gym

env = gym.make('CartPole-v0')

env.reset () # default values

for t in range(1000):

    env.render()

    env.step(env.action_space.sample()) # randomly choose an action from the available options

    
