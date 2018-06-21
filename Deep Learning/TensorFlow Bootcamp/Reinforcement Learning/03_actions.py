"""We are now going to create a policy: if the pole falls to the right, move it to the right to adjust."""
import gym

env = gym.make('CartPole-v0')

# print(env.action_space) # Discrete(2) - that is, the total actions that can be performed are 2.
# print(env.observation_space) # Box(4,), - that is, the total observations returned are 4.

observation = env.reset()

for t in range(1000):
    env.render()

    cart_pos, cart_vel, pole_ang, ang_vel = observation

    # if the angle is positive (moving to the right), then move the cart to the right (action=1).
    if pole_ang > 0:
        action = 1
    # if not, move it to the left (action = 0)
    else:
        action = 0

    # feed the action to the environment and see what comes back.
    observation, reward, done, info = env.step(action)
