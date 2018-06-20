""" The environment step() function we saw earlier returns back useful objects for our agent.
- Observation: Environment specific information
- Reward: Amount of reward achieved by the previous action. Scaled varies off environmentself.
- Done: Boolean indicating whether environment needs to be reset (game lost).
- Info: Diagnostic information for debugging.
"""
import gym

env = gym.make('CartPole-v0')

print('Initial observation')
observation = env.reset () # default values
print(observation)

for _ in range(2):

    # env.render()

    action = env.action_space.sample() # randomly choose an action from the available options
    observation, reward, done, info = env.step(action)

    print('Performed random action')
    print('observation', observation)
    print('reward', reward)
    print('done', done)
    print('info', info)
