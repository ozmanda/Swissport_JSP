import gym
import JSSEnv
import argparse


if __name__ == '__main__':
    env = gym.make('jss-v1', env_config={'instance_path': 'venv/Lib/site-packages/JSSEnv/envs/instances/test1'})
    observation = env.reset()

    for t in range(100):
        env.render()
        print(observation)
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        print(f'Observation:\n{observation}')
        print(f'Reward: {reward}')
        if done:
            print(f'finished after {t} timesteps')


    x=5
