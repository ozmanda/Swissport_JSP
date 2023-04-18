import gym
import os
from DJSP_env import DJSPEnv
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('instancepath', type=str, help='Relative path to instance specification')
    args = parser.parse_args()
    assert os.path.isfile(args.instancepath)

    env = DJSPEnv(instance_path=args.instancepath)
    observation = env.reset()

    for t in range(100):
        env.render()
        print(observation)
        action = env.action_space.sample()
        observation, reward, done = env.step(action)
        print(f'Observation:\n{observation}')
        print(f'Reward: {reward}')
        if done:
            print(f'finished after {t} timesteps')

