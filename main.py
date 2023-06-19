# import gym
import os
import argparse
from warnings import warn

import gymnasium as gym

from DJSP_env import DJSPEnv
from vis_utils import render_schedule
from stable_baselines3 import DQN, PPO

# from stable_baselines3.common.vec_env import DummyVecEnv
# from gymnasium.wrappers import FlattenObservation
# from sb3_contrib import MaskablePPO
# from sb3_contrib.common.wrappers import ActionMasker
# from sb3_contrib.common.envs import InvalidActionEnvDiscrete


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('instancepath', type=str, help='Relative path to instance specification')
    parser.add_argument('--modelname', default=None, help='Save name for model run, if none is given it will not '
                                                          'be saved. If a duplicate name is given it will be adjusted'
                                                          'to prevent data loss.')
    parser.add_argument('--tensorboard', default='./tensorboard/PPO_DJSPEnv/', help='Folder to save Tensorboard info')
    parser.add_argument('--test', default=False, type=bool, help='Indicator for model test run')
    parser.add_argument('--episodes', default=None, type=int, help='Number of episodes to train the RL agent on')
    args = parser.parse_args()

    assert os.path.isfile(args.instancepath)

    env = DJSPEnv(instance_path=args.instancepath)
    observation, info = env.reset(seed=42)
    model = PPO('MlpPolicy', env, verbose=1, tensorboard_log=args.tensorboard)
    model.learn(total_timesteps=1000000)

    if args.modelname:
        if os.path.isdir(os.path.join('./models', args.modelname)):
            args.modelname = f'{args.modelname}_1'
            warn(f'Duplicate modelname was given, it has been adjusted to {args.modelname}')
        model.save(path='./models/lunar2')

    if args.test:
        assert args.episodes, 'A number of episodes must be given for testing.'
        for episode in range(0, args.episodes):
            observation, _ = env.reset()
            print(f'\n\nEPISODE {episode + 1} / {args.episodes}......')
            terminated = False
            while not terminated:
                action = env.sample_action()
                observation, reward, terminated, truncated, info = env.step(action)
            render_schedule(env.assignment, env.machines_per_op, env.aircraft, env.operation_times)



