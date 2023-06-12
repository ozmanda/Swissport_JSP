import gym
import os
import argparse
from DJSP_env import DJSPEnv
from vis_utils import render_schedule

from stable_baselines3 import DQN, PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from gymnasium.wrappers import FlattenObservation
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.common.envs import InvalidActionEnvDiscrete


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('instancepath', type=str, help='Relative path to instance specification')
    parser.add_argument('episodes', type=int, help='Number of episodes to train the RL agent on')
    args = parser.parse_args()
    assert os.path.isfile(args.instancepath)

    env = DJSPEnv(instance_path=args.instancepath)
    model = DQN('MlpPolicy', env, verbose=1, tensorboard_log='./tensorboard/DQN_DJSPEnv/')
    dim = env.n_aircraft * env.n_operations
    model.learn(total_timesteps=dim*100000)

    for episode in range(0, args.episodes):
        observation, _ = env.reset()
        print(f'\n\nEPISODE {episode+1} / {args.episodes}......')
        terminated = False
        while not terminated:
            action = env.sample_action()
            observation, reward, terminated, truncated, info = env.step(action)
        render_schedule(env.assignment, env.machines_per_op, env.aircraft, env.operation_times)

    # t = True
    # obs, _ = env.reset()
    # while t:
    #     action, _states = model.predict(obs)
    #     obs, reward, terminated, truncated, info = env.step(action)
    #     if terminated or truncated:
    #         render_schedule(env.assignment, env.machines_per_op, env.aircraft, env.operation_times)
    #         t = False

        # # reset the environment and set terminated to False
        # observation, _ = env.reset()
        # terminated = False
        # t = 0
        #

        #
        #
        # # evaluate episode results
        # print(f'    reward: {env.rewards["total reward"]}')
        # render_schedule(env.assignment, env.machines_per_op, env.aircraft, env.operation_times)

