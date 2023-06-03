import gym
import os
from DJSP_env import DJSPEnv
import argparse
from vis_utils import render_schedule


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('instancepath', type=str, help='Relative path to instance specification')
    parser.add_argument('episodes', type=int, help='Number of episodes to train the RL agent on')
    args = parser.parse_args()
    assert os.path.isfile(args.instancepath)

    env = DJSPEnv(instance_path=args.instancepath)

    for episode in range(1, args.episodes):
        print(f'Episode {episode} / {args.episodes}......')
        # reset the environment and set terminated to False
        observation = env.reset()
        terminated = False
        t = 0

        while not terminated:
            print(f'    iteration {t}........................')
            action = env.sample_action()
            observation, reward, terminated = env.step(action)
            t += 1


        # evaluate episode results
        print(f'Episode {episode}: {env.rewards["total reward"]}')
        render_schedule(env.assignment, env.machines_per_op, env.aircraft, env.operation_times)



