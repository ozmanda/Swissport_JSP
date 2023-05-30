import gym
import os
from DJSP_env import DJSPEnv
import argparse
from vis_utils import render_schedule


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('instancepath', type=str, help='Relative path to instance specification')
    args = parser.parse_args()
    assert os.path.isfile(args.instancepath)

    env = DJSPEnv(instance_path=args.instancepath)
    observation = env.reset()
    # print(f'INITIAL ASSIGNMENT:\n{env.assignment}')

    for t in range(100):
        # print(f'Availability:\n{env.availability.astype(int)}')
        action = env.sample_action()
        observation, reward, done = env.step(action)
        # print(f'Assignment:\n{env.assignment.astype(int)}')
        # print(f'Unassigned operations: {observation["unassigned operations"]}\n\n')
        if done:
            print(f'finished after {t+1} timesteps')
            # print(f'Assignment: \n{env.assignment}')
            break
        if t == 99:
            print('Automatic termination after 100 timesteps')

    render_schedule(env.assignment, env.machines_per_op, env.aircraft, env.operation_times)

