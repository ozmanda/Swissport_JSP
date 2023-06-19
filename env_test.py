import os
import argparse
from DJSP_env import DJSPEnv
from vis_utils import render_schedule

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('instancepath', type=str, help='Relative path to instance specification')
    args = parser.parse_args()
    assert os.path.isfile(args.instancepath)

    env = DJSPEnv(instance_path=args.instancepath)
    obs, _ = env.reset()
    while True:
        action = env.sample_action()
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            print(f'Total reward: {env.rewards["total reward"]}')
            render_schedule(env.assignment, env.machines_per_op, env.aircraft, env.operation_times)
            break