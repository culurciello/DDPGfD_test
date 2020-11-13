# E. Culurciello
# November 2020

# DDPGfd test from: https://github.com/MrSyee/pg-is-all-you-need/blob/master/06.DDPGfD.ipynb

import torch
import gym
import argparse
from rl_modules.ddpg_agent import DDPGfDAgent
from rl_modules.utils import ActionNormalizer

def get_args():
    parser = argparse.ArgumentParser(description='DDPGfD test on Pendulum-v0')
    arg = parser.add_argument
    arg('trained_file', type=str, default='saved_models/model_20000.pth', help='environment name')
    # env:
    arg('--env_name', type=str, default='Pendulum-v0', help='environment name')
    # train:
    arg('--seed', type=int, default=543, help='')
    arg('--num_frames', type=int, default=50000, help='')
    arg('--memory_size', type=int, default=100000, help='')
    arg('--batch_size', type=int, default=128, help='')
    arg('--ou_noise_theta', type=float, default=1.0, help='')
    arg('--ou_noise_sigma', type=float, default=0.1, help='')
    arg('--initial_random_steps', type=int, default=10000, help='')
    arg('--n_step', type=int, default=3, help='')
    arg('--pretrain_step', type=int, default=1000, help='')
    arg('--save_dir', type=str, default='saved_models/', help='path to save the models')

    args = parser.parse_args()
    return args

args = get_args() # Holds all the input arguments
print(args)

# create the environment
env = gym.make(args.env_name)
env = ActionNormalizer(env)
env.seed(args.seed)

args.demo = None
# DDPGfD agent:
agent = DDPGfDAgent(
    args,
    env,
)

print('Loading model:', args.trained_file)

# load pre-trained networks for task:
pt_model = torch.load(args.trained_file)
agent.actor.load_state_dict(pt_model[0])
agent.actor.eval()

for i in range(100):
    frames = agent.test()