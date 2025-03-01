#!/bin/zsh

python scripts/reinforcement_learning/rlopt/train.py --task Isaac-Velocity-Rough-Digit-V3-L2T-v0 --num_envs 4096 agent.mixture_coeff=0.1 --headless --note mc0.1

python scripts/reinforcement_learning/rlopt/train.py --task Isaac-Velocity-Rough-Digit-V3-L2T-v0 --num_envs 4096 agent.mixture_coeff=0.3 --headless --note mc0.3

python scripts/reinforcement_learning/rlopt/train.py --task Isaac-Velocity-Rough-Digit-V3-L2T-v0 --num_envs 4096 agent.mixture_coeff=0.4 --headless --note mc0.4
