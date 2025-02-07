#!/bin/zsh

conda activate rl

python source/standalone/workflows/sb3/train.py --task Isaac-Velocity-Rough-Digit-V3-L2T-v0 --num_envs 4096 --headless --video

# python source/standalone/workflows/sb3/train_ts.py --task Isaac-Velocity-Rough-Digit-V3-L2T-v0 --num_envs 4096 --headless --video
