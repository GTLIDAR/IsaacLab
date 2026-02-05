# Feiyang Wu (feiyangwu@gatech.edu), based on sb3/trian.py

"""Script to train RL agent with Stable Baselines3."""

"""Launch Isaac Sim Simulator first."""

import argparse
import signal
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with Stable-Baselines3.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rlopt_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--log_interval", type=int, default=100_000, help="Log data every n timesteps.")
parser.add_argument("--checkpoint", type=str, default=None, help="Continue the training from checkpoint.")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument("--export_io_descriptors", action="store_true", default=False, help="Export IO descriptors.")
parser.add_argument(
    "--algo",
    "--algorithm",
    dest="algorithm",
    type=str.upper,
    default="PPO",
    choices=["PPO", "SAC", "IPMD"],
    help="RLOpt algorithm to train (must match the agent config).",
)

parser.add_argument(
    "--ray-proc-id", "-rid", type=int, default=None, help="Automatically configured by Ray integration, otherwise None."
)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


def cleanup_pbar(*args):
    """
    A small helper to stop training and
    cleanup progress bar properly on ctrl+c
    """
    import gc

    tqdm_objects = [obj for obj in gc.get_objects() if "tqdm" in type(obj).__name__]
    for tqdm_object in tqdm_objects:
        if "tqdm_rich" in type(tqdm_object).__name__:
            tqdm_object.close()
    raise KeyboardInterrupt


# disable KeyboardInterrupt override
signal.signal(signal.SIGINT, cleanup_pbar)

"""Rest everything follows."""

import logging
import os
import random
import time
from datetime import datetime

import gymnasium as gym
import torch
from rlopt.agent import IPMD, PPO, SAC
from torchrl.envs import (
    Compose,
    ObservationNorm,
    RewardClipping,
    RewardSum,
    StepCounter,
    TransformedEnv,
    VecNormV2,
)

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
)
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_yaml

from isaaclab_rl.rlopt import IsaacLabTerminalObsReader, IsaacLabWrapper, RLOptConfig

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config

torch.set_float32_matmul_precision("high")

# import logger
logger = logging.getLogger(__name__)
# PLACEHOLDER: Extension template (do not remove this comment)

ALGORITHM_CLASS_MAP = {
    "PPO": PPO,
    "SAC": SAC,
    "IPMD": IPMD,
}

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


def resolve_agent_cfg_entry_point(task_name: str | None, agent_entry_point: str, algorithm: str) -> str:
    """Resolve the agent config entry point based on algorithm and task registry."""
    if agent_entry_point != "rlopt_cfg_entry_point" or task_name is None:
        return agent_entry_point
    task_id = task_name.split(":")[-1]
    algo_entry_point = f"rlopt_{algorithm.lower()}_cfg_entry_point"
    try:
        spec = gym.spec(task_id)
    except Exception as exc:
        logger.warning("Could not resolve task '%s' from registry: %s", task_id, exc)
        return agent_entry_point
    if spec.kwargs.get(algo_entry_point) is not None:
        if algo_entry_point != agent_entry_point:
            print(f"[INFO] Using agent config entry point: {algo_entry_point}")
        return algo_entry_point
    if algorithm != "PPO":
        logger.warning(
            "No algorithm-specific agent config for '%s' (expected '%s'); using '%s'.",
            task_id,
            algo_entry_point,
            agent_entry_point,
        )
    return agent_entry_point


args_cli.agent = resolve_agent_cfg_entry_point(args_cli.task, args_cli.agent, args_cli.algorithm)


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RLOptConfig):
    """Train with stable-baselines agent."""
    # randomly sample a seed if seed = -1
    if args_cli.seed == -1:
        args_cli.seed = random.randint(0, 10000)

    # override configurations with non-hydra CLI arguments
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    agent_cfg.env.num_envs = env_cfg.scene.num_envs
    agent_cfg.env.env_name = args_cli.task
    agent_cfg.seed = args_cli.seed if args_cli.seed is not None else agent_cfg.seed
    # max iterations for training
    if args_cli.max_iterations is not None:
        agent_cfg.collector.total_frames = (
            args_cli.max_iterations * agent_cfg.collector.total_frames * env_cfg.scene.num_envs
        )
    agent_cfg.collector.frames_per_batch *= env_cfg.scene.num_envs
    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    if args_cli.algorithm == "SAC":
        # Off-policy: replay buffer and SAC-specific settings
        agent_cfg.collector.init_random_frames = 0
        agent_cfg.sac.utd_ratio = 24.0 / 4096.0 * 24  # ~num_envs updates per batch
        agent_cfg.replay_buffer.size = 1_000_000
        agent_cfg.loss.mini_batch_size = 4906 * 6  # Smaller batch for off-policy
        agent_cfg.optim.lr = 3e-4
        agent_cfg.replay_buffer.prb = False
        agent_cfg.replay_buffer.prefetch = 3
        agent_cfg.compile.compile = False
        agent_cfg.compile.compile_mode = "default"
        agent_cfg.compile.cudagraphs = False
        agent_cfg.sac.skip_done_states = False
    # IPMD is PPO-based: use PPO-style config (value function, epochs, no SAC/replay overrides)

    # directory for logging into
    run_info = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_root_path = os.path.abspath(os.path.join("logs", "rlopt", args_cli.task))
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    # The Ray Tune workflow extracts experiment name using the logging line below, hence,
    # do not change it (see PR #2346, comment-2819298849)
    print(f"Exact experiment name requested from command line: {run_info}")
    log_dir = os.path.join(log_root_path, run_info)
    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)

    # save command used to run the script
    command = " ".join(sys.orig_argv)
    (Path(log_dir) / "command.txt").write_text(command)

    # set the IO descriptors export flag if requested
    if isinstance(env_cfg, ManagerBasedRLEnvCfg):
        env_cfg.export_io_descriptors = args_cli.export_io_descriptors
    else:
        logger.warning(
            "IO descriptors are only supported for manager based RL environments. No IO descriptors will be exported."
        )

    # set the log directory for the environment (works for all environment types)
    env_cfg.log_dir = log_dir

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        raise NotImplementedError("DirectMARLEnv is not supported for RLOpt training yet.")

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)  # type: ignore

    start_time = time.time()

    env = IsaacLabWrapper(env)  # type: ignore
    env = env.set_info_dict_reader(
        IsaacLabTerminalObsReader(observation_spec=env.observation_spec, backend="gymnasium")  # type: ignore
    )
    env = TransformedEnv(
        env=env,
        transform=Compose(
            RewardSum(),  # type: ignore
            StepCounter(1000),  # type: ignore
            VecNormV2(in_keys=agent_cfg.policy.input_keys + ["reward"]),
        ),
    )

    agent_class = ALGORITHM_CLASS_MAP[args_cli.algorithm]
    agent = agent_class(
        env=env,
        config=agent_cfg,  # type: ignore
    )

    # run training
    agent.train()

    # close the simulator
    env.close()

    print(f"Training time: {round(time.time() - start_time, 2)} seconds")

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()  # type: ignore
