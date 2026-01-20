# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Replay reference trajectories and optionally record video."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Replay imitation reference data.")
parser.add_argument(
    "--video", action="store_true", default=False, help="Record videos during replay."
)
parser.add_argument(
    "--video_length",
    type=int,
    default=200,
    help="Length of the recorded video (in steps).",
)
parser.add_argument(
    "--disable_fabric",
    action="store_true",
    default=False,
    help="Disable fabric and use USD I/O operations.",
)
parser.add_argument(
    "--num_envs", type=int, default=None, help="Number of environments to simulate."
)
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent",
    type=str,
    default="rsl_rl_cfg_entry_point",
    help="Agent config entry point (unused).",
)
parser.add_argument(
    "--seed", type=int, default=None, help="Seed used for the environment"
)
parser.add_argument(
    "--real-time",
    action="store_true",
    default=False,
    help="Run in real-time, if possible.",
)
parser.add_argument(
    "--max_steps",
    type=int,
    default=1000,
    help="Maximum replay steps when not recording video.",
)
parser.add_argument(
    "--save_torchrl_rb",
    action="store_true",
    default=False,
    help="Save replay transitions to a TorchRL TensorDict replay buffer (memmap).",
)
parser.add_argument(
    "--torchrl_rb_dir",
    type=str,
    default=None,
    help="Output directory for the TorchRL replay buffer (defaults to <log_dir>/torchrl_rb).",
)
parser.add_argument(
    "--lerobot_dir",
    type=str,
    default=None,
    help="Optional output directory for a LeRobot dataset export.",
)
parser.add_argument(
    "--save_lerobot",
    action="store_true",
    default=False,
    help="Save replay data as a LeRobot dataset (defaults to <log_dir>/lerobot).",
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

"""Rest everything follows."""

import gymnasium as gym
import json
import numpy as np
import os
import time
import torch
from datetime import datetime
from typing import Any, Dict

import isaaclab_tasks  # noqa: F401
from isaaclab.utils.dict import print_dict
from isaaclab_tasks.utils.hydra import hydra_task_config

from tensordict import TensorDict, TensorDictBase

try:
    from torchrl.data import TensorDictReplayBuffer
    from torchrl.data.replay_buffers.storages import LazyMemmapStorage
except ImportError:  # pragma: no cover - optional dependency for dataset export
    TensorDictReplayBuffer = None
    LazyMemmapStorage = None

try:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
except ImportError:  # pragma: no cover - optional dependency for dataset export
    LeRobotDataset = None


def _as_tensordict(data: Any, batch_size: list[int]) -> TensorDictBase:
    if isinstance(data, TensorDictBase):
        return data
    return TensorDict(data, batch_size=batch_size)


def _to_cpu(data: Any) -> Any:
    if isinstance(data, TensorDictBase):
        return data.detach().to("cpu")
    if isinstance(data, dict):
        return {key: _to_cpu(value) for key, value in data.items()}
    if torch.is_tensor(data):
        return data.detach().cpu()
    return data


def _apply_final_obs(
    next_obs: Dict[str, Any],
    final_obs: Dict[str, Any],
    done_mask: torch.Tensor,
) -> Dict[str, Any]:
    updated = {}
    for key, value in next_obs.items():
        if isinstance(value, dict):
            updated[key] = _apply_final_obs(value, final_obs.get(key, {}), done_mask)
        elif torch.is_tensor(value) and key in final_obs:
            value = value.clone()
            value[done_mask] = final_obs[key][done_mask]
            updated[key] = value
        else:
            updated[key] = value
    return updated


def _init_replay_buffer(rb_dir: str, max_size: int) -> "TensorDictReplayBuffer":
    if TensorDictReplayBuffer is None or LazyMemmapStorage is None:
        raise ImportError(
            "torchrl is required for --save_torchrl_rb. Install torchrl to enable it."
        )
    os.makedirs(rb_dir, exist_ok=True)
    storage = LazyMemmapStorage(max_size=max_size, scratch_dir=rb_dir)
    return TensorDictReplayBuffer(storage=storage)


def _write_rb_metadata(rb_dir: str, metadata: Dict[str, Any]) -> str:
    metadata_path = os.path.join(rb_dir, "metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)
    return metadata_path


def _index_data(data: Any, env_id: int) -> Any:
    if isinstance(data, TensorDictBase):
        return data[env_id]
    if isinstance(data, dict):
        return {key: _index_data(value, env_id) for key, value in data.items()}
    if torch.is_tensor(data):
        return data[env_id]
    return data


def _normalize_array(value: Any) -> np.ndarray:
    if torch.is_tensor(value):
        value = value.detach().cpu().numpy()
    else:
        value = np.asarray(value)
    if value.shape == ():
        value = value.reshape(1)
    return value


def _flatten_nested(prefix: str, data: Any, out: Dict[str, Any]) -> None:
    if isinstance(data, TensorDictBase):
        items = data.items()
    elif isinstance(data, dict):
        items = data.items()
    else:
        out[prefix] = data
        return

    for key, value in items:
        if isinstance(key, tuple):
            key_str = ".".join(str(part) for part in key)
        else:
            key_str = str(key)
        next_prefix = f"{prefix}.{key_str}" if prefix else key_str
        _flatten_nested(next_prefix, value, out)


def _is_image_feature(name: str, value: np.ndarray) -> bool:
    if value.ndim != 3:
        return False
    if "image" in name or "rgb" in name or "camera" in name:
        return True
    return value.shape[0] in (1, 3, 4) or value.shape[-1] in (1, 3, 4)


def _infer_lerobot_features(sample_frame: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    features: Dict[str, Dict[str, Any]] = {}
    for key, value in sample_frame.items():
        if key == "task":
            continue
        value_np = _normalize_array(value)
        if _is_image_feature(key, value_np):
            features[key] = {"dtype": "image", "shape": value_np.shape, "names": None}
        else:
            features[key] = {
                "dtype": np.dtype(value_np.dtype).name,
                "shape": value_np.shape,
                "names": None,
            }
    return features


def _stringify_key(key: Any) -> str:
    if isinstance(key, tuple):
        return ".".join(str(part) for part in key)
    return str(key)


def _collect_field_info(data: Any, prefix: str = "") -> Dict[str, Any]:
    info: Dict[str, Any] = {}
    if isinstance(data, TensorDictBase):
        items = data.items()
    elif isinstance(data, dict):
        items = data.items()
    else:
        key = prefix or "value"
        if torch.is_tensor(data):
            info[key] = {"shape": tuple(data.shape), "dtype": str(data.dtype)}
        elif isinstance(data, np.ndarray):
            info[key] = {"shape": data.shape, "dtype": str(data.dtype)}
        else:
            info[key] = {"type": type(data).__name__}
        return info

    for key, value in items:
        key_str = _stringify_key(key)
        next_prefix = f"{prefix}.{key_str}" if prefix else key_str
        if isinstance(value, (TensorDictBase, dict)):
            info.update(_collect_field_info(value, next_prefix))
        else:
            if torch.is_tensor(value):
                info[next_prefix] = {
                    "shape": tuple(value.shape),
                    "dtype": str(value.dtype),
                }
            elif isinstance(value, np.ndarray):
                info[next_prefix] = {
                    "shape": value.shape,
                    "dtype": str(value.dtype),
                }
            else:
                info[next_prefix] = {"type": type(value).__name__}
    return info


class LeRobotExporter:
    def __init__(
        self,
        output_dir: str,
        repo_id: str,
        task_name: str,
        fps: int,
        num_envs: int,
        obs_sample: Any,
        action_sample: Any,
        reward_sample: Any,
        done_sample: Any,
        reference_sample: Any,
    ) -> None:
        if LeRobotDataset is None:
            raise ImportError(
                "lerobot is required for --lerobot_dir. Install lerobot to enable it."
            )
        if os.path.exists(output_dir):
            raise FileExistsError(
                f"LeRobot output directory already exists: {output_dir}"
            )
        self.task_name = task_name
        self.num_envs = num_envs
        sample_frame = self._build_frame(
            obs_sample,
            action_sample,
            reward_sample,
            done_sample,
            reference_sample,
            env_id=0,
        )
        features = _infer_lerobot_features(sample_frame)
        self.dataset = LeRobotDataset.create(
            repo_id=repo_id,
            fps=fps,
            features=features,
            root=output_dir,
            use_videos=False,
        )
        self.active_buffers = [
            self.dataset.create_episode_buffer(episode_index=env_id)
            for env_id in range(num_envs)
        ]
        self.next_episode_index = num_envs
        self.completed_buffers: list[dict] = []

    def _build_frame(
        self,
        obs: Any,
        action: Any,
        reward: Any,
        done: Any,
        reference: Any,
        env_id: int,
    ) -> Dict[str, Any]:
        frame: Dict[str, Any] = {"task": self.task_name}
        obs_env = _index_data(obs, env_id)
        ref_env = _index_data(reference, env_id)
        obs_flat: Dict[str, Any] = {}
        ref_flat: Dict[str, Any] = {}
        _flatten_nested("observation", obs_env, obs_flat)
        _flatten_nested("reference", ref_env, ref_flat)
        for key, value in {**obs_flat, **ref_flat}.items():
            frame[key] = _normalize_array(value)
        frame["action"] = _normalize_array(_index_data(action, env_id))
        frame["next.reward"] = _normalize_array(_index_data(reward, env_id))
        frame["next.done"] = _normalize_array(_index_data(done, env_id))
        return frame

    def add_step(
        self,
        obs: Any,
        action: Any,
        reward: Any,
        done: Any,
        reference: Any,
    ) -> None:
        done_cpu = done.detach().cpu() if torch.is_tensor(done) else done
        for env_id in range(self.num_envs):
            frame = self._build_frame(obs, action, reward, done, reference, env_id)
            self.dataset.episode_buffer = self.active_buffers[env_id]
            self.dataset.add_frame(frame)
            self.active_buffers[env_id] = self.dataset.episode_buffer
            if bool(done_cpu[env_id]):
                self.completed_buffers.append(self.active_buffers[env_id])
                self.active_buffers[env_id] = self.dataset.create_episode_buffer(
                    episode_index=self.next_episode_index
                )
                self.next_episode_index += 1

    def finalize(self) -> None:
        for buffer in self.active_buffers:
            if buffer["size"] > 0:
                self.completed_buffers.append(buffer)
        self.completed_buffers.sort(key=lambda buf: buf["episode_index"])
        for buffer in self.completed_buffers:
            self.dataset.save_episode(episode_data=buffer)
        self.dataset.stop_image_writer()


class ReplayExportTester:
    @staticmethod
    def inspect_torchrl_rb_buffer(
        rb: "TensorDictReplayBuffer", sample_size: int = 4
    ) -> Dict[str, Any]:
        sample = rb.sample(sample_size)
        return {
            "size": len(rb),
            "sample_fields": _collect_field_info(sample),
        }

    @staticmethod
    def inspect_torchrl_rb(rb_dir: str, sample_size: int = 4) -> Dict[str, Any]:
        if TensorDictReplayBuffer is None or LazyMemmapStorage is None:
            raise ImportError("torchrl is required to inspect the replay buffer.")
        if hasattr(TensorDictReplayBuffer, "load"):
            rb = TensorDictReplayBuffer.load(rb_dir)
            return ReplayExportTester.inspect_torchrl_rb_buffer(rb, sample_size)
        storage = LazyMemmapStorage(max_size=1)
        if not hasattr(storage, "load"):
            raise RuntimeError("TorchRL storage does not support load; update torchrl.")
        storage.load(rb_dir)
        rb = TensorDictReplayBuffer(storage=storage)
        return ReplayExportTester.inspect_torchrl_rb_buffer(rb, sample_size)

    @staticmethod
    def inspect_lerobot_dataset(lerobot_dir: str, sample_index: int = 0) -> Dict[str, Any]:
        if LeRobotDataset is None:
            raise ImportError("lerobot is required to inspect the dataset.")
        dataset = LeRobotDataset(repo_id="local", root=lerobot_dir)
        sample = dataset[sample_index]
        info = {}
        if hasattr(dataset, "meta") and hasattr(dataset.meta, "info"):
            info = dataset.meta.info
        return {
            "num_frames": len(dataset),
            "info": info,
            "sample_fields": _collect_field_info(sample),
        }


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg, agent_cfg):  # noqa: ARG001
    """Replay reference data."""
    if args_cli.num_envs is not None:
        env_cfg.scene.num_envs = args_cli.num_envs
    if args_cli.seed is not None:
        env_cfg.seed = args_cli.seed
    env_cfg.sim.device = (
        args_cli.device if args_cli.device is not None else env_cfg.sim.device
    )

    # force reference replay
    env_cfg.replay_reference = True
    env_cfg.replay_only = True

    task_name = args_cli.task.split(":")[-1]
    log_root_path = os.path.abspath(os.path.join("logs", "reference_replay", task_name))
    print(f"[INFO] Logging replay in directory: {log_root_path}")
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = os.path.join(log_root_path, log_dir)
    os.makedirs(log_dir, exist_ok=True)
    env_cfg.log_dir = log_dir

    # create isaac environment
    env = gym.make(
        args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None
    )

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "replay"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during reference replay.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # reset environment
    obs, _ = env.reset()

    dt = env.unwrapped.step_dt
    max_steps = args_cli.video_length if args_cli.video else args_cli.max_steps
    num_envs = env.unwrapped.num_envs

    rb = None
    rb_dir = None
    if args_cli.save_torchrl_rb:
        if max_steps is None:
            raise ValueError("--save_torchrl_rb requires a finite --max_steps.")
        rb_dir = (
            args_cli.torchrl_rb_dir
            if args_cli.torchrl_rb_dir is not None
            else os.path.join(log_dir, "torchrl_rb")
        )
        rb = _init_replay_buffer(rb_dir, max_size=max_steps * num_envs)
        _write_rb_metadata(
            rb_dir,
            {
                "task": args_cli.task,
                "num_envs": num_envs,
                "max_steps": max_steps,
                "device": str(env.unwrapped.device),
                "log_dir": log_dir,
            },
        )

    # prepare a zero action with the right shape
    action = torch.as_tensor(env.action_space.sample(), device=env.unwrapped.device)
    action = torch.zeros_like(action)
    reference = env.unwrapped.get_reference_data()
    lerobot_exporter = None
    lerobot_dir = args_cli.lerobot_dir
    if args_cli.save_lerobot and lerobot_dir is None:
        lerobot_dir = os.path.join(log_dir, "lerobot")
    if lerobot_dir is not None:
        fps = int(round(1.0 / dt)) if dt > 0 else 1
        reward_sample = torch.zeros(
            (num_envs,), device=env.unwrapped.device, dtype=torch.float32
        )
        done_sample = torch.zeros(
            (num_envs,), device=env.unwrapped.device, dtype=torch.bool
        )
        lerobot_exporter = LeRobotExporter(
            output_dir=lerobot_dir,
            repo_id=task_name.replace(":", "_"),
            task_name=task_name,
            fps=fps,
            num_envs=num_envs,
            obs_sample=obs,
            action_sample=action,
            reward_sample=reward_sample,
            done_sample=done_sample,
            reference_sample=reference,
        )

    timestep = 0
    while simulation_app.is_running():
        start_time = time.time()
        next_obs, reward, terminated, truncated, extras = env.step(action)
        next_reference = env.unwrapped.get_reference_data()
        if rb is not None:
            done = terminated | truncated
            final_obs = extras.get("final_obs_buf")
            if final_obs is not None:
                next_obs = _apply_final_obs(next_obs, final_obs, done)

            obs_td = _as_tensordict(_to_cpu(obs), batch_size=[num_envs])
            ref_td = _as_tensordict(_to_cpu(reference), batch_size=[num_envs])
            next_obs_td = _as_tensordict(_to_cpu(next_obs), batch_size=[num_envs])
            next_ref_td = _as_tensordict(_to_cpu(next_reference), batch_size=[num_envs])

            transition = TensorDict({}, batch_size=[num_envs])
            transition.set("observation", obs_td)
            transition.set("reference", ref_td)
            transition.set("action", _to_cpu(action))
            transition.set("reward", _to_cpu(reward))
            transition.set("terminated", _to_cpu(terminated))
            transition.set("truncated", _to_cpu(truncated))
            transition.set("done", _to_cpu(done))
            transition.set(("next", "observation"), next_obs_td)
            transition.set(("next", "reference"), next_ref_td)
            transition.set(("next", "reward"), _to_cpu(reward))
            transition.set(("next", "done"), _to_cpu(done))
            rb.extend(transition)

        if lerobot_exporter is not None:
            done = terminated | truncated
            lerobot_exporter.add_step(obs, action, reward, done, reference)

        obs = next_obs
        reference = next_reference
        timestep += 1
        if max_steps is not None and timestep >= max_steps:
            break

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    env.close()

    if rb is not None and rb_dir is not None:
        if hasattr(rb, "dump"):
            rb.dump(rb_dir)
        try:
            rb_shapes = ReplayExportTester.inspect_torchrl_rb_buffer(rb)
            print("[INFO] TorchRL replay buffer inspection:")
            print(json.dumps(rb_shapes, indent=2, sort_keys=True, default=str))
        except Exception as exc:
            print(f"[WARN] Failed to inspect TorchRL replay buffer: {exc}")
    if lerobot_exporter is not None:
        lerobot_exporter.finalize()
        if lerobot_dir is not None:
            try:
                lerobot_shapes = ReplayExportTester.inspect_lerobot_dataset(lerobot_dir)
                print("[INFO] LeRobot dataset inspection:")
                print(json.dumps(lerobot_shapes, indent=2, sort_keys=True, default=str))
            except Exception as exc:
                print(f"[WARN] Failed to inspect LeRobot dataset: {exc}")


if __name__ == "__main__":
    main()
    simulation_app.close()
