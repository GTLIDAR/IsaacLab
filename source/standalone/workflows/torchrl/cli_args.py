# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Copyright (c) 2024 Boston Dynamics AI Institute LLC. All rights reserved.

from __future__ import annotations

import argparse
import random
from typing import TYPE_CHECKING, Dict, Any, Type, Callable, List, Optional
from torch import nn
from dataclasses import dataclass

if TYPE_CHECKING:
    from omni.isaac.lab_tasks.utils.wrappers.torchrl import OnPolicyPPORunnerCfg

def add_torchrl_args(parser: argparse.ArgumentParser):
    """Add TorchRL arguments to the parser.
    Adds the following fields to argparse:
        - "--experiment_name" : Name of the experiment folder where logs will be stored (default: None).
        - "--run_name" : Run name suffix to the log directory (default: None).
        - "--resume" : Whether to resume from a checkpoint (default: None).
        - "--load_run" : Name of the run folder to resume from (default: None).
        - "--checkpoint" : Checkpoint file to resume from (default: None).
        - "--logger" : Logger module to use (default: None).
        - "--log_project_name" : Name of the logging project when using wandb or neptune (default: None).
    Args:
        parser: The parser to add the arguments to.
    """
    # create a new argument group
    arg_group = parser.add_argument_group("torchrl", description="Arguments for RSL-RL agent.")
    # -- experiment arguments
    arg_group.add_argument(
        "--experiment_name",
        type=str,
        default=None,
        help="Name of the experiment folder where logs will be stored.",
    )
    arg_group.add_argument(
        "--run_name",
        type=str,
        default=None,
        help="Run name suffix to the log directory.",
    )
    # -- load arguments
    arg_group.add_argument(
        "--resume",
        type=bool,
        default=None,
        help="Whether to resume from a checkpoint.",
    )
    arg_group.add_argument(
        "--load_run",
        type=str,
        default=None,
        help="Name of the run folder to resume from.",
    )
    arg_group.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Checkpoint file to resume from.",
    )
    # -- logger arguments
    arg_group.add_argument(
        "--logger",
        type=str,
        default=None,
        choices={"wandb", "tensorboard", "neptune"},
        help="Logger module to use.",
    )
    arg_group.add_argument(
        "--log_project_name",
        type=str,
        default=None,
        help="Name of the logging project when using wandb or neptune.",
    )


def parse_torchrl_cfg(task_name: str, args_cli: argparse.Namespace) -> OnPolicyPPORunnerCfg:
    """Parse configuration for RSL-RL agent based on inputs.
    Args:
        task_name: The name of the environment.
        args_cli: The command line arguments.
    Returns:
        The parsed configuration for RSL-RL agent based on inputs.
    """
    from omni.isaac.lab_tasks.utils.parse_cfg import load_cfg_from_registry

    # load the default configuration
    torchrl_cfg: OnPolicyPPORunnerCfg = load_cfg_from_registry(task_name, "torchrl_cfg_entry_point")

    # override the default configuration with CLI arguments
    torchrl_cfg.device = "cpu" if args_cli.cpu else f"cuda:{args_cli.physics_gpu}"

    # override the default configuration with CLI arguments
    if args_cli.seed is not None:
        torchrl_cfg.seed = args_cli.seed
    if args_cli.resume is not None:
        torchrl_cfg.resume = args_cli.resume
    if args_cli.load_run is not None:
        torchrl_cfg.load_run = args_cli.load_run
    if args_cli.checkpoint is not None:
        torchrl_cfg.load_checkpoint = args_cli.checkpoint
    if args_cli.run_name is not None:
        torchrl_cfg.run_name = args_cli.run_name
    if args_cli.logger is not None:
        torchrl_cfg.logger = args_cli.logger
    # set the project name for wandb and neptune
    if torchrl_cfg.logger == "wandb" and args_cli.log_project_name:
        torchrl_cfg.wandb_project = args_cli.log_project_name

    return torchrl_cfg


def update_torchrl_cfg(agent_cfg: OnPolicyPPORunnerCfg, args_cli: argparse.Namespace):
    """Update configuration for torchrl agent based on inputs.
    Args:
        agent_cfg: The configuration for torchrl agent.
        args_cli: The command line arguments.
    Returns:
        The updated configuration for torchrl agent based on inputs.
    """
    # override the default configuration with CLI arguments
    if hasattr(args_cli, "seed") and args_cli.seed is not None:
        # randomly sample a seed if seed = -1
        if args_cli.seed == -1:
            args_cli.seed = random.randint(0, 10000)
        agent_cfg.seed = args_cli.seed
    if args_cli.resume is not None:
        agent_cfg.resume = args_cli.resume
    if args_cli.load_run is not None:
        agent_cfg.load_run = args_cli.load_run
    if args_cli.checkpoint is not None:
        agent_cfg.load_checkpoint = args_cli.checkpoint
    if args_cli.run_name is not None:
        agent_cfg.run_name = args_cli.run_name
    if args_cli.logger is not None:
        agent_cfg.logger = args_cli.logger
    # set the project name for wandb and neptune
    if agent_cfg.logger in {"wandb"} and args_cli.log_project_name:
        agent_cfg.wandb_project = args_cli.log_project_name

@dataclass
class ModuleMapping:
    """Maps module names to their configuration classes."""
    cfg_class: Type
    network_keys: List[str]

class ConfigurationError(Exception):
    """Custom exception for configuration-related errors."""
    pass

class NNFromYAML:
    """Builds nn models from YAML configurations."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Args:
            config: Dictionary containing the model configuration.
        
        Raises:
            ValueError: If config is None or empty.
        """
        if not config:
            raise ValueError("Config cannot be None or empty")
        self.config = config
        self.SUPPORTED_MODELS = {"Sequential": self.load_sequential}

    def build_model_from_config(self) -> nn.Module:
        try:
            model_type = self.config.get("type")
            if not model_type:
                raise ConfigurationError("Model type not specified in configuration")
                
            builder = self.SUPPORTED_MODELS.get(model_type)
            if not builder:
                raise ConfigurationError(f"Unsupported model type: {model_type}")
                
            return builder()
        except Exception as e:
            raise ConfigurationError(f"Error building model: {str(e)}")

    def load_sequential(self):
        try:
            layers = []
            layer_configs = self.config.get("layers", [])
            
            if not layer_configs:
                raise ConfigurationError("No layers specified for Sequential model")
                
            for layer_config in layer_configs:
                if not isinstance(layer_config, dict):
                    raise ConfigurationError(f"Invalid layer configuration: {layer_config}")
                    
                layer_type = layer_config.get("type")
                if not layer_type:
                    raise ConfigurationError("Layer type not specified")
                    
                try:
                    layer_class = getattr(nn, layer_type)
                except AttributeError:
                    raise ConfigurationError(f"Unknown layer type: {layer_type}")
                
                # Extract parameters excluding the type
                parameters = {k: v for k, v in layer_config.items() if k != "type"}
                layers.append(layer_class(**parameters))
                
            return nn.Sequential(*layers)
        except Exception as e:
            raise ConfigurationError(f"Error building Sequential model: {str(e)}")

class DigitNN(nn.Module):
    def __init__(self, config: Dict[str,Any]):
        super().__init__()
        try:
            model_builder = NNFromYAML(config)
            self.model = model_builder.build_model_from_config()
        except Exception as e:
            raise ConfigurationError(f"Failed to initialize DigitNN: {str(e)}")

    def forward(self,x = None):
        if x is None:
            return self
        return self.model(x)

def update_torchrl_cfg_with_yaml(yaml_cfg: Dict[str, Any], args_cli: argparse.Namespace):
    """Update configuration for torchrl agent based on inputs.
    Args:
        agent_cfg: The configuration dictionary from yaml.
        args_cli: The command line arguments.
    Returns:
        The updated configuration for torchrl agent based on inputs.
    """
    from omni.isaac.lab_tasks.utils.wrappers.torchrl.torchrl_ppo_runner_cfg import (
        ClipPPOLossCfg, CollectorCfg, ProbabilisticActorCfg, ValueOperatorCfg, OnPolicyPPORunnerCfg
    )
    if not yaml_cfg:
        raise ValueError("yaml_cfg cannot be None or empty")

    MODULE_MAPPINGS = {
        'actor_module': ModuleMapping(
            ProbabilisticActorCfg,
            ['actor_network']
        ),
        'critic_module': ModuleMapping(
            ValueOperatorCfg,
            ['critic_network']
        ),
        'collector_module': ModuleMapping(
            CollectorCfg,
            ['actor_network']
        ),
        'loss_module': ModuleMapping(
            ClipPPOLossCfg,
            ['actor_network', 'value_network']
        ),
        'on_policy_runner': ModuleMapping(
            OnPolicyPPORunnerCfg,
            ['loss_module', 'collector_module']
        )
    }

    try:
        agent_cfg = {}

        for network_type in ['actor_network', 'critic_network']:
            if network_type in yaml_cfg:
                agent_cfg[network_type] = DigitNN(yaml_cfg[network_type])

        for module_name, mapping in MODULE_MAPPINGS.items():
            if module_name in yaml_cfg:
                module_cfg = {}
                for key, value in yaml_cfg[module_name].items():
                    if key in mapping.network_keys:
                        if value not in agent_cfg:
                            raise ConfigurationError(
                                f"Referenced network '{value}' not found in configuration"
                            )
                        module_cfg[key] = agent_cfg[value]
                    else:
                        module_cfg[key] = value
                agent_cfg[module_name] = mapping.cfg_class(**module_cfg)

        if 'on_policy_runner' not in agent_cfg:
            raise ConfigurationError("Missing required 'on_policy_runner' configuration")

        return agent_cfg.popitem()[1]

    except Exception as e:
        raise ConfigurationError(f"Error updating torchrl configuration: {str(e)}")


