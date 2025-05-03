import builtins
import torch
from collections.abc import Sequence
from typing import Any, Optional
import numpy as np
import math

import isaacsim.core.utils.torch as torch_utils
import omni.log
from isaacsim.core.simulation_manager import SimulationManager

from isaaclab.managers import (
    ActionManager,
    EventManager,
    ObservationManager,
    RecorderManager,
    CommandManager,
    CurriculumManager,
    RewardManager,
    TerminationManager,
)
from isaaclab.scene import InteractiveScene
from isaaclab.sim import SimulationContext
from isaaclab.ui.widgets import ManagerLiveVisualizer
from isaaclab.utils.timer import Timer

from tensordict import TensorDictBase, TensorDict
from tensordict.nn import TensorDictModule
from torchrl.data import Composite, Unbounded, Bounded
from torchrl.envs import EnvBase

from isaaclab.envs.common import VecEnvObs
from isaaclab.envs.manager_based_env_cfg import ManagerBasedEnvCfg
from isaaclab.envs.ui import ViewportCameraController
from isaaclab.envs.manager_based_rl_env_cfg import ManagerBasedRLEnvCfg
from isaaclab.envs.common import VecEnvStepReturn


class IsaacLabManagerBasedEnv(EnvBase):
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 30,
    }
    batch_locked: bool = True

    def __init__(
        self, cfg: ManagerBasedRLEnvCfg, render_mode: str | None = None, **kwargs
    ):
        """Initialize the environment.

        Args:
            cfg: The configuration object for the environment.

        Raises:
            RuntimeError: If a simulation context already exists. The environment must always create one
                since it configures the simulation context and controls the simulation.
        """
        # -- counter for curriculum
        self.common_step_counter = 0
        # check that the config is valid
        cfg.validate()  # type: ignore
        # store inputs to class
        self.cfg = cfg
        # initialize internal variables
        self._is_closed = False

        # set the seed for the environment
        if self.cfg.seed is not None:
            self.cfg.seed = self._set_seed(self.cfg.seed)
        else:
            omni.log.warn(
                "Seed not set for the environment. The environment creation may not be deterministic."
            )

        # create a simulation context to control the simulator
        if SimulationContext.instance() is None:
            # the type-annotation is required to avoid a type-checking error
            # since it gets confused with Isaac Sim's SimulationContext class
            self.sim: SimulationContext = SimulationContext(self.cfg.sim)
        else:
            # simulation context should only be created before the environment
            # when in extension mode
            if not builtins.ISAAC_LAUNCHED_FROM_TERMINAL:  # type: ignore
                raise RuntimeError(
                    "Simulation context already exists. Cannot create a new one."
                )
            self.sim: SimulationContext = SimulationContext.instance()

        # make sure torch is running on the correct device
        if "cuda" in self.device:
            torch.cuda.set_device(self.device)

        # print useful information
        print("[INFO]: Base environment:")
        print(f"\tEnvironment device    : {self.device}")
        print(f"\tEnvironment seed      : {self.cfg.seed}")
        print(f"\tPhysics step-size     : {self.physics_dt}")
        print(
            f"\tRendering step-size   : {self.physics_dt * self.cfg.sim.render_interval}"
        )
        print(f"\tEnvironment step-size : {self.step_dt}")

        if self.cfg.sim.render_interval < self.cfg.decimation:
            msg = (
                f"The render interval ({self.cfg.sim.render_interval}) is smaller than the decimation "
                f"({self.cfg.decimation}). Multiple render calls will happen for each environment step. "
                "If this is not intended, set the render interval to be equal to the decimation."
            )
            omni.log.warn(msg)

        # counter for simulation steps
        self._sim_step_counter = 0

        # generate scene
        with Timer("[INFO]: Time taken for scene creation", "scene_creation"):
            self.scene = InteractiveScene(self.cfg.scene)
        print("[INFO]: Scene manager: ", self.scene)

        # set up camera viewport controller
        # viewport is not available in other rendering modes so the function will throw a warning
        # FIXME: This needs to be fixed in the future when we unify the UI functionalities even for
        # non-rendering modes.
        if self.sim.render_mode >= self.sim.RenderMode.PARTIAL_RENDERING:
            self.viewport_camera_controller = ViewportCameraController(
                self, self.cfg.viewer  # type: ignore
            )
        else:
            self.viewport_camera_controller = None

        # create event manager
        # note: this is needed here (rather than after simulation play) to allow USD-related randomization events
        #   that must happen before the simulation starts. Example: randomizing mesh scale
        self.event_manager = EventManager(self.cfg.events, self)  # type: ignore
        print("[INFO] Event Manager: ", self.event_manager)

        # apply USD-related randomization events
        if "prestartup" in self.event_manager.available_modes:
            self.event_manager.apply(mode="prestartup")

        # play the simulator to activate physics handles
        # note: this activates the physics simulation view that exposes TensorAPIs
        # note: when started in extension mode, first call sim.reset_async() and then initialize the managers
        if builtins.ISAAC_LAUNCHED_FROM_TERMINAL is False:  # type: ignore
            print(
                "[INFO]: Starting the simulation. This may take a few seconds. Please wait..."
            )
            with Timer("[INFO]: Time taken for simulation start", "simulation_start"):
                self.sim.reset()
                # update scene to pre populate data buffers for assets and sensors.
                # this is needed for the observation manager to get valid tensors for initialization.
                # this shouldn't cause an issue since later on, users do a reset over all the environments so the lazy buffers would be reset.
                self.scene.update(dt=self.physics_dt)
            # add timeline event to load managers
            self.load_managers()

        # extend UI elements
        # we need to do this here after all the managers are initialized
        # this is because they dictate the sensors and commands right now
        if self.sim.has_gui() and self.cfg.ui_window_class_type is not None:
            # setup live visualizers
            self.setup_manager_visualizers()
            self._window = self.cfg.ui_window_class_type(self, window_name="IsaacLab")
        else:
            # if no window, then we don't need to store the window
            self._window = None

        # allocate dictionary to store metrics
        self.extras = {}

        # initialize observation buffers
        self.obs_buf = {}

        # store the render mode
        self.render_mode = render_mode

        # initialize data and constants
        # -- init buffers
        self.episode_length_buf = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.long
        )
        # -- set the framerate of the gym video recorder wrapper so that the playback speed of the produced video matches the simulation
        self.metadata["render_fps"] = 1 / self.step_dt
        # -- starting leg
        self.starting_leg = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.long
        )
        print("[INFO]: Completed setting up the environment...")

        # make the specs
        self._make_specs()

        # initialize the EnvBase
        super().__init__(
            device=self.device,
            batch_size=torch.Size([self.num_envs]),
            allow_done_after_reset=False,
            spec_locked=True,
        )

    def __del__(self):
        """Cleanup for the environment."""
        self.close()

    """
    Properties.
    """

    @property
    def unwrapped(self):
        return self

    @property
    def num_envs(self) -> int:
        """The number of instances of the environment that are running."""
        return self.scene.num_envs

    @property
    def physics_dt(self) -> float:
        """The physics time-step (in s).

        This is the lowest time-decimation at which the simulation is happening.
        """
        return self.cfg.sim.dt

    @property
    def step_dt(self) -> float:
        """The environment stepping time-step (in s).

        This is the time-step at which the environment steps forward.
        """
        return self.cfg.sim.dt * self.cfg.decimation

    @property
    def device(self):
        """The device on which the environment is running."""
        return self.sim.device

    @property
    def phase_dt(self) -> float:
        """Phase time interval in seconds."""
        return 0.64

    @property
    def max_episode_length_s(self) -> float:
        """Maximum episode length in seconds."""
        return self.cfg.episode_length_s

    @property
    def max_episode_length(self) -> int:
        """Maximum episode length in environment steps."""
        return math.ceil(self.max_episode_length_s / self.step_dt)

    """
    Operations - Setup.
    """

    def load_managers(self):
        # note: this order is important since observation manager needs to know the command and action managers
        # and the reward manager needs to know the termination manager
        # -- command manager
        self.command_manager: CommandManager = CommandManager(self.cfg.commands, self)  # type: ignore
        print("[INFO] Command Manager: ", self.command_manager)

        # -- recorder manager
        self.recorder_manager = RecorderManager(self.cfg.recorders, self)  # type: ignore
        print("[INFO] Recorder Manager: ", self.recorder_manager)
        # -- action manager
        self.action_manager = ActionManager(self.cfg.actions, self)  # type: ignore
        print("[INFO] Action Manager: ", self.action_manager)
        # -- observation manager
        self.observation_manager = ObservationManager(self.cfg.observations, self)  # type: ignore
        print("[INFO] Observation Manager:", self.observation_manager)

        # prepare the managers
        # -- termination manager
        self.termination_manager = TerminationManager(self.cfg.terminations, self)  # type: ignore
        print("[INFO] Termination Manager: ", self.termination_manager)
        # -- reward manager
        self.reward_manager = RewardManager(self.cfg.rewards, self)  # type: ignore
        print("[INFO] Reward Manager: ", self.reward_manager)
        # -- curriculum manager
        self.curriculum_manager = CurriculumManager(self.cfg.curriculum, self)  # type: ignore
        print("[INFO] Curriculum Manager: ", self.curriculum_manager)

        # perform events at the start of the simulation
        if "startup" in self.event_manager.available_modes:
            self.event_manager.apply(mode="startup")

    def setup_manager_visualizers(self):
        """Creates live visualizers for manager terms."""

        self.manager_visualizers = {
            "action_manager": ManagerLiveVisualizer(manager=self.action_manager),
            "observation_manager": ManagerLiveVisualizer(
                manager=self.observation_manager
            ),
            "command_manager": ManagerLiveVisualizer(manager=self.command_manager),
            "termination_manager": ManagerLiveVisualizer(
                manager=self.termination_manager
            ),
            "reward_manager": ManagerLiveVisualizer(manager=self.reward_manager),
            "curriculum_manager": ManagerLiveVisualizer(
                manager=self.curriculum_manager
            ),
        }

    """
    Operations - MDP
    """

    def get_phase(self) -> torch.Tensor:
        """Get the phase of the environment."""

        if not hasattr(self, "episode_length_buf") or self.episode_length_buf is None:
            return torch.zeros(self.num_envs, device=self.device, dtype=torch.float)

        phase = (
            torch.fmod(
                self.episode_length_buf.type(dtype=torch.float) * self.step_dt,
                self.phase_dt,
            )
            / self.phase_dt
        )
        return phase

    def get_starting_leg(self) -> torch.Tensor:
        """Get the starting leg of the environment. 0 for left and 1 for right."""
        if not hasattr(self, "starting_leg") or self.starting_leg is None:
            self.starting_leg = torch.randint(
                0, 2, (self.num_envs,), device=self.device
            )

        return self.starting_leg

    def _make_specs(self):
        # Create observation spec based on observation manager's active terms
        observation_spec = Composite(
            {}, batch_size=torch.Size([self.num_envs]), device=self.device
        )
        for (
            group_name,
            group_term_names,
        ) in self.observation_manager.active_terms.items():
            # extract quantities about the group
            has_concatenated_obs = self.observation_manager.group_obs_concatenate[
                group_name
            ]
            group_dim = self.observation_manager.group_obs_dim[group_name]

            # check if group is concatenated or not
            if has_concatenated_obs:
                # For parallel environments, shape should be (num_envs, *group_dim)
                obs_shape = (self.num_envs,) + (
                    group_dim if isinstance(group_dim, tuple) else (group_dim,)
                )
                observation_spec[group_name] = Unbounded(
                    shape=torch.Size(obs_shape), device=self.device  # type: ignore
                )
            else:
                observation_spec[group_name] = Composite(
                    {
                        term_name: Unbounded(
                            shape=torch.Size(
                                (self.num_envs,)
                                + (
                                    term_dim
                                    if isinstance(term_dim, tuple)
                                    else (term_dim,)
                                )
                            ),
                            device=self.device,
                        )
                        for term_name, term_dim in zip(group_term_names, group_dim)
                    },
                    batch_size=torch.Size([self.num_envs]),
                    device=self.device,
                )

        # do the same for action spec
        action_spec = Composite(
            {}, batch_size=torch.Size([self.num_envs]), device=self.device
        )

        if len(self.action_manager.active_terms) == 1:
            term_dim = self.action_manager.action_term_dim[0]
            action_spec["action"] = Bounded(
                low=-1,
                high=1,
                shape=torch.Size(
                    (self.num_envs,)
                    + (term_dim if isinstance(term_dim, tuple) else (term_dim,))
                ),
                device=self.device,
            )
        else:
            for term_name, term_dim in zip(
                self.action_manager.active_terms,
                self.action_manager.action_term_dim,
            ):
                print("term_name", term_name)
                action_spec[term_name] = Bounded(
                    low=-1,
                    high=1,
                    shape=torch.Size(
                        (self.num_envs,)
                        + (term_dim if isinstance(term_dim, tuple) else (term_dim,))
                    ),
                    device=self.device,
                )

        # Create reward spec (scalar reward)
        reward_spec = Unbounded(
            shape=torch.Size((self.num_envs, 1)),
            device=self.device,
        )

        # For state spec, we'll use the same as observation spec since we don't have a separate state
        state_spec = observation_spec.clone()

        # make full done spec
        full_done_spec = Composite(
            {
                "done": Unbounded(
                    shape=torch.Size((self.num_envs, 1)),
                    device=self.device,
                    dtype=torch.bool,
                ),
                "terminated": Unbounded(
                    shape=torch.Size((self.num_envs, 1)),
                    device=self.device,
                    dtype=torch.bool,
                ),
                "truncated": Unbounded(
                    shape=torch.Size((self.num_envs, 1)),
                    device=self.device,
                    dtype=torch.bool,
                ),
            },
            batch_size=torch.Size([self.num_envs]),
            device=self.device,
        )

        # create info spec
        info_spec = Composite(
            {
                "final_observation": observation_spec.clone(),
            },
            batch_size=torch.Size([self.num_envs]),
            device=self.device,
        )

        # Store the specs
        self.observation_spec = observation_spec
        self.state_spec = state_spec
        self.action_spec = action_spec
        self.reward_spec = reward_spec
        self.full_done_spec = full_done_spec
        self.info_spec = info_spec

    def _step(self, tensordict: TensorDictBase) -> TensorDictBase:  # type: ignore
        """Execute one time-step of the environment's dynamics and reset terminated environments.

        Unlike the :class:`ManagerBasedEnv.step` class, the function performs the following operations:

        1. Process the actions.
        2. Perform physics stepping.
        3. Perform rendering if gui is enabled.
        4. Update the environment counters and compute the rewards and terminations.
        5. Reset the environments that terminated.
        6. Compute the observations.
        7. Return the observations, rewards, resets and extras.

        Args:
            action: The actions to apply on the environment. Shape is (num_envs, action_dim).

        Returns:
            A tuple containing the observations, rewards, resets (terminated and truncated) and extras.
        """
        action: torch.Tensor = tensordict["action"]
        # process actions
        self.action_manager.process_action(action.to(self.device))

        self.recorder_manager.record_pre_step()

        # check if we need to do rendering within the physics loop
        # note: checked here once to avoid multiple checks within the loop
        is_rendering = self.sim.has_gui() or self.sim.has_rtx_sensors()

        # perform physics stepping
        for _ in range(self.cfg.decimation):
            self._sim_step_counter += 1
            # set actions into buffers
            self.action_manager.apply_action()
            # set actions into simulator
            self.scene.write_data_to_sim()
            # simulate
            self.sim.step(render=False)
            # render between steps only if the GUI or an RTX sensor needs it
            # note: we assume the render interval to be the shortest accepted rendering interval.
            #    If a camera needs rendering at a faster frequency, this will lead to unexpected behavior.
            if (
                self._sim_step_counter % self.cfg.sim.render_interval == 0
                and is_rendering
            ):
                self.sim.render()
            # update buffers at sim dt
            self.scene.update(dt=self.physics_dt)

        # post-step:
        # -- update env counters (used for curriculum generation)
        self.episode_length_buf += 1  # step in current episode (per env)
        self.common_step_counter += 1  # total step (common for all envs)
        # -- check terminations
        self.reset_buf = self.termination_manager.compute()
        self.reset_terminated = self.termination_manager.terminated
        self.reset_time_outs = self.termination_manager.time_outs
        # -- reward computation
        self.reward_buf = self.reward_manager.compute(dt=self.step_dt)

        obs_after_step = self.observation_manager.compute()

        if len(self.recorder_manager.active_terms) > 0:
            # update observations for recording if needed
            self.obs_buf = obs_after_step
            self.recorder_manager.record_post_step()

            # Create TensorDict with observations, rewards, and resets
        out = TensorDict(
            {
                "reward": self.reward_buf.unsqueeze(-1),
                "done": self.reset_buf,
                "terminated": self.reset_terminated,
                "truncated": self.reset_time_outs,
            },
            batch_size=[self.num_envs],
            device=self.device,
        )

        # print("out before obs handle", out)
        out = self.add_obs_to_td(obs_after_step, out)
        # print("out after obs handle", out)
        return out

    @staticmethod
    def seed(seed: int = -1) -> int:
        """Set the seed for the environment.

        Args:
            seed: The seed for random generator. Defaults to -1.

        Returns:
            The seed used for random generator.
        """
        # set seed for replicator
        try:
            import omni.replicator.core as rep

            rep.set_global_seed(seed)
        except ModuleNotFoundError:
            pass
        # set seed for torch and other libraries
        return torch_utils.set_seed(seed)

    def _reset(self, tensordict: TensorDictBase | None = None, **kwargs):
        # if tensordict is not None, use it as the initial state
        if tensordict is None:
            # Doing a complete reset
            self.complete_reset(kwargs.get("seed", -1))
            td_out = TensorDict(
                {}, batch_size=torch.Size([self.num_envs]), device=self.device
            )
            td_out = self.add_obs_to_td(self.obs_buf, td_out)
            return td_out

        # -- reset envs that terminated/timed-out and log the episode information
        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)

        if len(reset_env_ids) > 0:
            # record the final observation
            self.final_obs_buf: dict = self.observation_manager.compute()

            # trigger recorder terms for pre-reset calls
            self.recorder_manager.record_pre_reset(reset_env_ids)  # type: ignore

            self._reset_idx(reset_env_ids)  # type: ignore
            # update articulation kinematics
            self.scene.write_data_to_sim()
            self.sim.forward()

            # if sensors are added to the scene, make sure we render to reflect changes in reset
            if self.sim.has_rtx_sensors() and self.cfg.rerender_on_reset:
                self.sim.render()

            # trigger recorder terms for post-reset calls
            self.recorder_manager.record_post_reset(reset_env_ids)  # type: ignore
        else:
            self.final_obs_buf = dict()

        # -- update command
        self.command_manager.compute(dt=self.step_dt)
        # -- step interval events
        if "interval" in self.event_manager.available_modes:
            self.event_manager.apply(mode="interval", dt=self.step_dt)
        # -- compute observations
        # note: done after reset to get the correct observations for reset envs
        self.obs_buf = self.observation_manager.compute()

        td_out = TensorDict(
            {}, batch_size=torch.Size([self.num_envs]), device=self.device
        )
        td_out = self.add_obs_to_td(self.obs_buf, td_out)

        return td_out

    def complete_reset(
        self,
        seed: int | None = None,
        env_ids: Sequence[int] | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[VecEnvObs, dict]:
        """Resets the specified environments and returns observations.

        This function calls the :meth:`_reset_idx` function to reset the specified environments.
        However, certain operations, such as procedural terrain generation, that happened during initialization
        are not repeated.

        Args:
            seed: The seed to use for randomization. Defaults to None, in which case the seed is not set.
            env_ids: The environment ids to reset. Defaults to None, in which case all environments are reset.
            options: Additional information to specify how the environment is reset. Defaults to None.

                Note:
                    This argument is used for compatibility with Gymnasium environment definition.

        Returns:
            A tuple containing the observations and extras.
        """
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, dtype=torch.int64, device=self.device)

        # trigger recorder terms for pre-reset calls
        self.recorder_manager.record_pre_reset(env_ids)

        # set the seed
        if seed is not None:
            self.seed(seed)

        # reset state of scene
        self._reset_idx(env_ids)

        # update articulation kinematics
        self.scene.write_data_to_sim()
        self.sim.forward()
        # if sensors are added to the scene, make sure we render to reflect changes in reset
        if self.sim.has_rtx_sensors() and self.cfg.rerender_on_reset:
            self.sim.render()

        # trigger recorder terms for post-reset calls
        self.recorder_manager.record_post_reset(env_ids)

        # compute observations
        self.obs_buf = self.observation_manager.compute()

        if self.cfg.wait_for_textures and self.sim.has_rtx_sensors():
            while SimulationManager.assets_loading():
                self.sim.render()

        # return observations
        return self.obs_buf, self.extras

    def reset_to(
        self,
        state: dict[str, dict[str, dict[str, torch.Tensor]]],
        env_ids: Sequence[int] | None,
        seed: int | None = None,
        is_relative: bool = False,
    ) -> None:
        """Resets specified environments to known states.

        Note that this is different from reset() function as it resets the environments to specific states

        Args:
            state: The state to reset the specified environments to.
            env_ids: The environment ids to reset. Defaults to None, in which case all environments are reset.
            seed: The seed to use for randomization. Defaults to None, in which case the seed is not set.
            is_relative: If set to True, the state is considered relative to the environment origins. Defaults to False.
        """
        # reset all envs in the scene if env_ids is None
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, dtype=torch.int64, device=self.device)

        # trigger recorder terms for pre-reset calls
        self.recorder_manager.record_pre_reset(env_ids)

        # set the seed
        if seed is not None:
            self.seed(seed)

        self._reset_idx(env_ids)

        # set the state
        self.scene.reset_to(state, env_ids, is_relative=is_relative)

        # update articulation kinematics
        self.sim.forward()

        # if sensors are added to the scene, make sure we render to reflect changes in reset
        if self.sim.has_rtx_sensors() and self.cfg.rerender_on_reset:
            self.sim.render()

        # trigger recorder terms for post-reset calls
        self.recorder_manager.record_post_reset(env_ids)

        # compute observations
        self.obs_buf = self.observation_manager.compute()

        # return observations
        return self.obs_buf, self.extras

    def add_obs_to_td(
        self, obs_buf_post_step: dict, out: TensorDictBase
    ) -> TensorDictBase:
        # handle the observation buffer from obs_buf_post_step
        for group_name, term_names in self.observation_manager.active_terms.items():
            is_concatenated = self.observation_manager.group_obs_concatenate[group_name]
            if is_concatenated:
                out[group_name] = obs_buf_post_step[group_name]
            else:
                out[group_name] = TensorDict(
                    {term_name: obs_buf_post_step[term_name] for term_name in term_names},  # type: ignore
                    batch_size=[self.num_envs],
                    device=self.device,
                )
        return out

    def render(self, recompute: bool = False) -> np.ndarray | None:  # type: ignore
        """Run rendering without stepping through the physics.

        By convention, if mode is:

        - **human**: Render to the current display and return nothing. Usually for human consumption.
        - **rgb_array**: Return an numpy.ndarray with shape (x, y, 3), representing RGB values for an
          x-by-y pixel image, suitable for turning into a video.

        Args:
            recompute: Whether to force a render even if the simulator has already rendered the scene.
                Defaults to False.

        Returns:
            The rendered image as a numpy array if mode is "rgb_array". Otherwise, returns None.

        Raises:
            RuntimeError: If mode is set to "rgb_data" and simulation render mode does not support it.
                In this case, the simulation render mode must be set to ``RenderMode.PARTIAL_RENDERING``
                or ``RenderMode.FULL_RENDERING``.
            NotImplementedError: If an unsupported rendering mode is specified.
        """
        # run a rendering step of the simulator
        # if we have rtx sensors, we do not need to render again sin
        if not self.sim.has_rtx_sensors() and not recompute:
            self.sim.render()
        # decide the rendering mode
        if self.render_mode == "human" or self.render_mode is None:
            return None
        elif self.render_mode == "rgb_array":
            # check that if any render could have happened
            if self.sim.render_mode.value < self.sim.RenderMode.PARTIAL_RENDERING.value:
                raise RuntimeError(
                    f"Cannot render '{self.render_mode}' when the simulation render mode is"
                    f" '{self.sim.render_mode.name}'. Please set the simulation render mode to:"
                    f"'{self.sim.RenderMode.PARTIAL_RENDERING.name}' or '{self.sim.RenderMode.FULL_RENDERING.name}'."
                    " If running headless, make sure --enable_cameras is set."
                )
            # create the annotator if it does not exist
            if not hasattr(self, "_rgb_annotator"):
                import omni.replicator.core as rep

                # create render product
                self._render_product = rep.create.render_product(
                    self.cfg.viewer.cam_prim_path, self.cfg.viewer.resolution
                )
                # create rgb annotator -- used to read data from the render product
                self._rgb_annotator = rep.AnnotatorRegistry.get_annotator(
                    "rgb", device="cpu"
                )
                self._rgb_annotator.attach([self._render_product])  # type: ignore
            # obtain the rgb data
            rgb_data = self._rgb_annotator.get_data()
            # convert to numpy array
            rgb_data = np.frombuffer(rgb_data, dtype=np.uint8).reshape(*rgb_data.shape)
            # return the rgb data
            # note: initially the renerer is warming up and returns empty data
            if rgb_data.size == 0:
                return np.zeros(
                    (self.cfg.viewer.resolution[1], self.cfg.viewer.resolution[0], 3),
                    dtype=np.uint8,
                )
            else:
                return rgb_data[:, :, :3]
        else:
            raise NotImplementedError(
                f"Render mode '{self.render_mode}' is not supported. Please use: {self.metadata['render_modes']}."  # type: ignore
            )

    def close(self):
        if not self._is_closed:
            # destructor is order-sensitive
            del self.command_manager
            del self.reward_manager
            del self.termination_manager
            del self.curriculum_manager
            # destructor is order-sensitive
            del self.viewport_camera_controller
            del self.action_manager
            del self.observation_manager
            del self.event_manager
            del self.recorder_manager
            del self.scene
            # clear callbacks and instance
            self.sim.clear_all_callbacks()
            self.sim.clear_instance()
            # destroy the window
            if self._window is not None:
                self._window = None
            # update closing status
            self._is_closed = True

    def _reset_idx(self, env_ids: Sequence[int]):
        """Reset environments based on specified indices.

        Args:
            env_ids: List of environment ids which must be reset
        """
        # update the curriculum for environments that need a reset
        self.curriculum_manager.compute(env_ids=env_ids)
        # reset the internal buffers of the scene elements
        self.scene.reset(env_ids)
        # apply events such as randomizations for environments that need a reset
        if "reset" in self.event_manager.available_modes:
            env_step_count = self._sim_step_counter // self.cfg.decimation
            self.event_manager.apply(
                mode="reset", env_ids=env_ids, global_env_step_count=env_step_count
            )

        # iterate over all managers and reset them
        # this returns a dictionary of information which is stored in the extras
        # note: This is order-sensitive! Certain things need be reset before others.
        self.extras["log"] = dict()
        # -- observation manager
        info = self.observation_manager.reset(env_ids)
        self.extras["log"].update(info)
        # -- action manager
        info = self.action_manager.reset(env_ids)
        self.extras["log"].update(info)
        # -- rewards manager
        info = self.reward_manager.reset(env_ids)
        self.extras["log"].update(info)
        # -- curriculum manager
        info = self.curriculum_manager.reset(env_ids)
        self.extras["log"].update(info)
        # -- command manager
        info = self.command_manager.reset(env_ids)
        self.extras["log"].update(info)
        # -- event manager
        info = self.event_manager.reset(env_ids)
        self.extras["log"].update(info)
        # -- termination manager
        info = self.termination_manager.reset(env_ids)
        self.extras["log"].update(info)
        # -- recorder manager
        info = self.recorder_manager.reset(env_ids)
        self.extras["log"].update(info)

        # reset the episode length buffer
        self.episode_length_buf[env_ids] = 0

        # # reset the starting leg
        # self.starting_leg[env_ids] = torch.randint(
        #     0, 2, (len(env_ids),), device=self.device
        # )

    def _make_spec(self, td_params):
        pass

    def _set_seed(self, seed: Optional[int]):
        self.seed(seed)
