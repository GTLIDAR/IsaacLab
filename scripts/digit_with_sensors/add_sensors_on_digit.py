"""
This script demonstrates how to add and simulate on-board sensors for a robot.

We add the following sensors on the humanoid robot, Digit:

.. code-block:: bash

    # Usage
    ./isaaclab.sh -p scripts/digit_with_sensors/add_sensors_on_digit.py --enable_cameras

"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Tutorial on adding sensors on a robot.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""
import matplotlib.pyplot as plt
import numpy as np
import os
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors import CameraCfg, RayCasterCfg, patterns
from isaaclab.utils import configclass
from sensor_utils.buffer import RGB8FrameBuffer

##
# Pre-defined configs
##
from isaaclab_assets.robots.digit import DIGITV3_CFG  # isort: skip


@configclass
class SensorsSceneCfg(InteractiveSceneCfg):
    """Design the scene with sensors on the robot."""

    # ground plane
    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )

    # robot
    robot: ArticulationCfg = DIGITV3_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot") # type: ignore

    front_cube = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Cube",
        spawn=sim_utils.CuboidCfg(
            size=(0.3, 0.3, 0.3), 
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.7, 0.3))
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(1.5, 0.0, 0.15),  
            rot=(1.0, 0.0, 0.0, 0.0), 
        ),
    )
    # ----------------------- upper_velodyne_vlp16  -----------------------
    upper_velodyne_vlp16_depth = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        update_period=0.02,
        offset=RayCasterCfg.OffsetCfg(
            pos=(0.0, 0.0, 0.4),  
            rot=(1.0, 0.0, 0.0, 0.0)  
        ),
        attach_yaw_only=True,
        pattern_cfg=patterns.LidarPatternCfg(
            channels=16,  
            vertical_fov_range=(-15.0, 15.0),  
            horizontal_fov_range=(0.0, 360.0),  
            horizontal_res=0.2, 
        ),
        debug_vis=True,
        max_distance=200.0,
        mesh_prim_paths=["/World/defaultGroundPlane"],
    )
    # ----------------------- forward-tis-dfm27up  -----------------------
    forward_tis_dfm27up_color = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base/forward_tis_dfm27up_color",
        update_period=0.1,
        height=480,
        width=640,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=6.0,  
            focus_distance=400.0,
            horizontal_aperture=20.955,  
            clipping_range=(0.1, 1.0e5),
        ),
        offset=CameraCfg.OffsetCfg(
            pos=(0.3, 0.0, 0.15),  
            rot=(0.5, -0.5, 0.5, -0.5),  
            convention="ros",
        ),
    )
    # ----------------------- forward-chest-realsense-d435  -----------------------
    forward_chest_realsense_d435 = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base/forward_chest_realsense_d435",
        update_period=0.1,
        height=480,
        width=640,
        data_types=["rgb", "depth"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=6.0,  
            focus_distance=400.0,
            horizontal_aperture=20.955,  
            clipping_range=(0.1, 1.0e5),
        ),
        offset=CameraCfg.OffsetCfg(
            pos=(0.0, 0.0, 0.0), 
            rot=(0.5, -0.5, 0.5, -0.5),  
            convention="ros",
        ),
    )
    # ----------------------- downward Pelvis RealSense D430 -----------------------
    downward_pelvis_realsense_d430 = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base/downward_pelvis_realsense_d430",
        update_period=0.1,
        height=480,
        width=640,
        data_types=["depth"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=6.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.1, 1.0e5),
        ),
        offset=CameraCfg.OffsetCfg(
            pos=(0.0, 0.0, 0.1), 
            rot=(0.5, -0.5, 0.5, -0.5),  
        ),
    )
    # ----------------------- forward-pelvis-realsense-d430  -----------------------
    forward_pelvis_realsense_d430 = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base/forward_pelvis_realsense_d430",
        update_period=0.1,
        height=480,
        width=640,
        data_types=["depth"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=6.0,  
            focus_distance=400.0,
            horizontal_aperture=20.955,  
            clipping_range=(0.1, 1.0e5),
        ),
        offset=CameraCfg.OffsetCfg(
            pos=(0.0, 0.0, -0.1),  
            rot=(0.707, 0.0, 0.0, 0.707),  
            convention="ros",
        ),
    ) 

# ----------------------- backward-pelvis-realsense-d430  -----------------------
    backward_pelvis_realsense_d430 = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base/backward_pelvis_realsense_d430",
        update_period=0.1,
        height=480,
        width=640,
        data_types=["depth"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=6.0,  
            focus_distance=400.0,
            horizontal_aperture=20.955,  
            clipping_range=(0.1, 1.0e5),
        ),
        offset=CameraCfg.OffsetCfg(
            pos=(-0.2, 0.0, 0.1), 
            rot=(0.0, 0.707, 0.707, 0.0),
            convention="ros",
        ),
    )  

def save_images(
    images: list[torch.Tensor] | torch.Tensor,  
    cmap: str | None = None,
    nrow: int = 1,
    subtitles: list[str] | None = None,
    title: str | None = None,
    filename: str | None = None,
):
    """Save either a single image or a grid of images with optional title/subtitles.
    
    Args:
        images: Single image (H,W,C) or list of images. Each image should be (H,W,C).
        cmap: Colormap for display. None uses default colormap.
        nrow: Number of rows in grid layout. Ignored for single image.
        subtitles: List of subtitles for each image. Length should match number of images.
        title: Main title for the entire figure.
        filename: Path to save the image. If None, figure won't be saved.
    """
    # Convert single image to list for unified processing
    if isinstance(images, torch.Tensor):
        images = [images]
        nrow = 1  # Force single-image mode

    n_images = len(images)
    if n_images == 1:
        # Single image mode - simpler display without subplots
        plt.figure(figsize=(2, 2))
        img = images[0].detach().cpu().numpy()
        plt.imshow(img, cmap=cmap)
        plt.axis("off")
        if title:
            plt.title(title)  # Use title as main title for single image
        if filename:
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            plt.savefig(filename)
        plt.close()
    else:
        # Multi-image mode - grid layout
        ncol = int(np.ceil(n_images / nrow))
        fig, axes = plt.subplots(nrow, ncol, figsize=(ncol * 2, nrow * 2))
        axes = np.array(axes).ravel()  # Flatten axes array for easy iteration
        
        # Plot each image in the grid
        for idx, (img, ax) in enumerate(zip(images, axes)):
            img = img.detach().cpu().numpy()
            ax.imshow(img, cmap=cmap)
            ax.axis("off")
            if subtitles:
                ax.set_title(subtitles[idx])
        
        # Remove empty subplots if grid size > number of images
        for ax in axes[n_images:]:
            fig.delaxes(ax)
            
        if title:
            plt.suptitle(title)  # Add main title for the grid
            
        plt.tight_layout()
        if filename:
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            plt.savefig(filename)
        plt.close()

def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    """Run the simulator."""
    # Define simulation stepping

    sim_dt = sim.get_physics_dt()
    sim_time = 0.0
    count = 0
    print(scene.cfg.num_envs)
    rgb_buffer = RGB8FrameBuffer(
        max_cpu_frames=2000,
        auto_flush_interval=50,
        output_dir="scripts/digit_with_sensors/output/rgb",
        n_envs = scene.cfg.num_envs,
    )

    output_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "output")
    os.makedirs(output_dir, exist_ok=True)

    # Simulate physics
    while simulation_app.is_running():
        # Reset
        if count % 500 == 0:
            # reset counter
            count = 0
            # reset the scene entities
            # root state
            # we offset the root state by the origin since the states are written in simulation world frame
            # if this is not done, then the robots will be spawned at the (0, 0, 0) of the simulation world
            root_state = scene["robot"].data.default_root_state.clone()
            root_state[:, :3] += scene.env_origins
            scene["robot"].write_root_pose_to_sim(root_state[:, :7])
            scene["robot"].write_root_velocity_to_sim(root_state[:, 7:])
            # set joint positions with some noise
            joint_pos, joint_vel = (
                scene["robot"].data.default_joint_pos.clone(),
                scene["robot"].data.default_joint_vel.clone(),
            )

            scene["robot"].write_joint_state_to_sim(joint_pos, joint_vel)

            # clear internal buffers
            scene.reset()
            print("[INFO]: Resetting robot state...")

        targets = scene["robot"].data.default_joint_pos
        
        # -- apply action to the robot
        scene["robot"].set_joint_position_target(targets)
        # -- write data to sim
        scene.write_data_to_sim()
        # perform step
        sim.step()
        # update sim-time
        sim_time += sim_dt
        count += 1
        # update buffers
        scene.update(sim_dt)

        # print information from the sensors
        '''
        print("-------------------------------")
        print(scene["upper_velodyne_vlp16_depth"])
        '''
        print("-------------------------------")
        print(scene["forward_tis_dfm27up_color"])
        print("Received shape of rgb   image: ", scene["forward_tis_dfm27up_color"].data.output["rgb"].shape)

        rgb_data = scene["forward_tis_dfm27up_color"].data.output["rgb"]
        rgb_buffer.add_frame(rgb_data) 
        
        env_id = 0
        if count % 10 == 0:
            rgb_images = [scene["forward_tis_dfm27up_color"].data.output["rgb"][env_id, ..., :3]]
            save_images(
                rgb_images,
                title="RGB Image",
                filename=os.path.join(output_dir, "rgb", f"{count:04d}.jpg"),
            )
        
def main():
    """Main function."""

    # Initialize the simulation context
    sim_cfg = sim_utils.SimulationCfg(dt=0.005, device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    # Set main camera
    sim.set_camera_view(eye=(3.5, 3.5, 3.5), target=(0.0, 0.0, 0.0))
    # design scene
    scene_cfg = SensorsSceneCfg(num_envs=args_cli.num_envs, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    # Play the simulator
    sim.reset()
    # Now we are ready!
    print("[INFO]: Setup complete...")
    # Run the simulator
    run_simulator(sim, scene)


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()


