import argparse

from isaaclab_arena.examples.example_environments.example_environment_base import ExampleEnvironmentBase

class Xarm6OpenMicrowaveEnvironment(ExampleEnvironmentBase):
    name: str = "xarm6_open_microwave"
    
    def get_env(self, args_cli):
        import torch
        import tqdm

        import pinocchio  # noqa: F401
        from isaaclab.app import AppLauncher

        import fbot_arena.embodiments
        import fbot_arena.assets
        from fbot_arena.arena_tasks.open_door import OpenDoorTask
        from fbot_arena.arena_tasks.pour import PourTask
        from fbot_arena.policies.oracle.pour import OraclePourPolicy
        from fbot_arena.policies.oracle.open_door import OracleOpenDoorPolicy
        from fbot_arena.policies.lerobot.lerobot_policy import LeRobotPolicy

        from isaaclab_arena.assets.asset_registry import AssetRegistry
        from isaaclab_arena.cli.isaaclab_arena_cli import get_isaaclab_arena_cli_parser
        from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
        from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
        from isaaclab_arena.scene.scene import Scene
        from isaaclab_arena.utils.pose import Pose
        import isaaclab_arena.teleop_devices
        import fbot_arena.arena_devices.keyboard

        if args_cli.teleop_device is not None:
            teleop_device = self.device_registry.get_device_by_name(args_cli.teleop_device)()
        else:
            teleop_device = None

        # Step 1: Initialize and get the assets from the registry
        microwave = self.asset_registry.get_asset_by_name("microwave")()


        background = self.asset_registry.get_asset_by_name("kitchen")()
        embodiment = self.asset_registry.get_asset_by_name("xarm6")(enable_cameras=args_cli.enable_cameras, relative_action=(args_cli.action_mode=="relative"), joint_pos_action=(args_cli.action_type=="joint"), goal_object=microwave)

        
        microwave.set_initial_pose(
            Pose(position_xyz=(0.4, 0.0, 0.2), rotation_wxyz=(0.707, 0.0, 0.0, -0.707))
        )
        embodiment.set_initial_pose(
            Pose(position_xyz=(-0.3, 0.0, 0.0), rotation_wxyz=(1, 0, 0, 0))
        )

        # Step 2: Create a scene with the assets
        scene = Scene(assets=[background, microwave])

        # Step 3: Create a task
        task = OpenDoorTask(openable_object=microwave, episode_length_s=10.0, openness_threshold=0.15)

        # Step 4: Create the IsaacLab Arena environment
        isaaclab_arena_environment = IsaacLabArenaEnvironment(
            name="open_microwave_env",
            embodiment=embodiment,
            scene=scene,
            task=task,
            teleop_device=teleop_device,
        )

        oracle_factory = lambda **kwargs: OracleOpenDoorPolicy(openable_obj=microwave, openable_obj_handle_name="handle", pull_side="left", door_radius=0.45, **kwargs)
        isaaclab_arena_environment.oracle_policy_factory = oracle_factory

        lerobot_factory = lambda **kwargs: LeRobotPolicy(**kwargs, task=task.get_prompt(), goal_object=microwave, vlm_class_prompt=microwave.name.replace("_", " "))
        isaaclab_arena_environment.lerobot_policy_factory = lerobot_factory

        return isaaclab_arena_environment
    
    @staticmethod
    def add_cli_args(parser):
        parser.add_argument("--teleop_device", type=str, default="keyboard")
        parser.add_argument('--action_mode', choices=["abs", "relative"], default='relative')
        parser.add_argument('--action_type', choices=["joint", "ee"], default='ee')