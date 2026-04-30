import argparse

from isaaclab_arena.examples.example_environments.example_environment_base import ExampleEnvironmentBase
import isaaclab_arena.policy

class Xarm6OpenDrawerEnvironment(ExampleEnvironmentBase):
    name: str = "xarm6_open_drawer"
    
    def get_env(self, args_cli):
        import torch
        import tqdm

        import pinocchio  # noqa: F401
        from isaaclab.app import AppLauncher
        from scene_synthesizer.assets import USDAsset


        import fbot_arena.embodiments
        import fbot_arena.assets
        from isaaclab_arena.assets.object_reference import ObjectReference
        from fbot_arena.arena_tasks.pick_and_place import PickAndPlaceTask
        from fbot_arena.arena_tasks.pick import PickTask
        from fbot_arena.arena_tasks.pour import PourTask
        from fbot_arena.arena_tasks.open_drawer import OpenDrawerTask
        from fbot_arena.policies.oracle.pour import OraclePourPolicy
        from fbot_arena.policies.lerobot.lerobot_policy import LeRobotPolicy
        from fbot_arena.policies.llm.llm_policy import LLMPolicy
        from isaaclab_arena.assets.asset_registry import AssetRegistry
        from isaaclab_arena.cli.isaaclab_arena_cli import get_isaaclab_arena_cli_parser
        from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
        from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
        from isaaclab_arena.scene.scene import Scene
        from isaaclab_arena.utils.pose import Pose
        import isaaclab_arena.teleop_devices
        import fbot_arena.arena_devices.keyboard
        import fbot_arena.arena_devices.keyboard_bimanual
        import fbot_arena.arena_devices.mediapipe_teleop_device
        import fbot_arena.assets.exhaust_pipe
        from fbot_arena.policies.oracle.pick_place_exhaust_pipe import OraclePickPlaceExhaustPipePolicy
        from fbot_arena.policies.oracle.pick import OraclePickPolicy
        from fbot_arena.policies.oracle.open_drawer import OracleOpenDrawerPolicy
        from isaaclab_arena.tasks.dummy_task import DummyTask
        from isaaclab_rl.rsl_rl.rl_cfg import RslRlOnPolicyRunnerCfg, configclass, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg
        import omni.client
        

        teleop_device = self.device_registry.get_device_by_name(args_cli.teleop_device)()

        # Step 1: Initialize and get the assets from the registry

        background = self.asset_registry.get_asset_by_name("kitchen")()
        initial_background_pose: Pose = background.get_initial_pose()
        background.set_initial_pose(
            Pose(position_xyz=(
                initial_background_pose.position_xyz[0] + 0.2,
                initial_background_pose.position_xyz[1],
                initial_background_pose.position_xyz[2]
            ),
            rotation_wxyz=initial_background_pose.rotation_wxyz)
        )
        #counter_top_a = ObjectReference(
        #    name="counter_top_A",
        #    prim_path="{ENV_REGEX_NS}/kitchen/Kitchen_Counter/TRS_Base/TRS_Static",#/container_h20_inst/Container_H20_01
        #    parent_asset=background,
        #)
        
        cabinet = self.asset_registry.get_asset_by_name("sektion_cabinet")(openable_joint_name="drawer_top_joint")

        embodiment = self.asset_registry.get_asset_by_name("xarm6")(enable_cameras=args_cli.enable_cameras, relative_action=(args_cli.action_mode=="relative"), joint_pos_action=(args_cli.action_type=="joint"), goal_object=cabinet)
        
        #sorting_bin.set_initial_pose(
        #    Pose(position_xyz=(0.4, -0.2, 0.3), rotation_wxyz=(0.707, 0.0, 0.0, -0.707))
        #)
        cabinet.set_initial_pose(
            Pose(position_xyz=(-0.5, -1.0, -0.5), rotation_wxyz=(0.707, 0.0, 0.0, 0.707))
        )
        
        embodiment.set_initial_pose(
            #Pose(position_xyz=(0.2, 0.0, -0.2), rotation_wxyz=(1, 0, 0, 0))
            Pose(position_xyz=(-0.5, 0.0, 0.0), rotation_wxyz=(0.707, 0, 0, -0.707))
        )

        # Step 2: Create a scene with the assets
        scene = Scene(assets=[background, cabinet])

        #print(scene_description)
        #exit()


        # Step 3: Create a task
        #task = PickTask(pick_up_object=tomato_soup, background_scene=background, episode_length_s=20.0)
        task = OpenDrawerTask(openable_object=cabinet, openness_threshold=0.5, reset_openness=0.0, episode_length_s=12.0)
        #task = DummyTask()

        # Step 4: Create the IsaacLab Arena environment
        isaaclab_arena_environment = IsaacLabArenaEnvironment(
            name="open_drawer_env",
            embodiment=embodiment,
            scene=scene,
            task=task,
            teleop_device=teleop_device,
        )

        # Step 5: Create oracle policy
        oracle_factory = lambda **kwargs: OracleOpenDrawerPolicy(openable_obj=cabinet, openable_obj_handle_name="door_handle_top", **kwargs)
        isaaclab_arena_environment.oracle_policy_factory = oracle_factory

        lerobot_factory = lambda **kwargs: LeRobotPolicy(**kwargs, task=task.get_prompt(), goal_object=cabinet, vlm_class_prompt=cabinet.name.replace("_", " "))
        isaaclab_arena_environment.lerobot_policy_factory = lerobot_factory

        return isaaclab_arena_environment
    
    @staticmethod
    def add_cli_args(parser):
        parser.add_argument("--teleop_device", type=str, default="keyboard")
        parser.add_argument('--action_mode', choices=["abs", "relative"], default='relative')
        parser.add_argument('--action_type', choices=["joint", "ee"], default='ee')