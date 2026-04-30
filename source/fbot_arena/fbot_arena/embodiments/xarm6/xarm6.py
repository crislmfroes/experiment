from isaaclab.utils import configclass
from isaaclab_arena.embodiments.embodiment_base import EmbodimentBase
from isaaclab.assets import ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.sensors import FrameTransformerCfg, TiledCameraCfg
from isaaclab.managers import ActionTermCfg, ObservationTermCfg, ObservationGroupCfg, SceneEntityCfg, EventTermCfg
from isaaclab.envs import mdp, ManagerBasedRLMimicEnv
from isaaclab.envs.mdp.actions.rmpflow_actions_cfg import RMPFlowActionCfg, RmpFlowControllerCfg
import isaaclab.sim as sim_utils
import isaaclab.utils.math as PoseUtils
from isaaclab_arena.assets.register import register_asset
from isaaclab_arena.embodiments.common.mimic_utils import get_rigid_and_articulated_object_poses
import torch
from typing import Sequence
from .mdp import rel_ee_pos, rel_ee_quat, gripper_opening, all_poses_w
import fbot_arena

@register_asset
class Xarm6Embodiment(EmbodimentBase):

    name = "xarm6"

    def __init__(self, enable_cameras = False, initial_pose = None, relative_action=False, joint_pos_action=False, goal_object=None):
        super().__init__(enable_cameras, initial_pose)
        self.scene_config = Xarm6SceneCfg()
        self.action_config = Xarm6ActionsCfg()
        if joint_pos_action == False:
            self.action_config.arm_action.controller.use_relative_mode = relative_action
        else:
            self.action_config.joint_pos_action = mdp.JointPositionActionCfg(
                asset_name="robot",
                joint_names=[".*"]
            )
        self.observation_config = Xarm6ObservationsCfg()
        self.event_config = Xarm6EventsCfg()
        self.mimic_env = Xarm6MimicEnv
        if enable_cameras == False:
            self.scene_config.camera_top = None
            self.observation_config.camera_obs = None
            self.observation_config.policy.top_camera_rgb = None
        else:
            self.observation_config.policy.body_poses = None
            self.observation_config.policy.concatenate_terms = False
        if goal_object != None:
            self.observation_config.policy.object_pos = ObservationTermCfg(
                func=mdp.root_pos_w,
                params=dict(
                    asset_cfg=SceneEntityCfg(name=goal_object.name)
                )
            )
            self.observation_config.policy.object_quat = ObservationTermCfg(
                func=mdp.root_quat_w,
                params=dict(
                    asset_cfg=SceneEntityCfg(name=goal_object.name),
                    make_quat_unique=False,
                )
            )


@configclass
class Xarm6SceneCfg:
    robot: ArticulationCfg = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{fbot_arena.__path__[0]}/assets/robots/xarm6/xarm6/xarm6.usd",
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                fix_root_link=True,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=0
            ),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                #disable_gravity=True,
                max_depenetration_velocity=5.0
            ),
        ),
        actuators=dict(
            arm=ImplicitActuatorCfg(
                joint_names_expr=["joint.*"],
                stiffness=6000.0,
                damping=80.0,
                effort_limit_sim=1000.0,
                velocity_limit_sim=3.0
            ),
            gripper=ImplicitActuatorCfg(
                joint_names_expr=["drive_joint"],  # "right_outer_knuckle_joint" is its mimic joint
                #effort_limit_sim=500.0,
                effort_limit_sim=100.0,
                velocity_limit_sim=10.0,
                stiffness=17.0,
                damping=1.0,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            joint_pos=dict(
                joint4=3.1416,
                joint5=1.5708,
                joint6=3.1416
            )
        )
    )

    ee_frame: FrameTransformerCfg = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Robot/link_base",
        debug_vis=False,
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Robot/xarm_gripper_base_link",
                name="hand"
            ),
        ]
    )

    camera_top: TiledCameraCfg = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/link_base/top_camera",
        offset=TiledCameraCfg.OffsetCfg(
            pos=(0.0, 0.5, 0.5),
            rot=(0.924, 0.0, 0.383, 0.0),
            convention="world"
        ),
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0, focus_distance=400.0, horizontal_aperture=80.0, clipping_range=(0.001, 100.0)
        ),
        width=256,
        height=256,
        data_types=["rgb", "depth"]
    )

    camera_wrist: TiledCameraCfg = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/xarm_gripper_base_link/wrist_camera",
        offset=TiledCameraCfg.OffsetCfg(
            pos=(0.05, 0.0, 0.0),
            rot=(0.0, 0.383, 0.0, 0.924),
            #rot=(0.707, 0.0, -0.707, 0.0),
            convention="world"
        ),
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0, focus_distance=400.0, horizontal_aperture=80.0, clipping_range=(0.001, 100.0)
        ),
        width=256,
        height=256,
        data_types=["rgb", "depth"]
    )

@configclass
class Xarm6ActionsCfg:
    arm_action: ActionTermCfg = mdp.DifferentialInverseKinematicsActionCfg(
        asset_name="robot",
        joint_names=["joint.*"],
        body_name="xarm_gripper_base_link",
        controller=mdp.DifferentialIKControllerCfg(
            command_type="pose",
            use_relative_mode=True,
            ik_method="dls"
        )
    )

    hand_action: ActionTermCfg = mdp.BinaryJointPositionActionCfg(
        asset_name="robot",
        joint_names=["drive_joint",],
        open_command_expr=dict(
            drive_joint=0.0,
            #left_inner_knuckle_joint=0.0,
            #left_finger_joint=0.0,
            #right_outer_knuckle_joint=0.0,
            #right_inner_knuckle_joint=0.0,
            #right_finger_joint=0.0
        ),
        close_command_expr=dict(
            drive_joint=48.0,
            #left_inner_knuckle_joint=48.0,
            #left_finger_joint=48.0,
            #right_outer_knuckle_joint=48.0,
            #right_inner_knuckle_joint=48.0,
            #right_finger_joint=48.0
        )
    )

@configclass
class Xarm6ObservationsCfg:
    @configclass
    class PolicyCfg(ObservationGroupCfg):
        eef_pos = ObservationTermCfg(
            func=rel_ee_pos,
            params=dict()
        )
        eef_quat = ObservationTermCfg(
            func=rel_ee_quat,
            params=dict()
        )
        gripper_pos = ObservationTermCfg(
            func=gripper_opening,
            params=dict()
        )
        body_poses = ObservationTermCfg(
            func=all_poses_w
        )
        joint_pos = ObservationTermCfg(
            func=mdp.joint_pos
        )
        joint_vel = ObservationTermCfg(
            func=mdp.joint_vel
        )
        last_action = ObservationTermCfg(
            func=mdp.last_action
        )

        top_camera_rgb = ObservationTermCfg(
            func=mdp.image,
            params=dict(
                normalize=False,
                sensor_cfg=SceneEntityCfg(name="camera_top")
            )
        )

        wrist_camera_rgb = ObservationTermCfg(
            func=mdp.image,
            params=dict(
                normalize=False,
                sensor_cfg=SceneEntityCfg(name="camera_wrist")
            )
        )

        '''wrist_camera_depth = ObservationTermCfg(
            func=mdp.image,
            params=dict(
                normalize=False,
                sensor_cfg=SceneEntityCfg(name="camera_wrist"),
                data_type="depth"
            )
        )'''

    @configclass
    class ImageCfg(ObservationGroupCfg):
        top_camera_rgb = ObservationTermCfg(
            func=mdp.image,
            params=dict(
                normalize=False,
                sensor_cfg=SceneEntityCfg(name="camera_top")
            )
        )

    @configclass
    class CriticCfg(ObservationGroupCfg):
        body_poses = ObservationTermCfg(
            func=all_poses_w
        )

    policy = PolicyCfg(concatenate_terms=False)
    camera_obs = ImageCfg(concatenate_terms=True)
    #critic = CriticCfg(concatenate_terms=True)

@configclass
class Xarm6EventsCfg:
    reset_scene_to_initial_state = EventTermCfg(
        func=mdp.reset_scene_to_default,
        mode="reset"
    )

class Xarm6MimicEnv(ManagerBasedRLMimicEnv):
    def get_robot_eef_pose(self, eef_name: str, env_ids: Sequence[int] | None = None) -> torch.Tensor:
        """
        Get current robot end effector pose. Should be the same frame as used by the robot end-effector controller.

        Args:
            eef_name: Name of the end effector.
            env_ids: Environment indices to get the pose for. If None, all envs are considered.

        Returns:
            A torch.Tensor eef pose matrix. Shape is (len(env_ids), 4, 4)
        """

        # Retrieve end effector pose from the observation buffer
        #eef_pos = self.obs_buf["policy"][f"{eef_name}_eef_pos"][env_ids]
        #eef_quat = self.obs_buf["policy"][f"{eef_name}_eef_quat"][env_ids]
        eef_pos = rel_ee_pos(env=self)[env_ids]
        eef_quat = rel_ee_quat(env=self)[env_ids]
        # Quaternion format is w,x,y,z
        return PoseUtils.make_pose(eef_pos, PoseUtils.matrix_from_quat(eef_quat))

    def target_eef_pose_to_action(
        self,
        target_eef_pose_dict: dict,
        gripper_action_dict: dict,
        action_noise_dict: dict | None = None,
        env_id: int = 0,
    ) -> torch.Tensor:
        """
        Takes a target pose and gripper action for the end effector controller and returns an action
        (usually a normalized delta pose action) to try and achieve that target pose.
        Noise is added to the target pose action if specified.

        Args:
            target_eef_pose_dict: Dictionary of 4x4 target eef pose for each end-effector.
            gripper_action_dict: Dictionary of gripper actions for each end-effector.
            noise: Noise to add to the action. If None, no noise is added.
            env_id: Environment index to get the action for.

        Returns:
            An action torch.Tensor that's compatible with env.step().
        """
        if self.single_action_space.shape[0] == 7 or self.single_action_space.shape[0] == 8:
            eef_names = ["robot"]
        else:
            eef_names = ["left", "right"]
        eef_actions = []

        for eef_name in eef_names:
            if self.single_action_space.shape[0] == 14 or self.single_action_space.shape[0] == 7:
                # target position and rotation
                target_eef_pose = target_eef_pose_dict[eef_name]
                target_pos, target_rot = PoseUtils.unmake_pose(target_eef_pose)

                # current position and rotation
                curr_pose = self.get_robot_eef_pose(eef_name, env_ids=[env_id])[0]
                curr_pos, curr_rot = PoseUtils.unmake_pose(curr_pose)

                # normalized delta position action
                delta_position = target_pos - curr_pos

                # normalized delta rotation action
                delta_rot_mat = target_rot.matmul(curr_rot.transpose(-1, -2))
                delta_quat = PoseUtils.quat_from_matrix(delta_rot_mat)
                delta_rotation = PoseUtils.axis_angle_from_quat(delta_quat)

                # get gripper action for single eef
                gripper_action = gripper_action_dict[eef_name]

                # add noise to action
                pose_action = torch.cat([delta_position, delta_rotation], dim=0)
                if action_noise_dict is not None:
                    noise = action_noise_dict[eef_name] * torch.randn_like(pose_action)
                    pose_action += noise
                    pose_action = torch.clamp(pose_action, -1.0, 1.0)
                eef_actions.append(torch.cat([pose_action, gripper_action], dim=0))
            
            elif self.single_action_space.shape[0] == 16 or self.single_action_space.shape[0] == 8:
                # target position and rotation
                target_eef_pose = target_eef_pose_dict[eef_name]
                target_pos, target_rot = PoseUtils.unmake_pose(target_eef_pose)

                # normalized delta position action
                position = target_pos

                quat = PoseUtils.quat_from_matrix(target_rot)

                # get gripper action for single eef
                gripper_action = gripper_action_dict[eef_name]

                # add noise to action
                pose_action = torch.cat([position, quat], dim=0)
                if action_noise_dict is not None:
                    noise = action_noise_dict[eef_name] * torch.randn_like(pose_action)
                    pose_action += noise
                    pose_action = torch.clamp(pose_action, -1.0, 1.0)
                eef_actions.append(torch.cat([pose_action, gripper_action], dim=0))
        return torch.cat(eef_actions, dim=0)

    def action_to_target_eef_pose(self, action: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Converts action (compatible with env.step) to a target pose for the end effector controller.
        Inverse of @target_eef_pose_to_action. Usually used to infer a sequence of target controller poses
        from a demonstration trajectory using the recorded actions.

        Args:
            action: Environment action. Shape is (num_envs, action_dim)

        Returns:
            A dictionary of eef pose torch.Tensor that @action corresponds to
        """
        if self.single_action_space.shape[0] == 7 or self.single_action_space.shape[0] == 8:
            eef_names = ["robot"]
        else:
            eef_names = ["left", "right"]
        pose_dict = dict()
        for eef_index in range(len(eef_names)):
            if self.single_action_space.shape[0] == 14 or self.single_action_space.shape[0] == 7:

                eef_name = eef_names[eef_index]

                delta_position = action[:, (eef_index*7):(eef_index*7)+3]
                delta_rotation = action[:, (eef_index*7)+3:(eef_index*7)+6]

                # current position and rotation
                curr_pose = self.get_robot_eef_pose(eef_name, env_ids=None)
                curr_pos, curr_rot = PoseUtils.unmake_pose(curr_pose)

                # get pose target
                target_pos = curr_pos + delta_position

                # Convert delta_rotation to axis angle form
                delta_rotation_angle = torch.linalg.norm(delta_rotation, dim=-1, keepdim=True)
                delta_rotation_axis = delta_rotation / delta_rotation_angle

                # Handle invalid division for the case when delta_rotation_angle is close to zero
                is_close_to_zero_angle = torch.isclose(delta_rotation_angle, torch.zeros_like(delta_rotation_angle)).squeeze(1)
                delta_rotation_axis[is_close_to_zero_angle] = torch.zeros_like(delta_rotation_axis)[is_close_to_zero_angle]

                delta_quat = PoseUtils.quat_from_angle_axis(delta_rotation_angle.squeeze(1), delta_rotation_axis).squeeze(0)
                delta_rot_mat = PoseUtils.matrix_from_quat(delta_quat)
                target_rot = torch.matmul(delta_rot_mat, curr_rot)

                target_poses = PoseUtils.make_pose(target_pos, target_rot).clone()

                pose_dict.update({eef_name: target_poses})
            if self.single_action_space.shape[0] == 16 or self.single_action_space.shape[0] == 8:
                eef_name = eef_names[eef_index]

                abs_position = action[:, (eef_index*8):(eef_index*8)+3]
                abs_rotation = action[:, (eef_index*8)+3:(eef_index*8)+4]
                target_poses = PoseUtils.make_pose(pos=abs_position, rot=PoseUtils.matrix_from_quat(abs_rotation))
                pose_dict.update({eef_name: target_poses})
        return pose_dict

    def actions_to_gripper_actions(self, actions: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Extracts the gripper actuation part from a sequence of env actions (compatible with env.step).

        Args:
            actions: environment actions. The shape is (num_envs, num steps in a demo, action_dim).

        Returns:
            A dictionary of torch.Tensor gripper actions. Key to each dict is an eef_name.
        """
        if self.single_action_space.shape[0] == 7:
            return dict(
                robot=actions[:, [6]]
            )
        if self.single_action_space.shape[0] == 8:
            return dict(
                robot=actions[:, [7]]
            )
        if self.single_action_space.shape[0] == 16:
            # last dimension is gripper action
            return dict(
                left=actions[:, [7]],
                right=actions[:, [15]]
            )    
        # last dimension is gripper action
        return dict(
            left=actions[:, [6]],
            right=actions[:, [13]]
        )
    
    def get_object_poses(self, env_ids = None):
        """
        Gets the pose of each object(rigid and articulated) in the current scene.
        Args:
            env_ids: Environment indices to get the pose for. If None, all envs are considered.
        Returns:
            A dictionary that maps object names to object pose matrix (4x4 torch.Tensor)
        """
        if env_ids is None:
            env_ids = slice(None)

        state = self.scene.get_state(is_relative=True)

        object_pose_matrix = get_rigid_and_articulated_object_poses(state, env_ids)
        for k in object_pose_matrix.keys():
            #object_pose_matrix[k][..., 2, 3] -= object_pose_matrix['robot'][..., 2, 3]
            #object_pose_matrix[k][..., 0, 3] -= object_pose_matrix['robot'][..., 0, 3]
            object_pose_matrix[k] = object_pose_matrix['robot'].inverse() @ object_pose_matrix[k]
        return object_pose_matrix