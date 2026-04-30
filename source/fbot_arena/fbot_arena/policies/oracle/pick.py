from isaaclab_arena.policy.policy_base import PolicyBase
from fbot_arena.embodiments.openarm_bimanual.openarm_bimanual import OpenArmBimanualMimicEnv
from isaaclab_mimic.motion_planners.curobo.curobo_planner import CuroboPlanner
from isaaclab_mimic.motion_planners.curobo.curobo_planner_cfg import CuroboPlannerCfg
import curobo
import isaaclab.utils.math as PoseUtils
import torch
import yaml
import os
import fbot_arena
import random

class OraclePickPolicyStates:
    APPROACH_OBJ = 0
    GRASP_OBJ = 1

class OraclePickPolicy(PolicyBase):
    def __init__(self, pickup_obj, planners: dict[str, CuroboPlanner]=dict()):
        super().__init__()
        self.pickup_obj = pickup_obj
        self.planners = planners
        self.state = OraclePickPolicyStates.APPROACH_OBJ
        self.state_step_counter = 0
        self.load_grasps()
        #self.reset()

    def load_grasps(self):
        grasp_poses = []
        grasp_dir = f"{fbot_arena.__path__[0]}/assets/grasps/{self.pickup_obj.name}"
        grasp_datas = []
        for grasp_file in os.listdir(grasp_dir):
            with open(f"{grasp_dir}/{grasp_file}", 'r') as f:
                grasp_data = yaml.full_load(f)
            grasp_datas.append(grasp_data)
        threshold = torch.mean(torch.as_tensor([grasp_data['grasp_result']['joint_states']['joints/drive_joint'] for grasp_data in grasp_datas])).item()
        for grasp_data in grasp_datas:
            if grasp_data['grasp_result']['joint_states']['joints/drive_joint'] < threshold:
                grasp_pos = torch.as_tensor([grasp_data['grasp_result']['gripper_location']], device='cuda')
                grasp_quat = torch.as_tensor([[grasp_data['grasp_result']['gripper_orientation']['w'], *grasp_data['grasp_result']['gripper_orientation']['xyz']]], device='cuda')
                offset_pos = torch.as_tensor([[0.0, 0.0, -0.05]], device='cuda')
                offset_quat = torch.as_tensor([[1.0, 0.0, 0.0, 0.0]], device='cuda')
                offset_pose = PoseUtils.make_pose(pos=offset_pos, rot=PoseUtils.matrix_from_quat(offset_quat))
                grasp_pose = PoseUtils.make_pose(pos=grasp_pos, rot=PoseUtils.matrix_from_quat(grasp_quat)) @ offset_pose
                grasp_poses.append(grasp_pose)
        self.grasp_poses = grasp_poses

    def sort_grasp_poses(self, initial_pose, pickup_object_pose):
        return sorted(
            self.grasp_poses,
            key=lambda p: 1.0*torch.norm((pickup_object_pose @ p)[0, :3, 3] - initial_pose[0, 0, :3, 3], p=2, dim=-1).item()
        )

    def get_action(self, env: OpenArmBimanualMimicEnv, observation):
        action = torch.zeros(size=env.action_space.shape, device=env.device)
        if self.state == OraclePickPolicyStates.APPROACH_OBJ:
            if not self.planners['left'].has_next_waypoint():
                self.planners['left'].update_world()
                pickup_obj_pose = env.get_object_poses()[self.pickup_obj.name]
                world_model_pickup_object_pose = self.planners['left'].get_object_pose(object_name=self.pickup_obj.name)
                left_eef_pose = env.get_robot_eef_pose(eef_name="robot")
                sorted_grasp_poses = self.sort_grasp_poses(initial_pose=left_eef_pose, pickup_object_pose=pickup_obj_pose)
                #random.shuffle(sorted_grasp_poses)
                for grasp_pose_in_obj_frame in sorted_grasp_poses:
                    grasp_pose = pickup_obj_pose @ grasp_pose_in_obj_frame
                    #grasp_pose[..., 0, 3] = pickup_obj_pose[..., 0, 3] - 0.25 # - 0.2# - 0.05
                    #grasp_pose[..., 1, 3] = pickup_obj_pose[..., 1, 3]
                    #grasp_pose[..., 2, 3] = pickup_obj_pose[..., 2, 3] + 0.1
                    #grasp_pose[..., :3, :3] = PoseUtils.matrix_from_euler(euler_angles=torch.as_tensor([[0.0, torch.pi/2, 0.0]], device=env.device), convention="XYZ")
                    #self.right_target_pose = env.get_robot_eef_pose(eef_name="right")
                    self.left_target_pose = grasp_pose
                    self.left_direction = self.left_target_pose[..., :3, 3] - left_eef_pose[..., :3, 3]/50.0
                    self.left_plan_succeeded = self.planners['left'].update_world_and_plan_motion(target_pose=self.left_target_pose[0], step_size=env.step_dt)
                    print('Plan success: ', self.left_plan_succeeded)
                    #right_plan_succeeded = self.planners['right'].plan_motion(target_pose=self.right_target_pose[0], step_size=env.step_dt)
                    self.reached_next_left_pose = False
                    #self.reached_next_right_pose = False
                    self.next_left_pose = self.planners['left'].get_ee_pose(joint_state=self.planners['left']._get_current_joint_state_for_curobo()).get_matrix()
                    #self.next_right_pose = env.get_robot_eef_pose(eef_name="right").clone()
                    if self.left_plan_succeeded == True:
                        break
                assert self.left_plan_succeeded
            if self.planners['left'].has_next_waypoint() and self.left_plan_succeeded:
                self.next_left_pose = self.planners['left'].get_next_waypoint_ee_pose()
                self.next_left_pose = self.next_left_pose.get_matrix()
                #self.next_left_pose[..., 0, 3] -= -0.3
                #self.next_left_pose[..., 1, 3] -= (0.4-0.15)
                #self.next_left_pose[..., 2, 3] -= 0.2
                #self.next_left_pose[..., 1, 3] -= (0.031+0.09)
                #self.next_left_pose[..., 1, 3] += 0.698
            #if self.planners['right'].has_next_waypoint():
            #    self.next_right_pose = self.planners['right'].get_next_waypoint_ee_pose().get_matrix()
            action[..., :] = env.target_eef_pose_to_action(
                target_eef_pose_dict=dict(
                    robot=self.next_left_pose[0],
                    #right=self.next_right_pose[0]
                ),
                gripper_action_dict=dict(
                    robot=torch.as_tensor([1.0,], device=env.device),
                    #right=torch.as_tensor([1.0,], device=env.device)
                ),
            )    
            left_eef_pose = env.get_robot_eef_pose(eef_name="robot")
            left_quat = PoseUtils.quat_from_matrix(left_eef_pose[..., :3, :3])
            left_pos = left_eef_pose[..., :3, 3].reshape((1, 3))
            left_target_quat = PoseUtils.quat_from_matrix(self.next_left_pose[..., :3, :3])
            left_target_pos = self.next_left_pose[..., :3, 3].reshape((1, 3))

            #right_eef_pose = env.get_robot_eef_pose(eef_name="right")
            #right_quat = PoseUtils.quat_from_matrix(right_eef_pose[..., :3, :3])
            #right_pos = right_eef_pose[..., :3, 3].reshape((1, 3))
            #right_target_quat = PoseUtils.quat_from_matrix(self.next_right_pose[..., :3, :3])
            #right_target_pos = self.next_right_pose[..., :3, 3].reshape((1, 3))
            #print(torch.norm(left_pos[0] - left_target_pos[0], p=2))
            #print(torch.norm(left_quat[0] - left_target_quat[0], p=2))
            if (torch.norm(left_pos[0] - left_target_pos[0], p=2) <= 0.008 and torch.norm(left_quat[0] - left_target_quat[0], p=2) <= 0.01) or True:
                self.reached_next_left_pose = True
            else:
                self.reached_next_left_pose = False

            #if (torch.norm(right_pos[0] - right_target_pos[0], p=2) <= 0.001 and torch.norm(right_quat[0] - right_target_quat[0], p=2) <= 0.01):
            #    self.reached_next_right_pose = True
            
            #print(self.next_left_pose.shape, self.left_target_pose.shape, left_eef_pose.shape)
            #self.next_left_pose[..., :3, 3] += 0.1*(self.left_target_pose[..., 0, :3, 3] - left_eef_pose[..., 0, :3, 3])

            if not self.planners['left'].has_next_waypoint() and self.reached_next_left_pose == True:
                self.state = OraclePickPolicyStates.GRASP_OBJ
                self.state_step_counter = 0
                pass
            
        if self.state == OraclePickPolicyStates.GRASP_OBJ:
            if self.state_step_counter < 20+30:
                offset_pos = torch.as_tensor([[0.0, 0.0, (0.05+0.05)/(20+30)]], device='cuda')
                offset_quat = torch.as_tensor([[1.0, 0.0, 0.0, 0.0]], device='cuda')
                offset_pose = PoseUtils.make_pose(pos=offset_pos, rot=PoseUtils.matrix_from_quat(offset_quat))
                self.left_target_pose = self.left_target_pose @ offset_pose
                action[..., :] = env.target_eef_pose_to_action(
                    target_eef_pose_dict=dict(
                        robot=self.left_target_pose[0],
                        #right=self.right_target_pose[0]
                    ),
                    gripper_action_dict=dict(
                        robot=torch.as_tensor([1.0,], device=env.device),
                        #right=torch.as_tensor([1.0,], device=env.device)
                    ),
                )
            elif self.state_step_counter < 50+30:
                action[..., :] = env.target_eef_pose_to_action(
                    target_eef_pose_dict=dict(
                        robot=self.left_target_pose[0],
                        #right=self.right_target_pose[0]
                    ),
                    gripper_action_dict=dict(
                        robot=torch.as_tensor([-1.0,], device=env.device),
                        #right=torch.as_tensor([1.0,], device=env.device)
                    ),
                )
            else:
                self.left_target_pose[..., 2, 3] += 0.01
                action[..., :] = env.target_eef_pose_to_action(
                    target_eef_pose_dict=dict(
                        robot=self.left_target_pose[0],
                        #right=self.right_target_pose[0]
                    ),
                    gripper_action_dict=dict(
                        robot=torch.as_tensor([-1.0,], device=env.device),
                        #right=torch.as_tensor([1.0,], device=env.device)
                    ),
                )
            if self.state_step_counter >= (50+30+50) and env.get_object_poses()[self.pickup_obj.name][0, 2, 3] <= 0.15:
                self.state_step_counter = 0
                self.state = OraclePickPolicyStates.APPROACH_OBJ
                action[..., :] = env.target_eef_pose_to_action(
                    target_eef_pose_dict=dict(
                        robot=self.left_target_pose[0],
                        #right=self.right_target_pose[0]
                    ),
                    gripper_action_dict=dict(
                        robot=torch.as_tensor([1.0,], device=env.device),
                        #right=torch.as_tensor([1.0,], device=env.device)
                    ),
                )

        #print(self.state)
        self.state_step_counter += 1
        return action
    
    def reset(self, env_ids = None):
        self.state = OraclePickPolicyStates.APPROACH_OBJ
        self.state_step_counter = 0
        for k in self.planners.keys():
            self.planners[k].reset_plan()