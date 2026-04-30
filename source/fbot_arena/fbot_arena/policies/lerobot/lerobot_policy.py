from isaaclab_arena.policy.policy_base import PolicyBase
from isaaclab_arena.policy.zero_action_policy import ZeroActionPolicy
from fbot_arena.embodiments.xarm6.xarm6 import Xarm6MimicEnv
from lerobot.policies.factory import make_policy, make_policy_config, make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy, SmolVLAConfig
from lerobot.policies.groot.modeling_groot import GrootPolicy
from lerobot.policies.xvla.modeling_xvla import XVLAPolicy, XVLAConfig
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
import isaaclab.utils.math as PoseUtils
from isaaclab.sensors import TiledCamera, TiledCameraCfg
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image
from isaaclab.sensors.camera.utils import create_pointcloud_from_depth
import random

def project_2d_to_3d_pytorch(pixels, depths, K)->torch.Tensor:
    """
    Projeta pontos 2D para 3D usando PyTorch.
    
    Args:
        pixels: Tensor (N, 2) contendo coordenadas [u, v]
        depths: Tensor (N, 1) ou (N,) com a profundidade de cada pixel
        K: Matriz intrínseca (3, 3)
    """
    # 1. Converter pixels para coordenadas homogêneas [u, v, 1]
    n = pixels.shape[0]
    ones = torch.ones((n, 1), device=pixels.device, dtype=pixels.dtype)
    pixels_homo = torch.cat([pixels, ones], dim=1) # Shape (N, 3)
    
    # 2. Inverter a matriz intrínseca
    K_inv = torch.inverse(K) # Shape (3, 3)
    
    # 3. Aplicar K_inv aos pixels (normalização)
    # (N, 3) @ (3, 3) -> transpomos K_inv para multiplicar à direita
    points_normalized = pixels_homo @ K_inv.T # Shape (N, 3)


    
    # 4. Escalar pela profundidade
    # Garante que depths tenha shape (N, 1) para o broadcast
    if depths.dim() == 1:
        depths = depths.unsqueeze(1)
        
    points_3d = points_normalized * depths
    
    return points_3d

class LeRobotPolicy(PolicyBase):
    def __init__(self, repo_id, task: str, policy_type="smolvla", use_pose_obs=True, use_contact_point=False, oracle_policy=None, goal_object=None, vlm_class_prompt=None) -> None:
        super().__init__()
        #self.policy_config = make_policy_config(policy_type=policy_type)
        #self.policy_config.pretrained_path = repo_id
        #assert policy_type == "smolvla"
        if policy_type == "smolvla":
            self.policy = SmolVLAPolicy.from_pretrained(pretrained_name_or_path=repo_id)
            #self.policy.config.n_action_steps = 10
            #self.policy.config.num_steps = 100
        elif policy_type == "groot":
            self.policy = GrootPolicy.from_pretrained(pretrained_name_or_path=repo_id).bfloat16()
            #self.policy.config.chunk_size = 5
            #print(self.policy.config.max_steps)
        elif policy_type == "xvla":
            self.policy = XVLAPolicy.from_pretrained(pretrained_name_or_path=repo_id)
            #self.policy.config.n_action_steps = 1
            print(self.policy.config.action_mode)
            exit()
        elif policy_type == "diffusion":
            self.policy = DiffusionPolicy.from_pretrained(pretrained_name_or_path=repo_id)
        else:
            raise ValueError()
        torch.cuda.empty_cache()
        self.use_pose_obs = use_pose_obs
        #self.policy.config.n_action_steps = 1
        self.policy_config = self.policy.config
        self.pre_processor, self.post_processor = make_pre_post_processors(self.policy_config, pretrained_path=repo_id)
        #self.policy = make_policy(self.policy_config)
        self.task = task
        self.oracle_policy = oracle_policy
        self.use_contact_point = use_contact_point
        self.contact_point = None
        self.goal_object = goal_object
        '''self.vlm = AutoModelForCausalLM.from_pretrained(
            "vikhyatk/moondream2",
            revision="2025-06-21",
            trust_remote_code=True,
            device_map={"": "cuda"}  # ...or 'mps', on Apple Silicon
        )
        self.vlm_class_prompt = vlm_class_prompt'''
        self.idle_policy = ZeroActionPolicy()

    def get_action(self, env: Xarm6MimicEnv, observation):
        frame = {
                        "observation.images.wrist": observation["policy"]["wrist_camera_rgb"].float().reshape((env.num_envs, 3, 256, 256)).to(self.policy.config.device),
                        "observation.images.top": observation["policy"]["top_camera_rgb"].float().reshape((env.num_envs, 3, 256, 256)).to(self.policy.config.device),
                        "observation.state": torch.cat([
                            observation["policy"]["eef_pos"],
                            observation["policy"]["eef_quat"],
                            observation["policy"]["gripper_pos"],
                        ], dim=-1),
                        #"observation.state": observation["policy"]["joint_pos"],
                        #"observation.state": observation["policy"].to(self.policy.config.device),
                        "task": [self.task,]*env.num_envs,
                    }
        if self.use_pose_obs == True:
            frame["observation.state"] = torch.cat([
                frame["observation.state"],
                observation["policy"]["object_pos"],
                observation["policy"]["object_quat"],
            ], dim=-1)
        if self.use_contact_point == True:
            if self.contact_point == None:
                '''vlm_image = Image.fromarray(observation["policy"]["wrist_camera_rgb"].reshape((256, 256, 3)).detach().cpu().numpy().astype(np.uint8))
                print(vlm_image)
                print(self.vlm_class_prompt)
                contact_points_2d = self.vlm.point(vlm_image, self.vlm_class_prompt)["points"]
                if len(contact_points_2d) == 0:
                    return self.idle_policy.get_action(env=env, observation=observation)
                contact_point_2d = contact_points_2d[0]
                top_camera: TiledCamera = env.scene["camera_top"]
                x = contact_point_2d["x"]*top_camera.cfg.width
                y = contact_point_2d["y"]*top_camera.cfg.height
                depth_image = top_camera.data.output["depth"]
                depth = depth_image[..., int(y), int(x), 0].item()
                point_y_3d = depth*np.cos(x*np.radians(54.4) - np.radians(54.4/2))
                point_z_3d = depth*np.cos(y*np.radians(54.4) - np.radians(54.4/2))
                point_x_3d = np.sqrt(depth**2 + point_y_3d**2 + point_z_3d**2)
                camera_pos = top_camera.cfg.offset.pos
                self.contact_point = torch.as_tensor([[point_x_3d+camera_pos[0], point_y_3d+camera_pos[1], point_z_3d+camera_pos[2]]], device=env.device).float()
                print(contact_point_2d)
                print(self.contact_point)'''

                '''K = top_camera.data.intrinsic_matrices
                depth_image = top_camera.data.output["depth"]
                depths = depth_image[..., int(y), int(x), 0]
                pixels = torch.as_tensor([[y, x]], device=env.device)/256.0
                #pixels = (pixels*2.0) - 1.0
                contact_point_3d_pos = project_2d_to_3d_pytorch(pixels=pixels, depths=depths, K=K).reshape((1, 3))
                contact_point_3d_quat = torch.as_tensor([[1.0, 0.0, 0.0, 0.0]], device='cuda')
                contact_point_3d_pose = PoseUtils.make_pose(pos=contact_point_3d_pos, rot=PoseUtils.matrix_from_quat(contact_point_3d_quat))
                eef_pose = env.get_robot_eef_pose(eef_name="robot")
                #robot_base_pos = env.scene.articulations["robot"].data.root_pos_w
                #robot_base_quat = env.scene.articulations["robot"].data.root_quat_w
                #robot_base_pose = PoseUtils.make_pose(pos=robot_base_pos, rot=PoseUtils.matrix_from_quat(robot_base_quat))
                camera_pos_offset = torch.as_tensor([top_camera.cfg.offset.pos], device=env.device)
                camera_quat_offset = torch.as_tensor([top_camera.cfg.offset.rot], device=env.device)
                camera_pose_offset = PoseUtils.make_pose(pos=camera_pos_offset, rot=PoseUtils.matrix_from_quat(camera_quat_offset))
                contact_pose_robot_frame = camera_pose_offset @ contact_point_3d_pose
                #contact_pose_robot_frame[..., 2, 3] *= 1.0
                print(contact_pose_robot_frame.shape)
                self.contact_point = contact_pose_robot_frame[..., :3, 3].reshape((1, 3)).clone()
                self.contact_point[0, 0] = contact_pose_robot_frame[..., 2, 3]
                self.contact_point[0, 2] = contact_pose_robot_frame[..., 0, 3]
                print(torch.max(depth_image.flatten(), dim=-1))
                print(self.contact_point, eef_pose[..., :3, 3])'''
                
                eef_pose = env.get_robot_eef_pose(eef_name="robot")
                object_pose = env.get_object_poses()[self.goal_object.name]
                #object_pose[..., 2, 3] -= 0.05
                #object_pose[..., 0, 3] += 0.05
                #grasp_pose = self.oracle_policy.sort_grasp_poses(initial_pose=eef_pose, pickup_object_pose=object_pose)[0]
                grasp_pose = random.choice(self.oracle_policy.grasp_poses)
                offset_pos = torch.as_tensor([[0.0, 0.0, 0.1+0.18]], device='cuda')
                offset_quat = torch.as_tensor([[1.0, 0.0, 0.0, 0.0]], device='cuda')
                offset_pose = PoseUtils.make_pose(pos=offset_pos, rot=PoseUtils.matrix_from_quat(offset_quat))
                #contact_pose = (object_pose @ grasp_pose) @ offset_pose
                contact_pose = object_pose
                self.contact_point = contact_pose[..., :3, 3]
            frame["observation.state"] = torch.cat([
                frame["observation.state"],
                self.contact_point
            ], dim=-1)
        batch = self.pre_processor(frame)
        #print(batch)
        #exit()
        action = self.post_processor(
            self.policy.select_action(
                batch,
            )
        ).to(env.device)
        #action[..., 3] += torch.pi/2
        #if action.shape[-1] == 16:
        #    action[..., 7] = torch.where(action[..., 7] > 0.04, 1.0, -1.0)
        #    action[..., 15] = torch.where(action[..., 15] > 0.04, 1.0, -1.0)
        return action
    
    def reset(self, env_ids=None):
        self.policy.reset()
        self.contact_point = None