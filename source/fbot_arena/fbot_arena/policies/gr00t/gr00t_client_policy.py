from isaaclab_arena.policy.policy_base import PolicyBase
from gr00t.policy.server_client import PolicyClient
import torch
import numpy as np

class GR00TClientPolicy(PolicyBase):
    def __init__(self, task: str, horizon: int=8):
        super().__init__()
        self.client = PolicyClient()
        self.task = task
        self.action_index = 0
        self.action = None
        self.horizon = horizon

    def get_action(self, env, observation):
        if self.action == None or self.action_index >= self.horizon:
            action, info = self.client.get_action(
                observation={
                    "video": {
                        "top": observation["policy"]["top_camera_rgb"].detach().cpu().numpy().reshape((env.num_envs, 1, 256, 256, 3)),
                        "wrist": observation["policy"]["wrist_camera_rgb"].detach().cpu().numpy().reshape((env.num_envs, 1, 256, 256, 3)),
                    },
                    "state": {
                        "single_arm": torch.cat([
                                observation["policy"]["eef_pos"],
                                observation["policy"]["eef_quat"],
                            ], dim=-1).detach().cpu().numpy().reshape((env.num_envs, 1, 7)),
                        "gripper": observation["policy"]["gripper_pos"].detach().cpu().numpy().reshape((env.num_envs, 1, 1)),
                        "object_pose": torch.cat([
                                observation["policy"]["object_pos"],
                                observation["policy"]["object_quat"],
                            ], dim=-1).detach().cpu().numpy().reshape((env.num_envs, 1, 7)),
                    },
                    "language": {
                        "annotation.human.task_description": [[self.task]]
                    }
                }
            )
            self.action = action
            self.action_index = 0
        action_to_return = torch.as_tensor(
            np.concatenate([
                self.action["single_arm"][:, self.action_index, :],
                self.action["gripper"][:, self.action_index, :]
            ], axis=-1),
            device=env.device
        )
        self.action_index += 1
        return action_to_return

    def reset(self, env_ids = None):
        self.client.reset()
        self.action_index = 0
        self.action = None
