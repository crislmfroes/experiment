from isaaclab_arena.assets.object_library import LibraryObject, ObjectType
from isaaclab_arena.affordances.openable import Openable
from isaaclab_arena.assets.register import register_asset
import fbot_arena
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
import torch

@register_asset
class SektionCabinet(LibraryObject, Openable):
    name = "sektion_cabinet"
    tags = ["object"]
    usd_path = f"{ISAAC_NUCLEUS_DIR}/Props/Sektion_Cabinet/sektion_cabinet_instanceable.usd"
    object_type = ObjectType.ARTICULATION
    open_joint_pos = 0.4
    closed_joint_pos = 0.0

    def open(self, env, env_ids, asset_cfg = None, percentage = 1):
        if env_ids == None:
            env_ids = torch.as_tensor(list(range(env.num_envs)), device=env.device)
        env.scene.articulations[self.name].write_joint_position_to_sim(position=percentage*self.open_joint_pos*torch.ones((max(len(env_ids), 1), max(len(env_ids), 1)), device=env.device), joint_ids=env.scene.articulations[self.name].joint_names.index(self.openable_joint_name), env_ids=env_ids)

    def close(self, env, env_ids, asset_cfg = None, percentage = 1):
        if env_ids == None:
            env_ids = torch.as_tensor(list(range(env.num_envs)), device=env.device)
        env.scene.articulations[self.name].write_joint_position_to_sim(position=percentage*self.open_joint_pos*torch.ones((max(len(env_ids), 1), max(len(env_ids), 1)), device=env.device), joint_ids=env.scene.articulations[self.name].joint_names.index(self.openable_joint_name), env_ids=env_ids)

    def get_openness(self, env, asset_cfg = None):
        return env.scene.articulations[self.name].data.joint_pos[..., env.scene.articulations[self.name].joint_names.index(self.openable_joint_name)]/self.open_joint_pos
    
    def is_open(self, env, asset_cfg = None, threshold = None):
        return self.get_openness(env=env, asset_cfg=asset_cfg) >= threshold