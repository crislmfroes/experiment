from gr00t.configs.data.embodiment_configs import register_modality_config
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import (
    ActionConfig,
    ActionFormat,
    ActionRepresentation,
    ActionType,
    ModalityConfig,
)


xarm6_config = {
    # Video: use current frame only ([0]); list camera view names matching modality.json
    "video": ModalityConfig(
        delta_indices=[0],
        modality_keys=[
            "top",
            "wrist",
        ],
    ),
    # State: current proprioceptive reading; keys must match modality.json "state" entries
    "state": ModalityConfig(
        delta_indices=[0],
        modality_keys=[
            "single_arm",
            "gripper",
            "object_pose"
        ],
    ),
    # Action: 16-step prediction horizon; each key needs an ActionConfig
    "action": ModalityConfig(
        delta_indices=list(range(0, 16)),  # predict 16 future steps
        modality_keys=[
            "single_arm",
            "gripper",
        ],
        action_configs=[
            # single_arm: RELATIVE = delta from current state (better generalization)
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
                type=ActionType.NON_EEF,       # joint-space, not end-effector
                format=ActionFormat.DEFAULT,
            ),
            # gripper: ABSOLUTE = target position (binary open/close works better absolute)
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
        ],
    ),
    # Language: task instruction from annotation field in the dataset
    "language": ModalityConfig(
        delta_indices=[0],
        modality_keys=["annotation.human.task_description"],
    ),
}

# Important: always register under EmbodimentTag.NEW_EMBODIMENT for custom embodiments
register_modality_config(xarm6_config, embodiment_tag=EmbodimentTag.NEW_EMBODIMENT)