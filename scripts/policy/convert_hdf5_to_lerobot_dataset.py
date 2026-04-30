# Copyright 2025 Lightwheel Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import ast
import os
from pathlib import Path

import yaml
import cv2
import h5py
import numpy as np
import tqdm
import glob

from lerobot.datasets.lerobot_dataset import LeRobotDataset

from threading import Thread
#from lw_benchhub.utils.math_utils.transform_utils.numpy_impl import compute_delta_pose, pose_left_multiply
from scipy.spatial.transform import Rotation

def convert_isaaclab_to_lerobot(args, config):
    # Load configuration: features, robot_type, and default task
    features = config["features"]
    goal_object_name = args.goal_object_name
    use_environment_state = goal_object_name != None
    robot_type = config["robot_type"]
    repo_id = args.tgt_repo_id or f"{Path(args.root_path).stem}-lerobot"
    root = Path(args.root_path).parent / repo_id

    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        root=str(root),
        fps=30,
        robot_type=robot_type,
        features=features,
    )

    clip_names = os.listdir(args.root_path)
    #dataset_files = [os.path.join(args.root_path, clip_name, 'dataset_success.hdf5') for clip_name in clip_names]
    dataset_files = glob.glob(f"{args.root_path}/*/*.hdf5")
    success = 0
    failed = 0
    failed_list = []

    threads = []

    for i, dataset_file in enumerate(dataset_files):
        try:
            process_hdf5(dataset, dataset_file, args.select_cameras, config["task_description"], use_env_state=use_environment_state, goal_object_name=goal_object_name, use_contact_point=args.use_contact_point)
            success += 1
            print(f"Processed {i+1}/{len(dataset_files)}: {dataset_file}")
        except Exception as e:
            failed += 1
            failed_list.append(dataset_file)
            print(f"Failed to process {dataset_file}: {e}")
        print(f"Success: {success}, Failed: {failed}")
        print(f"Failed list: {failed_list}")
    dataset.finalize()
    dataset.push_to_hub()

def process_hdf5(dataset, hdf5_path, cam_names, task, use_env_state=False, goal_object_name=None, use_contact_point=False):
    with h5py.File(hdf5_path, "r") as f:
        demo_names = list(f["data"].keys())
        episode_count = len(demo_names)
        #episode_count = 50
        print(f"Found {len(demo_names)} demos: {demo_names}")
        demo_names.sort(key=lambda x: int(x.split("_")[-1]))

        for i in tqdm.tqdm(range(0, episode_count), desc="Convert last demo"):
            demo_name = demo_names[i]
            demo_group = f["data"][demo_name]

            if "actions" not in demo_group.keys():
                continue

            if use_env_state == True:
                if "rigid_object" in demo_group["states"].keys() and goal_object_name in demo_group["states"]["rigid_object"].keys():
                    goal_object_pose = demo_group["states"]["rigid_object"][goal_object_name]["root_pose"]
                elif "articulation" in demo_group["states"].keys() and goal_object_name in demo_group["states"]["articulation"].keys():
                    goal_object_pose = demo_group["states"]["articulation"][goal_object_name]["root_pose"]

            if len(demo_group["actions"]) <= 50:
                continue

            #actions = demo_group["actions"]
            actions = demo_group["actions"]
            #actions = demo_group["obs"]["joint_pos"]
            
            if hasattr(demo_group["obs"], "keys"):
                states = np.concatenate([
                    demo_group["obs"]["eef_pos"],
                    demo_group["obs"]["eef_quat"],
                    demo_group["obs"]["gripper_pos"],
                ], axis=-1)
            else:
                states = demo_group["obs"]
            #states = demo_group["obs"]["joint_pos"]
            #camera_top = demo_group["camera_obs"]
            camera_top = demo_group["obs"]["top_camera_rgb"]
            camera_wrist = demo_group["obs"]["wrist_camera_rgb"]
            #camera_right_wrist = demo_group["obs"]["right_wrist_camera_rgb"]
            T = actions.shape[0]

            if use_contact_point == True:
                for j in range(50, T):
                    if actions[j][-1] == -1.0:
                        contact_eef_pos = demo_group["obs"]["eef_pos"][j]
                        contact_eef_quat = demo_group["obs"]["eef_quat"][j]
                        contact_eef_pose = np.zeros((4,4), dtype=np.float32)
                        contact_eef_pose[3, 3] = 1.0
                        contact_eef_pose[:3, 3] = contact_eef_pos
                        contact_eef_pose[:3, :3] = Rotation.from_quat(contact_eef_quat, scalar_first=True).as_matrix()
                        contact_eef_offset_pos = np.asarray([0.0, 0.0, 0.18], dtype=np.float32)
                        contact_eef_offset_quat = np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
                        contact_eef_offset_pose = np.zeros((4,4), dtype=np.float32)
                        contact_eef_offset_pose[3, 3] = 1.0
                        contact_eef_offset_pose[:3, 3] = contact_eef_offset_pos
                        contact_eef_offset_pose[:3, :3] = Rotation.from_quat(contact_eef_offset_quat, scalar_first=True).as_matrix()
                        contact_pose = contact_eef_pose @ contact_eef_offset_pose
                        contact_point = contact_pose[:3, 3].flatten()
                        break


            for j in tqdm.trange(50, T):
                #if np.allclose(actions[j, :6], np.zeros(shape=(6,)), rtol=0.005, atol=0.005) and actions[j, 6] > 0.5 and np.allclose(actions[j, 7:13], np.zeros(shape=(6,)), rtol=0.005, atol=0.005) and actions[j, 13] > 0.5:
                #    print('no ops action detected!')
                #    continue
                state = states[j]
                #next_state = states[min(j+1, T-1)]
                #action = generator.normal(loc=0.0, scale=0.01, size=state.shape) + state
                #action[:7] = next_state[:7]
                #action[7] = actions[j, 6]
                #action[8:15] = next_state[8:15]
                #action[15] = actions[j, 13]

                action = actions[j]
                frame = {
                    "observation.state": state, #generator.normal(loc=0.0, scale=0.05, size=state.shape).astype(np.float32) + state,
                    "action": action,
                    "observation.images.top": camera_top[j],
                    "observation.images.wrist": camera_wrist[j],
                    "task": task
                }
                if use_env_state == True:
                    frame["observation.state"] = np.concatenate([frame["observation.state"], goal_object_pose[j]])
                if use_contact_point == True:
                    frame["observation.state"] = np.concatenate([frame["observation.state"], contact_point])
                dataset.add_frame(frame)
            dataset.save_episode()

            '''actions = np.array(demo_group["eef/relative_left_pose"])    # (T,7) [dx,dy,dz,qw,qx,qy,qz]
            actions_abs = np.array(demo_group["eef/left_pose"])         # (T,7) [x,y,z,  qw,qx,qy,qz]

            pose_curr = pose_left_multiply(actions_abs, actions)

            first_base = np.array([[0.3724, 0.1508, 0.7425, 0, 0, 0, 0]])
            urdf_base = np.array([[0.3725, 0.1508, 0.263, 0, 0, 0, 0]])
            offset = first_base - urdf_base

            pose_urdf_curr = pose_curr - offset

            delta = compute_delta_pose(pose_urdf_curr, actions_abs).astype(np.float32)

            action_gripper = (np.array(demo_group["obs/raw_action/lgrasp"])[:, None] + 1) / 2  # (T,1)
            action_6d = np.concatenate([delta, action_gripper], axis=-1)

            state_gripper = np.array(demo_group["obs/joint_pos"])[:, -1:] / 0.044  # (T,1)
            state_6d = np.concatenate([pose_urdf_curr.astype(np.float32), state_gripper], axis=-1)

            T = action_6d.shape[0]

            video_paths = {
                cam_name: Path(hdf5_path).parent / 'replay_results' / demo_name / f"{cam_name}.mp4" for cam_name in cam_names
            }
            cap_cams = {cam_name: cv2.VideoCapture(str(video_paths[cam_name])) for cam_name in cam_names}

            for j in range(5):
                for cam_name in cam_names:
                    _, _ = cap_cams[cam_name].read()

            for i in tqdm.tqdm(range(5, T), desc="Processing frames"):
                frame = {
                    "observation.state": state_6d[i],   # (7,) = [x,y,z, wx,wy,wz, gripper]
                    "action": action_6d[i],             # (7,) = [dx,dy,dz, wx_rel,wy_rel,wz_rel, gripper]
                }
                for cam_name in cam_names:
                    _, img = cap_cams[cam_name].read()
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    frame[f"observation.images.{cam_name}"] = img

                dataset.add_frame(frame, task="Grab the block and lift it up.")
            dataset.save_episode()
            for cam_name in cam_names:
                cap_cams[cam_name].release()'''


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tgt_repo_id", type=str, default=None, help="LeRobot dataset repo_id (folder name)")
    parser.add_argument("--config_yaml", type=str, default=None, help="Path to YAML configuration file")
    parser.add_argument("--root_path", type=str, default=None, help="Path to the root directory of the dataset")
    parser.add_argument("--goal_object_name", type=str, required=False, default=None)
    parser.add_argument("--use_contact_point", type=bool, default=False)
    args = parser.parse_args()

    # Load YAML config
    with open(args.config_yaml, "r") as f:
        config = yaml.safe_load(f)
    config['features']['observation.state']['shape'] = ast.literal_eval(config['features']['observation.state']['shape'])
    config['features']['action']['shape'] = ast.literal_eval(config['features']['action']['shape'])
    args.select_cameras = [i.split(".")[-1] for i in config['features'] if config['features'][i]['dtype'] == 'video']
    convert_isaaclab_to_lerobot(args, config)
