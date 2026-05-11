import subprocess
import time
import argparse
import random
import os
import yaml
import shutil
import signal
import rclpy
from datetime import datetime
from PIL import Image
from datasets import Dataset, Features, Sequence, Value, Image as HfImage

from lerobot_robot_aic.aic_robot_aic_controller import AICRobotAICController, AICRobotAICControllerConfig

os.environ["DBX_CONTAINER_MANAGER"] = "docker"

ROOT_DIR = os.path.expanduser("~/ws_aic/src/aic/aic_data_collection")
ANALYTICS_DIR = os.path.join(ROOT_DIR, "analytics")
DATASET_DIR = os.path.join(ROOT_DIR, "dataset")

# Slice off the end of the dataset to prevent learning "terminal hesitation"
STABILIZATION_TRIM_SECONDS = 5.0

def generate_single_trial_config():
    board_pose = {
        'x': round(random.uniform(0.10, 0.26), 4),
        'y': round(random.uniform(-0.30, 0.15), 4),
        'z': round(random.uniform(1.135, 1.145), 4),
        'roll': 0.0, 'pitch': 0.0, 'yaw': round(random.uniform(2.5, 3.8), 4)
    }

    scene_elements = {}
    active_nics = []
    active_scs = []
    task_type = random.choice(["sfp", "sc"])

    if task_type == "sfp":
        for i in range(5):
            present = True if i == 0 else random.choice([True, False])
            if present: active_nics.append(i)
            scene_elements[f"nic_rail_{i}"] = {
                'entity_present': present,
                'entity_name': f"nic_card_{i}",
                'entity_pose': {'translation': round(random.uniform(0.0, 0.062), 4), 'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0}
            }
        for i in range(2):
            scene_elements[f"sc_rail_{i}"] = {'entity_present': False}

        target_idx = random.choice(active_nics)
        target_name = f"nic_card_mount_{target_idx}"
        port_name = f"sfp_port_{random.choice([0, 1])}"
        plug_type = "sfp"; plug_name = "sfp_tip"; port_type = "sfp"
        spawn_cable_type = "sfp_sc_cable"; grasp_z_offset = 0.04245
        cable_name = "cable_0" # SFP uses cable_0
        prompt = f"Insert the SFP module into {port_name} on {target_name}."

    else:
        for i in range(5):
            scene_elements[f"nic_rail_{i}"] = {'entity_present': False}
        for i in range(2):
            present = True if i == 0 else random.choice([True, False])
            if present: active_scs.append(i)
            scene_elements[f"sc_rail_{i}"] = {
                'entity_present': present,
                'entity_name': f"sc_mount_{i}",
                'entity_pose': {'translation': round(random.uniform(0.0, 0.115), 4), 'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0}
            }

        target_idx = random.choice(active_scs)
        target_name = f"sc_port_{target_idx}"
        port_name = "sc_port_base"
        plug_type = "sc"; plug_name = "sc_tip"; port_type = "sc"
        spawn_cable_type = "sfp_sc_cable_reversed"; grasp_z_offset = 0.04045
        cable_name = "cable_1" # SC uses cable_1
        prompt = f"Insert the SC plug into {port_name} on {target_name}."

    for rail in ['lc_mount_rail_0', 'sfp_mount_rail_0', 'sc_mount_rail_0',
                 'lc_mount_rail_1', 'sfp_mount_rail_1', 'sc_mount_rail_1']:
        scene_elements[rail] = {
            'entity_present': random.choice([True, False]),
            'entity_pose': {'translation': round(random.uniform(-0.096, 0.096), 4), 'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0}
        }
    
    config = {
        'scoring': {
            'topics': [
                {'topic': {'name': '/joint_states', 'type': 'sensor_msgs/msg/JointState'}},
                {'topic': {'name': '/tf', 'type': 'tf2_msgs/msg/TFMessage'}},
                {'topic': {'name': '/tf_static', 'type': 'tf2_msgs/msg/TFMessage', 'latched': True}},
                {'topic': {'name': '/scoring/tf', 'type': 'tf2_msgs/msg/TFMessage'}},
                {'topic': {'name': '/aic/gazebo/contacts/off_limit', 'type': 'ros_gz_interfaces/msg/Contacts'}},
                {'topic': {'name': '/fts_broadcaster/wrench', 'type': 'geometry_msgs/msg/WrenchStamped'}},
                {'topic': {'name': '/aic_controller/joint_commands', 'type': 'aic_control_interfaces/msg/JointMotionUpdate'}},
                {'topic': {'name': '/aic_controller/pose_commands', 'type': 'aic_control_interfaces/msg/MotionUpdate'}},
                {'topic': {'name': '/scoring/insertion_event', 'type': 'std_msgs/msg/String'}},
                {'topic': {'name': '/aic_controller/controller_state', 'type': 'aic_control_interfaces/msg/ControllerState'}}
            ]
        },
        'task_board_limits': {
            'nic_rail': {'min_translation': -0.0215, 'max_translation': 0.0234},
            'sc_rail': {'min_translation': -0.06, 'max_translation': 0.055},
            'mount_rail': {'min_translation': -0.09425, 'max_translation': 0.09425}
        },
        'robot': {
            'home_joint_positions': {
                'shoulder_pan_joint': -0.1597, 'shoulder_lift_joint': -1.3542, 'elbow_joint': -1.6648,
                'wrist_1_joint': -1.6933, 'wrist_2_joint': 1.5710, 'wrist_3_joint': 1.4110
            }
        },
        'trials': {
            'trial_1': {
                'scene': {
                    'task_board': {**{'pose': board_pose}, **scene_elements},
                    'cables': {
                        cable_name: {
                            'pose': {
                                'gripper_offset': {'x': 0.0, 'y': 0.015385, 'z': grasp_z_offset},
                                'roll': 0.4432, 'pitch': -0.4838, 'yaw': 1.3303
                            },
                            'attach_cable_to_gripper': True,
                            'cable_type': spawn_cable_type
                        }
                    }
                },
                'tasks': {
                    'task_1': {
                        'cable_type': "sfp_sc", 'cable_name': cable_name, 'plug_type': plug_type,
                        'plug_name': plug_name, 'port_type': port_type, 'port_name': port_name,
                        'target_module_name': target_name, 'time_limit': 180
                    }
                }
            }
        }
    }

    config_path = f"/tmp/proc_config.yaml"
    with open(config_path, 'w') as f:
        yaml.dump(config, f, sort_keys=False)
    return config_path, prompt, task_type

def check_trial_success(results_dir, task_type, trial_id="trial_1"):
    scoring_file = os.path.join(results_dir, "scoring.yaml")
    if not os.path.exists(scoring_file): return False
    
    with open(scoring_file, 'r') as f:
        scoring_data = yaml.safe_load(f)
        
    try:
        tier_3_score = float(scoring_data[trial_id]['tier_3']['score'])
        total_score = float(scoring_data['total'])
        if task_type == "sfp":
            return tier_3_score == 75.0
        else:
            return total_score >= 60.0
    except (KeyError, TypeError, ValueError):
        return False

def hard_reset_container():
    """Nuclear option: Clears RAM, /dev/shm, and resets networking completely."""
    print("[Orchestrator] Executing Hard Container Reset to clear shared memory...")
    subprocess.run(["distrobox", "stop", "aic_eval", "--yes"], capture_output=True)
    subprocess.run(["docker", "restart", "aic_eval"], capture_output=True)
    time.sleep(5) # Let Docker allocate new virtual interfaces

def run_episode(episode_index, policy_name, target_episodes):
    config_path, language_prompt, task_type = generate_single_trial_config()
    ep_results_dir = os.path.join(ANALYTICS_DIR, f"ep_{episode_index}")
    os.makedirs(ep_results_dir, exist_ok=True)
    
    sim_env = os.environ.copy()
    sim_env["AIC_RESULTS_DIR"] = ep_results_dir
    
    sim_cmd = ["distrobox", "enter", "aic_eval", "--", "bash", "-c",
               f"source /ws_aic/install/setup.bash && ros2 launch aic_bringup aic_gz_bringup.launch.py ground_truth:=true start_aic_engine:=true shutdown_on_aic_engine_exit:=true aic_engine_config_file:={config_path} use_sim_time:=true"]
    sim_proc = subprocess.Popen(sim_cmd, env=sim_env, preexec_fn=os.setsid)

    time.sleep(15) # Wait for Gazebo to fully stand up
    
    robot = AICRobotAICController(AICRobotAICControllerConfig(teleop_target_mode="cartesian", teleop_frame_id="gripper/tcp"))
    try:
        robot.connect(calibrate=False)
    except Exception as e:
        print(f"[Orchestrator] Failed to connect LeRobot controller: {e}")
        os.killpg(os.getpgid(sim_proc.pid), signal.SIGKILL)
        return False

    policy_env = os.environ.copy()
    policy_env["PYTHONPATH"] = f"{os.getcwd()}:{policy_env.get('PYTHONPATH', '')}"
    policy_cmd = ["pixi", "run", "ros2", "run", "aic_model", "aic_model", "--ros-args", "-p", "use_sim_time:=true", "-p", f"policy:={policy_name}", "--log-level", "WARN"]
    policy_process = subprocess.Popen(policy_cmd, env=policy_env, preexec_fn=os.setsid)

    dataset_dict = {"action": [], "observation.state": [], "observation.image": [], "observation.wrist_image": [], "timestamp": [], "frame_index": [], "episode_index": [], "index": [], "task_index": [], "language_instruction": []}
    frame_idx = 0
    success_flag = False

    try:
        episode_start = time.time()
        while sim_proc.poll() is None:
            if time.time() - episode_start > 240.0:
                print("[Orchestrator] Maximum trial time exceeded. Forcing teardown...")
                break
                
            loop_start = time.time()
            if policy_process.poll() is None:
                obs = robot.get_observation(); act = robot.get_latest_action()
                if obs and act:
                    full_state = [obs["tcp_pose.position.x"], obs["tcp_pose.position.y"], obs["tcp_pose.position.z"], obs["tcp_pose.orientation.x"], obs["tcp_pose.orientation.y"], obs["tcp_pose.orientation.z"], obs["tcp_pose.orientation.w"]] + [obs["observation.force.fx"], obs["observation.force.fy"], obs["observation.force.fz"], obs["observation.force.tx"], obs["observation.force.ty"], obs["observation.force.tz"]]
                    dataset_dict["action"].append(act); dataset_dict["observation.state"].append(full_state)
                    dataset_dict["observation.image"].append(Image.fromarray(obs["left_camera"][..., ::-1]))
                    dataset_dict["observation.wrist_image"].append(Image.fromarray(obs["right_camera"][..., ::-1]))
                    dataset_dict["timestamp"].append(loop_start); dataset_dict["frame_index"].append(frame_idx)
                    dataset_dict["episode_index"].append(episode_index); dataset_dict["index"].append(frame_idx)
                    dataset_dict["task_index"].append(0); dataset_dict["language_instruction"].append(language_prompt)
                    frame_idx += 1
            time.sleep(max(0, 0.05 - (time.time() - loop_start)))
            
    except KeyboardInterrupt:
        raise
    finally:
        # Tear everything down
        if policy_process.poll() is None: os.killpg(os.getpgid(policy_process.pid), signal.SIGKILL)
        robot.disconnect()
        if rclpy.ok(): rclpy.shutdown()
        if sim_proc.poll() is None: os.killpg(os.getpgid(sim_proc.pid), signal.SIGKILL)
        
        # Kill the stragglers
        subprocess.run(["distrobox", "enter", "aic_eval", "--", "bash", "-c", "pkill -9 -f 'ruby' || true && pkill -9 -f 'ros2' || true && pkill -9 -f 'gz' || true"], capture_output=True)

    if check_trial_success(ep_results_dir, task_type):
        trim_frames = int(STABILIZATION_TRIM_SECONDS / 0.05)
        if frame_idx > trim_frames:
            for key in dataset_dict: dataset_dict[key] = dataset_dict[key][:-trim_frames]
            
        hf_dataset = Dataset.from_dict(dataset_dict, features=Features({"action": Sequence(Value("float32"), length=7), "observation.state": Sequence(Value("float32"), length=13), "observation.image": HfImage(), "observation.wrist_image": HfImage(), "timestamp": Value("float32"), "frame_index": Value("int64"), "episode_index": Value("int64"), "index": Value("int64"), "task_index": Value("int64"), "language_instruction": Value("string")}))
        save_dir = os.path.join(DATASET_DIR, f"{datetime.now().strftime('%y%m%d_%H%M')}_episode_{episode_index:06d}")
        os.makedirs(save_dir, exist_ok=True); hf_dataset.to_parquet(os.path.join(save_dir, f"episode.parquet"))
        success_flag = True

    if os.path.exists(ep_results_dir): shutil.rmtree(ep_results_dir)
    return success_flag

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--policy", type=str, required=True)
    args = parser.parse_args()
    
    for d in [ROOT_DIR, ANALYTICS_DIR, DATASET_DIR]: 
        os.makedirs(d, exist_ok=True)
        
    current_episodes = 0
    zenoh_proc = None

    try:
        while current_episodes < args.episodes:
            print(f"\n{'='*50}\nExecuting Episode {current_episodes + 1} / {args.episodes}\n{'='*50}")
            
            if zenoh_proc and zenoh_proc.poll() is None:
                os.killpg(os.getpgid(zenoh_proc.pid), signal.SIGKILL)
                zenoh_proc.wait()
            
            zenoh_proc = subprocess.Popen(["ros2", "run", "rmw_zenoh_cpp", "rmw_zenohd"], preexec_fn=os.setsid, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            time.sleep(2)

            success = run_episode(current_episodes, args.policy, args.episodes)
            
            if success:
                current_episodes += 1
            else:
                print("[Orchestrator] Trial failed (Score threshold not met, or TF crash). Re-rolling environment...")
                # The Nuclear Option: Completely clears the container's RAM, TF Cache, and Shared Memory before retrying.
                hard_reset_container()
                
    except KeyboardInterrupt:
        print("\n[Orchestrator] Data collection interrupted by user. Performing clean host exit...")
    finally:
        if zenoh_proc and zenoh_proc.poll() is None:
            os.killpg(os.getpgid(zenoh_proc.pid), signal.SIGKILL)
        subprocess.run("pkill -9 -f 'ruby' || true && pkill -9 -f 'ros2' || true && pkill -9 -f 'gz' || true", shell=True, capture_output=True)

if __name__ == "__main__": 
    main()