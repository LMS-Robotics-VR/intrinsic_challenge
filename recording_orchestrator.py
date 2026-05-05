import subprocess
import time
import argparse
import random
import os
import yaml
import csv
import statistics  # Added for median calculations
from datetime import datetime

os.environ["DBX_CONTAINER_MANAGER"] = "docker"

def generate_procedural_trial(cable_id_num):
    # 1. Broad Continuous Distribution for the Board
    # We use random.uniform and round to 4 decimal places for near-continuous granularity
    board_pose = {
        'x': round(random.uniform(0.10, 0.26), 4),
        'y': round(random.uniform(-0.30, 0.15), 4),
        'z': round(random.uniform(1.135, 1.145), 4), # Slight height variations
        'roll': 0.0,
        'pitch': 0.0,
        'yaw': round(random.uniform(2.5, 3.8), 4) # Wide yaw arc
    }

    scene_elements = {}
    active_nics = []

    # NIC Cards
    for i in range(5):
        # Guarantee at least one NIC is present so we always have an SFP target
        present = True if i == 0 else random.choice([True, False])
        if present:
            active_nics.append(f"nic_card_mount_{i}")
        scene_elements[f"nic_rail_{i}"] = {
            'entity_present': present,
            'entity_name': f"nic_card_{i}",
            'entity_pose': {
                'translation': round(random.uniform(0.0, 0.062), 4), # Full official range
                'roll': 0.0, 
                'pitch': 0.0, 
                'yaw': 0.0 
            }
        }

    # SC Ports
    for i in range(2):
        scene_elements[f"sc_rail_{i}"] = {
            'entity_present': random.choice([True, False]),
            'entity_name': f"sc_mount_{i}",
            'entity_pose': {
                'translation': round(random.uniform(0.0, 0.115), 4),
                'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0
            }
        }

    # Side Mounts (Zones 3 & 4)
    # Randomly populating the side rails with heavy rotations to confuse the ML model
    for rail in ['lc_mount_rail_0', 'sfp_mount_rail_0', 'sc_mount_rail_0', 
                 'lc_mount_rail_1', 'sfp_mount_rail_1', 'sc_mount_rail_1']:
        scene_elements[rail] = {
            'entity_present': random.choice([True, False]),
            'entity_pose': {
                'translation': round(random.uniform(-0.096, 0.096), 4),
                'roll': 0.0, 
                'pitch': 0.0, 
                'yaw': 0.0
            }
        }

    # Force target_is_nic to True so it only picks SFP tasks
    target_is_nic = True
    cable_name = f"cable_{cable_id_num}"
    
    if target_is_nic:
        target_name = random.choice(active_nics)
        cable_type = "sfp_sc_cable"
        plug_type = "sfp"
        port_type = "sfp"
        port_name = f"sfp_port_{random.choice([0, 1])}" 
        gripper_z_offset = 0.04245
    else:
        target_name = random.choice(active_sc_ports) if active_sc_ports else "sc_port_0"
        cable_type = "sfp_sc_cable_reversed"
        plug_type = "sc"
        port_type = "sc"
        port_name = "sc_port_base"
        gripper_z_offset = 0.04045

    trial_data = {
        'scene': {
            'task_board': {**{'pose': board_pose}, **scene_elements},
            'cables': {
                cable_name: {
                    'pose': {
                        'gripper_offset': {'x': 0.0, 'y': 0.015385, 'z': gripper_z_offset},
                        'roll': 0.4432, 'pitch': -0.4838, 'yaw': 1.3303
                    },
                    'attach_cable_to_gripper': True,
                    'cable_type': cable_type
                }
            }
        },
        'tasks': {
            'task_1': {
                'cable_type': "sfp_sc",
                'cable_name': cable_name,
                'plug_type': plug_type,
                'plug_name': f"{plug_type}_tip",
                'port_type': port_type,
                'port_name': port_name,
                'target_module_name': target_name,
                'time_limit': 180
            }
        }
    }
    return trial_data

def generate_procedural_config(batch_num):
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
        'trials': {
            'trial_1': generate_procedural_trial(0),
            'trial_2': generate_procedural_trial(1),
            'trial_3': generate_procedural_trial(2)
        },
        'robot': {
            'home_joint_positions': {
                'shoulder_pan_joint': -0.1597, 'shoulder_lift_joint': -1.3542,
                'elbow_joint': -1.6648, 'wrist_1_joint': -1.6933,
                'wrist_2_joint': 1.5710, 'wrist_3_joint': 1.4110
            }
        }
    }
    
    config_path = f'/tmp/proc_config_batch_{batch_num}.yaml'
    with open(config_path, 'w') as f:
        yaml.dump(config, f, sort_keys=False)
    
    return config_path, config

def flatten_dict(d, parent_key='', sep='_'):
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

def compile_dashboard(results_dir, batch_number, master_csv_path, batch_config, run_timestamp):
    scoring_file = os.path.join(results_dir, "scoring.yaml")
    
    if not os.path.exists(scoring_file):
        print(f"[Dashboard] Error: No scoring file found at {scoring_file}")
        return

    with open(scoring_file, 'r') as f:
        scoring_data = yaml.safe_load(f)
    
    file_exists = os.path.exists(master_csv_path)
    rows_to_write = []
    
    for trial_id, trial_cfg in batch_config['trials'].items():
        task_info = trial_cfg['tasks']['task_1']
        board_pose = trial_cfg['scene']['task_board']['pose']
        
        trial_scoring = scoring_data.get(trial_id, {})
        flat_scoring = flatten_dict(trial_scoring)
        
        duration_score = 0
        t1_val = 0.0
        t2_val = 0.0
        t3_val = 0.0

        for k, v in flat_scoring.items():
            if isinstance(v, (int, float)):
                kl = k.lower()
                if 'duration' in kl and 'score' in kl:
                    duration_score = float(v)
                if 'tier_1' in kl and 'score' in kl:
                    t1_val = float(v)
                elif 'tier_2' in kl and 'score' in kl and all(x not in kl for x in ['smoothness', 'duration', 'efficiency', 'force', 'contact', 'limit']):
                    t2_val = float(v)
                elif 'tier_3' in kl and 'score' in kl:
                    t3_val = float(v)
        
        total_percentage = ((t1_val + t2_val + t3_val) / 100.0) * 100.0
        
        row_data = {
            'run_timestamp': run_timestamp,
            'batch_num': batch_number,
            'trial_id': trial_id,
            'insertion_succeeded': 'YES' if duration_score > 0 else 'NO',
            'total_score_percentage': f"{total_percentage:.2f}%",
            'port_type': task_info['port_type'],
            'target_module': task_info['target_module_name'],
            'board_x': board_pose['x'],
            'board_y': board_pose['y'],
            'board_yaw': board_pose['yaw']
        }
        
        valid_keywords = ['tier', 'score', 'penalty', 'duration', 'smoothness', 'jerk', 'efficiency', 'force']
        clean_scores = {}
        
        for k, v in flat_scoring.items():
            kl = k.lower()
            
            if 'max' in kl and 'force' in kl:
                clean_scores['tier_2_max_force_detected'] = str(v)
                continue
                
            if 'distance' in kl:
                clean_scores['tier_2_final_distance'] = str(v)
                continue
                
            if any(word in kl for word in valid_keywords) and 'message' not in kl:
                if isinstance(v, (int, float)):
                    max_val = None
                    if 'tier_1' in kl:
                        max_val = 1.0
                    elif 'tier_3' in kl:
                        max_val = 75.0
                    elif 'smoothness' in kl:
                        max_val = 6.0
                    elif 'duration' in kl:
                        max_val = 12.0
                    elif 'efficiency' in kl:
                        max_val = 6.0
                    elif 'force' in kl:
                        max_val = 12.0
                    elif 'contact' in kl or 'limit' in kl:
                        max_val = 24.0
                    elif 'tier_2' in kl:
                        max_val = 24.0
                    
                    if max_val is not None:
                        clean_scores[k] = f"{(float(v) / max_val) * 100:.2f}%"
                    else:
                        clean_scores[k] = str(v)
                else:
                    clean_scores[k] = str(v)
        
        row_data.update(clean_scores)
        rows_to_write.append(row_data)

    with open(master_csv_path, 'a', newline='') as csvfile:
        all_keys = set()
        for r in rows_to_write:
            all_keys.update(r.keys())
            
        priority_cols = [
            'run_timestamp',
            'batch_num', 
            'trial_id', 
            'insertion_succeeded',
            'total_score_percentage',
            'port_type', 
            'target_module',
            'board_x', 
            'board_y', 
            'board_yaw'
        ]
        
        t1_keys = sorted([k for k in all_keys if 'tier_1' in k.lower()])
        t2_keys = sorted([k for k in all_keys if 'tier_2' in k.lower()])
        t3_keys = sorted([k for k in all_keys if 'tier_3' in k.lower()])
        
        ordered_fieldnames = priority_cols + t1_keys + t2_keys + t3_keys
        
        remaining_cols = sorted([k for k in all_keys if k not in ordered_fieldnames])
        final_fieldnames = ordered_fieldnames + remaining_cols
                    
        writer = csv.DictWriter(csvfile, fieldnames=final_fieldnames)
        if not file_exists:
            writer.writeheader()
            
        for row in rows_to_write:
            writer.writerow(row)
        
    print(f"[Dashboard] Batch {batch_number} recorded. {len(rows_to_write)} trials added to dashboard.")

def generate_high_level_summary(master_csv_path, dashboard_dir):
    """Parses the entire master dashboard to generate macro-level KPIs."""
    if not os.path.exists(master_csv_path):
        return

    total_trials = 0
    successes = 0
    total_scores = []
    
    with open(master_csv_path, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            total_trials += 1
            if row.get('insertion_succeeded') == 'YES':
                successes += 1
            
            # Extract float from percentage string (e.g. "95.50%")
            total_val_str = row.get('total_score_percentage', '0').replace('%', '')
            try:
                total_scores.append(float(total_val_str))
            except ValueError:
                pass

    if total_trials == 0:
        return

    success_rate = (successes / total_trials) * 100.0

    summary_text = (
        f"\n{'='*55}\n"
        f"HIGH-LEVEL PIPELINE SUMMARY\n"
        f"{'='*55}\n"
        f"Total Procedural Trials: {total_trials}\n"
        f"Overall Success Rate:    {success_rate:.2f}% ({successes}/{total_trials})\n"
        f"{'='*55}\n"
    )

    print(summary_text)

    # Save to disk alongside the master CSV
    summary_path = os.path.join(dashboard_dir, "high_level_summary.txt")
    with open(summary_path, 'w') as f:
        f.write(summary_text)
    print(f"[Dashboard] KPI Summary written to {summary_path}\n")

def run_batch(batch_number, policy_name, dashboard_dir, _):
    print(f"\n{'='*50}")
    print(f"STARTING BATCH {batch_number}")
    print(f"{'='*50}\n")

    episode_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    config_path, batch_config = generate_procedural_config(batch_number)
    
    batch_results_dir = os.path.join(dashboard_dir, f"batch_{batch_number}")
    os.makedirs(batch_results_dir, exist_ok=True)

    custom_env = os.environ.copy()
    custom_env["DBX_CONTAINER_MANAGER"] = "docker"
    custom_env["AIC_RESULTS_DIR"] = batch_results_dir

    sim_cmd = [
        "distrobox", "enter", "-r", "aic_eval", "--", 
        "/entrypoint.sh", 
        "ground_truth:=true", 
        "start_aic_engine:=true", 
        "shutdown_on_aic_engine_exit:=true",
        f"aic_engine_config_file:={config_path}"
    ]
    
    print("[Orchestrator] Launching Simulation...")
    sim_process = subprocess.Popen(sim_cmd, env=custom_env)

    time.sleep(15) 

    # INJECTING PATH FOR LOCAL POLICY RUNS
    policy_env = os.environ.copy()
    policy_env["PYTHONPATH"] = f"{os.getcwd()}:{policy_env.get('PYTHONPATH', '')}"

    policy_cmd = [
        "pixi", "run", "ros2", "run", "aic_model", "aic_model", 
        "--ros-args", 
        "-p", "use_sim_time:=true", 
        "-p", f"policy:={policy_name}",
        "--log-level", "WARN"
    ]
    
    print(f"[Orchestrator] Launching Policy: {policy_name}...")
    policy_process = subprocess.Popen(policy_cmd, env=policy_env)

    try:
        sim_process.wait()
    except KeyboardInterrupt:
        print("\n[Orchestrator] Manual interrupt detected. Cleaning up...")
    finally:
        if policy_process.poll() is None:
            policy_process.terminate()
            policy_process.wait()
        if sim_process.poll() is None:
            sim_process.terminate()
            sim_process.wait()

    master_csv_path = os.path.join(dashboard_dir, "master_dashboard.csv")
    compile_dashboard(batch_results_dir, batch_number, master_csv_path, batch_config, episode_timestamp)
    print(f"\nBATCH {batch_number} COMPLETE.\n")

def main():
    parser = argparse.ArgumentParser(description="Run procedural AIC episodes with Dashboard.")
    parser.add_argument("--batches", type=int, default=1)
    
    parser.add_argument("--policy", type=str, default="RobustPolicy.RobustPolicy")
    args = parser.parse_args()

    workspace_dir = os.path.expanduser("~/ws_aic/src/aic")
    dashboard_dir = os.path.join(workspace_dir, "analytics_dashboard")
    os.makedirs(dashboard_dir, exist_ok=True)
    
    run_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    print(f"Starting orchestrator for {args.batches} batches.")
    print(f"Run Timestamp: {run_timestamp}")
    print(f"Analytics will be saved to: {dashboard_dir}")

    for i in range(1, args.batches + 1):
        run_batch(i, args.policy, dashboard_dir, run_timestamp)
        time.sleep(5) 
        
    print("All requested batches finished successfully!")
    
    # NEW: Generate and display the final summary after all loops finish
    master_csv_path = os.path.join(dashboard_dir, "master_dashboard.csv")
    generate_high_level_summary(master_csv_path, dashboard_dir)

if __name__ == "__main__":
    main()