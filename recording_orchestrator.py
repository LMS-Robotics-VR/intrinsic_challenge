import subprocess
import time
import argparse
import sys

def run_batch(batch_number, policy_name):
    print(f"\n{'='*50}")
    print(f"STARTING BATCH {batch_number}")
    print(f"{'='*50}\n")

    # 1tart the Evaluation Environment
    sim_cmd = [
        "distrobox", "enter", "-r", "aic_eval", "--", 
        "/entrypoint.sh", 
        "ground_truth:=true", 
        "start_aic_engine:=true", 
        "shutdown_on_aic_engine_exit:=true"
    ]
    
    print("[Orchestrator] Launching Simulation...")
    # Popen runs the process in the background
    sim_process = subprocess.Popen(sim_cmd)

    # Give Gazebo and Zenoh time warm up
    time.sleep(15) 

    # 2. Start the Policy (Your Brain)
    policy_cmd = [
        "pixi", "run", "ros2", "run", "aic_model", "aic_model", 
        "--ros-args", 
        "-p", "use_sim_time:=true", 
        "-p", f"policy:={policy_name}"
    ]
    
    print(f"[Orchestrator] Launching Policy: {policy_name}...")
    policy_process = subprocess.Popen(policy_cmd)

    try:
        print("[Orchestrator] Waiting for trials to complete...")
        sim_process.wait()
        
    except KeyboardInterrupt:
        print("\n[Orchestrator] Manual interrupt detected. Cleaning up...")
    
    finally:
        # The sim is dead, kill the policy process
        if policy_process.poll() is None:
            print("[Orchestrator] Terminating Policy process...")
            policy_process.terminate()
            policy_process.wait()
            
        # Ensure the sim process is also dead if interrupted
        if sim_process.poll() is None:
            sim_process.terminate()
            sim_process.wait()

    print(f"\nBATCH {batch_number} COMPLETE.\n")

def main():
    parser = argparse.ArgumentParser(description="Run AIC episodes continuously.")
    parser.add_argument("--batches", type=int, default=1, help="Number of times to run the engine (each run has 3 trials).")
    parser.add_argument("--policy", type=str, default="aic_example_policies.ros.CheatCode", help="The policy class to run.")
    args = parser.parse_args()


    print(f"Starting orchestrator for {args.batches} batches (Approx {args.batches * 3} trials).")

    for i in range(1, args.batches + 1):
        run_batch(i, args.policy)
        time.sleep(5) 
        
    print("All requested batches finished successfully!")

if __name__ == "__main__":
    main()