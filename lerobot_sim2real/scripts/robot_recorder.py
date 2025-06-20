#!/usr/bin/env python3
"""
robot_recorder.py

Modified version of planner.py for recording demonstration data.
Records video frames, joint angles, and end-effector poses while you manually guide the robot.
Data is saved in a structured format for later training/analysis.
"""
import os
import json
import time
import threading
import numpy as np
import cv2
from datetime import datetime
from pathlib import Path

from dm_control.mujoco import Physics
from dm_control.mujoco.wrapper import mjbindings

from lerobot_sim2real.config.real_robot import create_real_robot
from mani_skill.agents.robots.lerobot.manipulator import LeRobotRealAgent
from webcam_gemini import WebcamVisionPipeline
from soarm_control import (
    quat_to_euler,
    RESET_Q
)

from dotenv import load_dotenv
load_dotenv()

# Thread to continuously fetch latest camera frame
def capture_thread(cap, frame_container, stop_event):
    while not stop_event.is_set():
        ret, frame = cap.read()
        if ret:
            frame_container['frame'] = frame
        time.sleep(0.01)  # Small delay to prevent excessive CPU usage

# Thread to display the camera at full speed
class DisplayThread(threading.Thread):
    def __init__(self, frame_container, stop_event, recording_status):
        super().__init__(daemon=True)
        self.frame_container = frame_container
        self.stop_event = stop_event
        self.recording_status = recording_status

    def run(self):
        while not self.stop_event.is_set():
            frame = self.frame_container.get('frame')
            if frame is not None:
                # Add recording indicator to display
                display_frame = frame.copy()
                if self.recording_status['recording']:
                    # Add red recording dot
                    cv2.circle(display_frame, (30, 30), 15, (0, 0, 255), -1)
                    cv2.putText(display_frame, "REC", (50, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                else:
                    cv2.putText(display_frame, "READY - Press 'r' to record", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                cv2.putText(display_frame, "Press 'q' to quit, 's' to stop recording", (10, frame.shape[0] - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                cv2.imshow('Robot Recorder', display_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                self.stop_event.set()
            elif key == ord('r'):
                self.recording_status['start_recording'] = True
            elif key == ord('s'):
                self.recording_status['stop_recording'] = True
                
        cv2.destroyAllWindows()

class RobotDataRecorder:
    def __init__(self, agent, physics, vision, output_dir="recorded_data"):
        self.agent = agent
        self.physics = physics
        self.vision = vision
        self.output_dir = Path(output_dir)
        
        # Create output directory structure
        self.current_session_dir = None
        self.recording_data = []
        self.recording = False
        self.recording_start_time = None
        
    def create_session_directory(self):
        """Create a new directory for this recording session"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.current_session_dir = self.output_dir / f"session_{timestamp}"
        self.current_session_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        (self.current_session_dir / "frames").mkdir(exist_ok=True)
        (self.current_session_dir / "data").mkdir(exist_ok=True)
        
        print(f"Created session directory: {self.current_session_dir}")
        return self.current_session_dir

    def get_robot_state(self) -> dict:
        """Get current robot joint angles and end-effector pose"""
        # Get joint angles
        q = self.agent.qpos.squeeze().cpu().numpy().tolist()
        
        # Get end-effector pose
        mj = mjbindings.mjlib
        mj.mj_fwdPosition(self.physics.model.ptr, self.physics.data.ptr)
        pos = self.physics.named.data.site_xpos['gripper'].tolist()
        mat = self.physics.named.data.site_xmat['gripper']
        quat = np.empty(4, dtype=np.float64)
        mj.mju_mat2Quat(quat, mat)
        rpy = np.rad2deg(quat_to_euler(quat)).tolist()
        
        return {
            "joint_angles": q,
            "end_effector_position": pos,  # [x, y, z] in meters
            "end_effector_orientation": rpy,  # [roll, pitch, yaw] in degrees
            "end_effector_pose": pos + rpy  # combined [x, y, z, roll, pitch, yaw]
        }

    def start_recording(self):
        """Start a new recording session"""
        if self.recording:
            print("Already recording!")
            return
            
        self.create_session_directory()
        self.recording_data = []
        self.recording = True
        self.recording_start_time = time.time()
        
        print("🔴 Recording started! Manually guide the robot to perform the task.")
        print(f"Data will be saved to: {self.current_session_dir}")

    def stop_recording(self):
        """Stop recording and save all data"""
        if not self.recording:
            print("Not currently recording!")
            return
            
        self.recording = False
        total_time = time.time() - self.recording_start_time
        
        # Save the recording data as JSON
        data_file = self.current_session_dir / "data" / "recording_data.json"
        recording_metadata = {
            "session_info": {
                "start_time": self.recording_start_time,
                "total_duration": total_time,
                "total_samples": len(self.recording_data),
                "sample_rate": "~1 Hz",
                "description": "Robot demonstration recording with joint angles and end-effector poses"
            },
            "data": self.recording_data
        }
        
        with open(data_file, 'w') as f:
            json.dump(recording_metadata, f, indent=2)
        
        print(f"🟢 Recording stopped!")
        print(f"Recorded {len(self.recording_data)} samples over {total_time:.1f} seconds")
        print(f"Data saved to: {data_file}")
        
        # Save a summary
        self.save_recording_summary()

    def save_recording_summary(self):
        """Save a human-readable summary of the recording"""
        summary_file = self.current_session_dir / "recording_summary.txt"
        
        if not self.recording_data:
            return
            
        with open(summary_file, 'w') as f:
            f.write("ROBOT DEMONSTRATION RECORDING SUMMARY\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Session Directory: {self.current_session_dir.name}\n")
            f.write(f"Total Samples: {len(self.recording_data)}\n")
            f.write(f"Duration: {time.time() - self.recording_start_time:.1f} seconds\n\n")
            
            f.write("SAMPLE DATA STRUCTURE:\n")
            f.write("- timestamp: Time since recording started\n")
            f.write("- joint_angles: [6 joint values in radians/degrees]\n")
            f.write("- end_effector_position: [x, y, z] in meters\n")
            f.write("- end_effector_orientation: [roll, pitch, yaw] in degrees\n")
            f.write("- frame_filename: Corresponding camera frame\n\n")
            
            # Show first and last samples
            if len(self.recording_data) > 0:
                f.write("FIRST SAMPLE:\n")
                f.write(json.dumps(self.recording_data[0], indent=2))
                f.write("\n\nLAST SAMPLE:\n")
                f.write(json.dumps(self.recording_data[-1], indent=2))
        
        print(f"Summary saved to: {summary_file}")

    def record_sample(self, frame: np.ndarray):
        """Record a single data sample"""
        if not self.recording:
            return
            
        # Get current timestamp relative to recording start
        current_time = time.time()
        relative_timestamp = current_time - self.recording_start_time
        
        # Get robot state
        robot_state = self.get_robot_state()
        
        # Save frame
        frame_filename = f"frame_{len(self.recording_data):06d}.jpg"
        frame_path = self.current_session_dir / "frames" / frame_filename
        cv2.imwrite(str(frame_path), frame)
        
        # Create data sample
        sample = {
            "sample_id": len(self.recording_data),
            "timestamp": relative_timestamp,
            "absolute_timestamp": current_time,
            "joint_angles": robot_state["joint_angles"],
            "end_effector_position": robot_state["end_effector_position"],
            "end_effector_orientation": robot_state["end_effector_orientation"],
            "end_effector_pose": robot_state["end_effector_pose"],
            "frame_filename": frame_filename
        }
        
        self.recording_data.append(sample)
        
        # Print progress
        if len(self.recording_data) % 5 == 0:  # Print every 5 samples
            print(f"Recorded {len(self.recording_data)} samples... "
                  f"EEF pos: [{robot_state['end_effector_position'][0]:.3f}, "
                  f"{robot_state['end_effector_position'][1]:.3f}, "
                  f"{robot_state['end_effector_position'][2]:.3f}]")

def main():
    os.chdir("lerobot_sim2real/scripts")
    load_dotenv()

    # Robot setup
    uid = "so101"
    real = create_real_robot(uid)
    real.connect()
    agent = LeRobotRealAgent(real)
    # agent.reset(RESET_Q)
    # print('Robot initialized at home position:', RESET_Q)
    physics = Physics.from_xml_string(open("scene.xml").read())

    real.bus.disable_torque()

    # Vision setup
    vision = WebcamVisionPipeline(camera_index=4)
    if not vision.initialize_camera():
        print("Failed to initialize camera!")
        return

    # Shared containers and events
    frame_container = {'frame': None}
    stop_event = threading.Event()
    recording_status = {
        'recording': False,
        'start_recording': False,
        'stop_recording': False
    }

    # Start capture and display threads
    threading.Thread(target=capture_thread, args=(vision.cap, frame_container, stop_event), daemon=True).start()
    displayer = DisplayThread(frame_container, stop_event, recording_status)
    displayer.start()

    # Recording setup
    recorder = RobotDataRecorder(agent, physics, vision)

    print("🤖 Robot Data Recorder Ready!")
    print("Instructions:")
    print("- Press 'r' in the camera window to START recording")
    print("- Press 's' in the camera window to STOP recording")
    print("- Press 'q' in the camera window to QUIT")
    print("- Manually guide the robot with your hands while recording")
    print("- Data will be recorded at ~1 Hz (once per second)")
    
    last_record_time = 0
    RECORD_INTERVAL = 1.0  # Record once per second
    
    try:
        while not stop_event.is_set():
            # Handle recording state changes
            if recording_status['start_recording']:
                recorder.start_recording()
                recording_status['recording'] = True
                recording_status['start_recording'] = False
                last_record_time = time.time()
                
            if recording_status['stop_recording']:
                recorder.stop_recording()
                recording_status['recording'] = False
                recording_status['stop_recording'] = False
            
            # Record data if recording and enough time has passed
            current_time = time.time()
            if (recording_status['recording'] and 
                frame_container.get('frame') is not None and
                current_time - last_record_time >= RECORD_INTERVAL):
                
                recorder.record_sample(frame_container['frame'])
                last_record_time = current_time
                
            time.sleep(0.1)  # Main loop sleep
            
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        if recording_status['recording']:
            print("Saving final recording...")
            recorder.stop_recording()
        stop_event.set()
        vision.cleanup()
        print("Cleanup complete. Goodbye!")

if __name__ == "__main__":
    main()

