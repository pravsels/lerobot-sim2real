#!/usr/bin/env python3
"""
planner.py

Glue code that uses the WebcamVisionPipeline and core IK/motion helpers
and runs vision, display, and planning in separate threads to keep
camera preview smooth while heavy planning executes asynchronously.

Enhanced with comprehensive IK feedback history to Gemini for adaptive planning
and successful demonstration examples for in-context learning.
"""
import os
import json
import time
import threading
import numpy as np
import glob
from pathlib import Path

import cv2
from PIL import Image
from google import genai
from google.genai import types
from dm_control.mujoco import Physics
from dm_control.mujoco.wrapper import mjbindings

from lerobot_sim2real.config.real_robot import create_real_robot
from mani_skill.agents.robots.lerobot.manipulator import LeRobotRealAgent
from webcam_gemini import WebcamVisionPipeline
from soarm_control import (
    qpos_from_site_pose,
    euler_to_quat,
    quat_to_euler,
    move_to_qpos,
    JOINT_LIMITS,
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

# Thread to display the camera at full speed
class DisplayThread(threading.Thread):
    def __init__(self, frame_container, stop_event):
        super().__init__(daemon=True)
        self.frame_container = frame_container
        self.stop_event = stop_event

    def run(self):
        while not self.stop_event.is_set():
            frame = self.frame_container.get('frame')
            if frame is not None:
                cv2.imshow('Live Camera', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.stop_event.set()
        cv2.destroyAllWindows()

class DemonstrationLoader:
    """Loads and manages demonstration data from annotated_data folder"""
    
    def __init__(self, annotated_data_path="annotated_data"):
        self.annotated_data_path = annotated_data_path
        self.demonstrations = []
        self.load_demonstrations()
    
    def load_demonstrations(self):
        """Load all demonstration sequences from annotated_data subfolders"""
        if not os.path.exists(self.annotated_data_path):
            print(f"Warning: {self.annotated_data_path} not found. No demonstrations loaded.")
            return
        
        # Find all subdirectories
        subdirs = [d for d in os.listdir(self.annotated_data_path) 
                  if os.path.isdir(os.path.join(self.annotated_data_path, d))]
        
        for subdir in sorted(subdirs):
            subdir_path = os.path.join(self.annotated_data_path, subdir)
            metadata_path = os.path.join(subdir_path, "metadata.json")
            
            if os.path.exists(metadata_path):
                try:
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                    
                    # Validate that image files exist
                    frames_with_images = []
                    for frame in metadata.get('frames', []):
                        frame_filename = frame.get('frame_filename')
                        if frame_filename:
                            image_path = os.path.join(subdir_path, frame_filename)
                            if os.path.exists(image_path):
                                frame['image_path'] = image_path
                                frames_with_images.append(frame)
                    
                    if frames_with_images:
                        demo = {
                            'sequence_name': subdir,
                            'frames': frames_with_images
                        }
                        self.demonstrations.append(demo)
                        print(f"Loaded demonstration '{subdir}' with {len(frames_with_images)} frames")
                
                except Exception as e:
                    print(f"Error loading demonstration from {subdir}: {e}")
        
        print(f"Total demonstrations loaded: {len(self.demonstrations)}")
    
    def format_demonstrations_for_prompt(self):
        """Format demonstrations as text for system prompt"""
        if not self.demonstrations:
            return "No demonstration examples available."
        
        demo_text = "SUCCESSFUL DEMONSTRATION EXAMPLES:\n\n"
        
        for demo_idx, demo in enumerate(self.demonstrations):
            demo_text += f"=== DEMONSTRATION {demo_idx + 1}: {demo['sequence_name']} ===\n"
            
            for frame in demo['frames']:
                demo_text += f"\nFrame {frame['frame_id']}:\n"
                demo_text += f"- Joint angles: {frame['joint_angles']}\n"
                demo_text += f"- End effector position: {frame['end_effector_position']}\n"
                demo_text += f"- End effector orientation: {frame['end_effector_orientation']}\n"
                
                if frame.get('objects'):
                    demo_text += f"- Objects detected:\n"
                    for obj in frame['objects']:
                        demo_text += f"  * {obj['label']}: bbox {obj['box_2d']}\n"
                else:
                    demo_text += f"- Objects detected: None\n"
            
            demo_text += "\n" + "="*50 + "\n\n"
        
        return demo_text

class NextPosePlanner:
    BASE_SYSTEM_PROMPT = (
        """
        You see a tabletop scene with our 6-DOF robot and various cubes.

        The robot is attached to a table. The position of the base is 0, 0, 0 in 3D space.

        You're provided images from a webcam that is placed to the left of the robot base.

        You're able to control the robot by providing end effector deltas.

        To move the end-effector up, the direction is +Z. To the the end-effector down, it is -Z.

        To move the end-effector left, relative to the base, is +X. To move it to the right, relative to the base, is -X.

        To move it straight ahead, it is +Y. and to move it backwards, it is -Y.

        Based on this image and the provided robot state and joint limits, 
        propose the next end-effector pose (X,Y,Z in meters; Roll,Pitch,Yaw in degrees) 
        to pick up the cube with the robot gripper and then place it into a box.

        IMPORTANT CONSIDERATIONS:
        - Prefer smaller incremental movements over large jumps
        - Consider the robot's current configuration when planning the next move
        - You've been given a scratchpad in the JSON response to do some reasoning. Please do some intense 3D reasoning before you give me your end effector delta.
        - You should attempt to move the gripper open position 10% at a time
        - You must not give up
        - If the inverse kinematics for your proposed end effector deltas fails, we will inform you of this and you must propose new deltas. The previous deltas you provided are discarded. Sometimes increasing the magnitude of changes works better.
        - The end effector rotation is controlled in Euler angles in degrees
        - End effector deltas of around 0.1 are good for the positional deltas
        - The gripper open percentage is not a delta but a value between 0 and 100 inclusive. At 100, the gripper is fully open, at 0, the gripper is fully closed.
        - You should describe the image you see at the start of your reasoning trace, and everything you see in it.

        {demonstrations}

        Respond with your reasoning and with JSON: {{"image_description": (string), "reasoning": (string), "next_eef_delta" : [x, y, z, roll, pitch, yaw], "gripper_open_percent": (int) }}
        """
    )

    def __init__(self, agent, physics, vision, api_key, model_id="gemini-2.0-flash-lite", temperature=0.2, annotated_data_path="annotated_data"):
        self.agent = agent
        self.physics = physics
        self.vision = vision
        self.client = genai.Client(api_key=api_key)
        self.model_id = model_id
        self.temperature = temperature
        self.conversation_history = []  # Full conversation history
        self.max_history = 50  # Keep last 50 exchanges
        self.ik_tolerance = 2e-4  # Store the IK tolerance for feedback
        self.gripper_percent = 100
        self.last_successful_pose = [0.011, -0.208, 0.102, 81.2, -90, -5] 
        self.first_action = True
        
        # Load demonstrations
        self.demo_loader = DemonstrationLoader(annotated_data_path)
        self.system_prompt = self._build_system_prompt()

    def _build_system_prompt(self):
        """Build system prompt with demonstrations included"""
        demonstrations = self.demo_loader.format_demonstrations_for_prompt()
        return self.BASE_SYSTEM_PROMPT.format(demonstrations=demonstrations)

    def get_robot_state(self) -> dict:
        q = self.agent.qpos.squeeze().cpu().numpy().tolist()
        mj = mjbindings.mjlib
        mj.mj_fwdPosition(self.physics.model.ptr, self.physics.data.ptr)
        pos = self.physics.named.data.site_xpos['gripper'].tolist()
        mat = self.physics.named.data.site_xmat['gripper']
        quat = np.empty(4, dtype=np.float64)
        mj.mju_mat2Quat(quat, mat)
        rpy = np.rad2deg(quat_to_euler(quat)).tolist()
        return {"joints": q, "eef_pose": pos + rpy, "joint_limits": JOINT_LIMITS}

    def build_conversation_contents(self, frame: np.ndarray, state: dict) -> list:
        """Build the full conversation contents including history and current request"""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb)
        
        # Start with the current request
        contents = [
            pil_image,
            json.dumps({
                "robot": state,
                "ik_tolerance": self.ik_tolerance,
                "request": "Please propose the next end-effector delta for pick-and-place task"
            }, indent=2)
        ]
        
        # Add conversation history (alternating user context and assistant responses)
        for entry in self.conversation_history:
            if entry["type"] == "user_feedback":
                contents.append(f"FEEDBACK: {json.dumps(entry['content'], indent=2)}")
            elif entry["type"] == "assistant_response":
                contents.append(f"YOUR_PREVIOUS_RESPONSE: {entry['content']}")
        
        return contents

    def add_feedback_to_history(self, proposed_pose: list, ik_result: dict):
        """Add IK feedback to conversation history"""
        feedback_entry = {
            "type": "user_feedback",
            "content": {
                "current_pose": self.last_successful_pose,
                "proposed_pose": proposed_pose,
                "ik_result": {
                    "success": ik_result["success"],
                    "error_norm": ik_result["error_norm"],
                    "ik_steps": ik_result["steps"],
                    "tolerance": self.ik_tolerance,
                    "analysis": (
                        "SUCCESS: Pose was reachable and executed" if ik_result["success"]
                        else f"FAILURE: IK error {ik_result['error_norm']:.6f} > tolerance {self.ik_tolerance:.6f}"
                    )
                }
            },
            "timestamp": time.time()
        }
        
        self.conversation_history.append(feedback_entry)
        
        # Trim history if too long
        if len(self.conversation_history) > self.max_history * 2:  # *2 because we have feedback + response pairs
            self.conversation_history = self.conversation_history[-self.max_history * 2:]
        
        success_str = "SUCCESS" if ik_result["success"] else "FAILURE"
        print(f"Added {success_str} feedback to history: pose={proposed_pose}, error_norm={ik_result['error_norm']:.6f}")

    def add_response_to_history(self, response_text: str):
        """Add assistant's response to conversation history"""
        response_entry = {
            "type": "assistant_response", 
            "content": response_text,
            "timestamp": time.time()
        }
        self.conversation_history.append(response_entry)

    def propose_next_pose(self, frame: np.ndarray) -> list:
        state = self.get_robot_state()
        contents = self.build_conversation_contents(frame, state)

        response = self.client.models.generate_content(
            model=self.model_id,
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=self.system_prompt,  # Now includes demonstrations
                temperature=self.temperature,
            )
        )
        raw = response.text

        print('RAW response : ', raw)
        
        # Add the response to history
        self.add_response_to_history(raw)

        # Clean and parse JSON
        json_text = self.vision._clean_json_response(raw)
        print('cleaned json : ', json_text)
        data = json.loads(json_text)
        pose = data.get("next_eef_delta")
        if not (isinstance(pose, list) and len(pose) == 6):
            raise RuntimeError(f"Invalid pose format: {pose}")

        self.gripper_percent = data.get("gripper_open_percent")

        return pose

    def plan_and_execute(self, frame: np.ndarray) -> None:
        try:
            proposed_eef_delta = self.propose_next_pose(frame)
            
            current_eef_pose = self.get_robot_state()['eef_pose']
            proposed_pose = np.asarray(current_eef_pose) + np.asarray(proposed_eef_delta)
            proposed_pose = [float(x) for x in list(proposed_pose)]

            if self.first_action:
                self.first_action = False
                proposed_pose = [0, -0.1, 0.2, 0,0, -90]
                proposed_pose = self.last_successful_pose 

            x, y, z, roll, pitch, yaw = proposed_pose
            print(f"Gemini suggests next pose: {proposed_pose}")
        except Exception as e:
            print(f"Planning error: {e}")
            return

        # Attempt IK solving
        target_quat = euler_to_quat(
            np.deg2rad(roll), np.deg2rad(pitch), np.deg2rad(yaw)
        )
        self.physics.reset()
        res = qpos_from_site_pose(
            self.physics, "gripper",
            target_pos=np.array([x, y, z], dtype=np.float64),
            target_quat=target_quat,
            tol=self.ik_tolerance,
            rot_weight=0.1
        )
        
        # Prepare IK result for feedback
        ik_result = {
            "success": res.success,
            "error_norm": res.err_norm,
            "steps": res.steps
        }
        
        # Add feedback to conversation history
        self.add_feedback_to_history(proposed_pose, ik_result)
        
        if not res.success:
            print(f"IK failed: err_norm={res.err_norm:.6f}, steps={res.steps}, tolerance={self.ik_tolerance:.6f}")
            proposed_pose = self.last_successful_pose
            return
        else:
            print(f"IK succeeded: err_norm={res.err_norm:.6f}, steps={res.steps}")
            self.last_successful_pose = proposed_pose

        # Set gripper position
        maxv = 1.7
        minv = -0.17
        res.qpos[5] = minv + (self.gripper_percent / 100) * (maxv - minv)

        # Execute successful pose
        print(f"Executing pose with IK solution")
        threading.Thread(
            target=move_to_qpos,
            args=(self.agent, res.qpos),
            daemon=True
        ).start()

        time.sleep(1)

    def print_history_summary(self):
        """Print a summary of the conversation history for debugging"""
        if not self.conversation_history:
            print("No conversation history yet")
            return
            
        successes = sum(1 for entry in self.conversation_history 
                       if entry["type"] == "user_feedback" and entry["content"]["ik_result"]["success"])
        failures = sum(1 for entry in self.conversation_history 
                      if entry["type"] == "user_feedback" and not entry["content"]["ik_result"]["success"])
        
        print(f"History summary: {successes} successes, {failures} failures, {len(self.conversation_history)} total entries")
        print(f"Demonstrations loaded: {len(self.demo_loader.demonstrations)} sequences")


def main():
    os.chdir("lerobot_sim2real/scripts")
    load_dotenv()

    # Robot setup
    uid = "so101"
    real = create_real_robot(uid)
    real.connect()
    agent = LeRobotRealAgent(real)
    agent.reset(RESET_Q)
    print('starting with home joint pos : ', RESET_Q)
    physics = Physics.from_xml_string(open("scene.xml").read())

    # Vision setup
    vision = WebcamVisionPipeline(camera_index=4)
    if not vision.initialize_camera():
        return

    # Shared frame container and stop event
    frame_container = { 'frame': None }
    stop_event = threading.Event()

    # Start capture and display threads
    threading.Thread(target=capture_thread, args=(vision.cap, frame_container, stop_event), daemon=True).start()
    displayer = DisplayThread(frame_container, stop_event)
    displayer.start()

    # Planning setup with demonstrations
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("Error: GOOGLE_API_KEY not set")
        stop_event.set()
        return
    
    # Pass the annotated_data path (modify as needed)
    # import pdb; pdb.set_trace()
    planner = NextPosePlanner(agent, physics, vision, api_key, 
                              annotated_data_path="recorded_data/annotated_data")

    print("Starting pick-and-place planning loop with comprehensive IK feedback history and demonstration examples.")
    print("Close the window or press 'q' to exit.")
    
    iteration_count = 0
    try:
        while not stop_event.is_set():
            frame = frame_container.get('frame')
            if frame is not None:
                iteration_count += 1
                print(f"\n--- Planning Iteration {iteration_count} ---")
                planner.plan_and_execute(frame)
                
                # Print history summary every 10 iterations
                if iteration_count % 10 == 0:
                    planner.print_history_summary()
                    
            time.sleep(0.05)
    except KeyboardInterrupt:
        pass
    finally:
        print("\nFinal history summary:")
        planner.print_history_summary()
        stop_event.set()
        vision.cleanup()

if __name__ == "__main__":
    main()