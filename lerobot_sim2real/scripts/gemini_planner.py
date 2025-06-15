#!/usr/bin/env python3
"""
planner.py

Glue code that uses the WebcamVisionPipeline and core IK/motion helpers
and runs vision, display, and planning in separate threads to keep
camera preview smooth while heavy planning executes asynchronously.

Enhanced with comprehensive IK feedback history to Gemini for adaptive planning.
"""
import os
import json
import time
import threading
import numpy as np

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

class NextPosePlanner:
    SYSTEM_PROMPT = (
        "You see a tabletop scene with our 6-DOF robot and various objects. "
        "The robot has these joints (in order): "
        "1. shoulder_pan (joint 1) - rotates the arm left/right "
        "2. shoulder_lift (joint 2) - lifts the arm up/down "
        "3. elbow_flex (joint 3) - bends the elbow "
        "4. wrist_flex (joint 4) - bends the wrist up/down "
        "5. wrist_roll (joint 5) - rotates the wrist "
        "6. gripper (joint 6) - opens/closes the gripper "
        "\n"
        "Based on this image and the provided robot state and joint limits, "
        "propose the next end-effector pose (X,Y,Z in meters; Roll,Pitch,Yaw in degrees) "
        "to grasp a cube and then place it into a box. "
        "\n"
        "IMPORTANT CONSIDERATIONS: "
        "- Check the current joint values against the joint limits to avoid impossible poses "
        "- Shoulder_pan limits how far left/right the arm can reach "
        "- Shoulder_lift + elbow_flex determine the arm's reach and height "
        "- Wrist joints affect end-effector orientation - avoid extreme rotations "
        "- Learn from the IK feedback history provided - avoid poses that previously failed "
        "- IK tolerance is typically 1e-5, so error_norm should be much smaller than this "
        "- Prefer smaller incremental movements over large jumps "
        "- Consider the robot's current configuration when planning the next move "
        "\n"
        "Respond ONLY with JSON: { \"next_eef_pose\" : [x, y, z, roll, pitch, yaw] }"
    )

    def __init__(self, agent, physics, vision, api_key, model_id="gemini-2.0-flash-lite-001", temperature=0.2):
        self.agent = agent
        self.physics = physics
        self.vision = vision
        self.client = genai.Client(api_key=api_key)
        self.model_id = model_id
        self.temperature = temperature
        self.conversation_history = []  # Full conversation history
        self.max_history = 50  # Keep last 50 exchanges
        self.ik_tolerance = 2e-4  # Store the IK tolerance for feedback

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
                "request": "Please propose the next end-effector pose for pick-and-place task"
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
                system_instruction=self.SYSTEM_PROMPT,
                temperature=self.temperature,
            )
        )
        raw = response.text
        
        # Add the response to history
        self.add_response_to_history(raw)

        # Clean and parse JSON
        json_text = self.vision._clean_json_response(raw)
        data = json.loads(json_text)
        pose = data.get("next_eef_pose")
        if not (isinstance(pose, list) and len(pose) == 6):
            raise RuntimeError(f"Invalid pose format: {pose}")
        return pose

    def plan_and_execute(self, frame: np.ndarray) -> None:
        try:
            proposed_pose = self.propose_next_pose(frame)
            x, y, z, roll, pitch, yaw = proposed_pose
            print(f"Gemini suggests next pose: {proposed_pose}")
        except Exception as e:
            print(f"Planning error: {e}")
            return

        # Attempt IK solving
        target_quat = euler_to_quat(
            np.deg2rad(roll), np.deg2rad(pitch), np.deg2rad(yaw)
        )
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
            return
        else:
            print(f"IK succeeded: err_norm={res.err_norm:.6f}, steps={res.steps}")

        # Execute successful pose
        print(f"Executing pose with IK solution")
        threading.Thread(
            target=move_to_qpos,
            args=(self.agent, res.qpos),
            daemon=True
        ).start()

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

    # Planning setup
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("Error: GOOGLE_API_KEY not set")
        stop_event.set()
        return
    planner = NextPosePlanner(agent, physics, vision, api_key)

    print("Starting pick-and-place planning loop with comprehensive IK feedback history.")
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
                    
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    finally:
        print("\nFinal history summary:")
        planner.print_history_summary()
        stop_event.set()
        vision.cleanup()

if __name__ == "__main__":
    main()