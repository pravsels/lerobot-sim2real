#!/usr/bin/env python3
"""
planner.py

Glue code that uses the WebcamVisionPipeline and core IK/motion helpers
and runs vision, display, and planning in separate threads to keep
camera preview smooth while heavy planning executes asynchronously.
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
    JOINT_LIMITS
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
        "Based on this image and the provided robot state and joint limits, "
        "propose the next end-effector pose (X,Y,Z in meters; Roll,Pitch,Yaw in degrees) "
        "to grasp a cube and then place it into a box. "
        "Respond ONLY with JSON: { \"next_eef_pose\" : [x, y, z, roll, pitch, yaw] }"
    )

    def __init__(self, agent, physics, vision, api_key, model_id="gemini-2.0-flash-lite-001", temperature=0.2):
        self.agent = agent
        self.physics = physics
        self.vision = vision
        self.client = genai.Client(api_key=api_key)
        self.model_id = model_id
        self.temperature = temperature

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

    def build_user_context(self, state: dict) -> str:
        return json.dumps({"robot": state})

    def propose_next_pose(self, frame: np.ndarray) -> list:
        state = self.get_robot_state()
        context = self.build_user_context(state)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb)

        response = self.client.models.generate_content(
            model=self.model_id,
            contents=[pil_image, context],
            config=types.GenerateContentConfig(
                system_instruction=self.SYSTEM_PROMPT,
                temperature=self.temperature,
            )
        )
        raw = response.text

        # Clean and parse JSON
        json_text = self.vision._clean_json_response(raw)
        data = json.loads(json_text)
        pose = data.get("next_eef_pose")
        if not (isinstance(pose, list) and len(pose) == 6):
            raise RuntimeError(f"Invalid pose format: {pose}")
        return pose

    def plan_and_execute(self, frame: np.ndarray) -> None:
        try:
            x, y, z, roll, pitch, yaw = self.propose_next_pose(frame)
            print(f"Gemini suggests next pose: {[x, y, z, roll, pitch, yaw]}")
        except Exception as e:
            print(f"Planning error: {e}")
            return

        target_quat = euler_to_quat(
            np.deg2rad(roll), np.deg2rad(pitch), np.deg2rad(yaw)
        )
        res = qpos_from_site_pose(
            self.physics, "gripper",
            target_pos=np.array([x, y, z], dtype=np.float64),
            target_quat=target_quat
        )
        if not res.success:
            print(f"IK failed: err_norm={res.err_norm}, steps={res.steps}")
            return

        threading.Thread(
            target=move_to_qpos,
            args=(self.agent, res.qpos),
            daemon=True
        ).start()


def main():
    os.chdir("lerobot_sim2real/scripts")
    load_dotenv()

    # Robot setup
    uid = "so101"
    real = create_real_robot(uid)
    real.connect()
    agent = LeRobotRealAgent(real)
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

    print("Starting pick-and-place planning loop. Close the window or press 'q' to exit.")
    try:
        while not stop_event.is_set():
            frame = frame_container.get('frame')
            if frame is not None:
                planner.plan_and_execute(frame)
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    finally:
        stop_event.set()
        vision.cleanup()

if __name__ == "__main__":
    main()
