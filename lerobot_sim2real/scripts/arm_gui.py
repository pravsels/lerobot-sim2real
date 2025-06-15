# gui.py
"""
Contains a streamlined, attractive Tkinter GUI for LeRobot.
"""
import threading
import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
from dm_control.mujoco import Physics
from dm_control.mujoco.wrapper import mjbindings
import tyro
from dataclasses import dataclass

from soarm_control import (qpos_from_site_pose, euler_to_quat, quat_to_euler,
                  JOINT_LIMITS, RESET_Q, move_to_qpos)
from lerobot_sim2real.config.real_robot import create_real_robot
from mani_skill.agents.robots.lerobot.manipulator import LeRobotRealAgent

import os; os.chdir("lerobot_sim2real/scripts")

class JointControlApp(ttk.Frame):
    def __init__(self, master, uid: str="so101"):
        super().__init__(master, padding=10)
        self.sliders = []  # initialize slider list
        self.master, self.uid = master, uid
        super().__init__(master, padding=10)
        self.master, self.uid = master, uid
        self.agent = self.physics = None
        self._init_style()
        self._build_ui()
        self.master.title("LeRobot Controller")
        self.pack(fill="both", expand=True)
        # Allow window resizing
        self.master.columnconfigure(0, weight=1)
        self.master.rowconfigure(0, weight=1)

    def _init_style(self):
        style = ttk.Style(self.master)
        style.theme_use('clam')
        style.configure('TFrame', padding=5)
        style.configure('TButton', padding=6)
        style.configure('TLabel', padding=4)

    def _build_ui(self):
        # Top toolbar
        toolbar = ttk.Frame(self)
        toolbar.pack(fill='x')
        ttk.Button(toolbar, text="Connect", command=self._connect).pack(side='left')
        ttk.Button(toolbar, text="Reset Home", command=self._reset).pack(side='left')
        ttk.Button(toolbar, text="Quit", command=self.master.quit).pack(side='right')

        container = ttk.Frame(self)
        container.pack(fill='both', expand=True, pady=10)

        # Configure resize behavior
        container.columnconfigure(0, weight=1)
        container.columnconfigure(1, weight=1)
        container.rowconfigure(0, weight=1)

        # Left: Live State
        live = ttk.LabelFrame(container, text="Live State")
        live.grid(row=0, column=0, sticky='nsew', padx=5, pady=5)
        self._make_readouts(live)

        # Right: Command Panel
        cmd = ttk.LabelFrame(container, text="Command Panel")
        cmd.grid(row=0, column=1, sticky='nsew', padx=5, pady=5)
        self._make_commands(cmd)

    def _make_readouts(self, frame):
        self.joint_vars = [tk.StringVar() for _ in range(6)]
        for i, var in enumerate(self.joint_vars, 1):
            ttk.Label(frame, text=f"J{i}:").grid(row=i-1, column=0, padx=2, pady=2, sticky='e')
            ttk.Label(frame, textvariable=var, width=8, relief='sunken').grid(row=i-1, column=1, padx=2, pady=2, sticky='w')

        self.pose_vars = {k: tk.StringVar() for k in ['X','Y','Z','Roll','Pitch','Yaw']}
        for i,(k,v) in enumerate(self.pose_vars.items()):
            ttk.Label(frame, text=f"{k}:").grid(row=i, column=2, padx=2, pady=2, sticky='e')
            ttk.Label(frame, textvariable=v, width=8, relief='sunken').grid(row=i, column=3, padx=2, pady=2, sticky='w')

    def _make_commands(self, frame):
        entries = {}
        for idx, lbl in enumerate(['X (m)','Y (m)','Z (m)','Roll°','Pitch°','Yaw°']):
            ttk.Label(frame, text=lbl).grid(row=idx, column=0, sticky='e', padx=4, pady=2)
            e = ttk.Entry(frame, width=10)
            e.grid(row=idx, column=1, sticky='ew', padx=4, pady=2)
            entries[lbl.split()[0].lower()] = e
        ttk.Button(frame, text="Compute IK", command=lambda: self._compute_ik(entries)).grid(row=6, column=0, columnspan=2, pady=8)

        # Make frame columns stretch
        frame.columnconfigure(1, weight=1)
        for i,(mn,mx) in enumerate(JOINT_LIMITS,1):
            s = ttk.Scale(frame, from_=mn, to=mx, orient='horizontal')
            s.grid(row=6+i, column=0, columnspan=2, sticky='ew', pady=4)
            self.sliders.append(s)
        ttk.Button(frame, text="Apply Sliders", command=self._apply_sliders).grid(row=13, column=0, columnspan=2, pady=8)

    def _connect(self):
        try:
            r = create_real_robot(self.uid); r.connect()
            self.agent = LeRobotRealAgent(r)
            xml = open('scene.xml').read(); self.physics = Physics.from_xml_string(xml)
            self.after(100, self._poll)
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def _poll(self):
        q = self.agent.qpos.squeeze().cpu().numpy()
        for var,val in zip(self.joint_vars, q): var.set(f"{val:.3f}")
        self.physics.data.qpos[:6] = q
        mjbindings.mjlib.mj_fwdPosition(self.physics.model.ptr, self.physics.data.ptr)
        pos = self.physics.named.data.site_xpos['gripper']; quat = np.empty(4)
        mjbindings.mjlib.mju_mat2Quat(quat, self.physics.named.data.site_xmat['gripper'])
        rpy = quat_to_euler(quat) * (180/np.pi)
        for k,val in zip(self.pose_vars, np.hstack([pos, rpy])): self.pose_vars[k].set(f"{val:.2f}")
        self.after(200, self._poll)

    def _compute_ik(self, entries):
        try:
            tp = np.array([float(entries[k].get()) for k in ['x','y','z']], float)
            rq = euler_to_quat(*np.deg2rad([float(entries[k].get()) for k in ['roll','pitch','yaw']]))
        except ValueError:
            return messagebox.showerror("Input Error","Invalid numbers.")
        res = qpos_from_site_pose(self.physics, 'gripper', tp, rq)
        if res.success:
            for s,v in zip(self.sliders, res.qpos): s.set(v)
            messagebox.showinfo("IK","Converged")
        else:
            messagebox.showwarning("IK","Failed to converge")

    def _apply_sliders(self):
        tgt = np.array([s.get() for s in self.sliders], float)
        threading.Thread(target=move_to_qpos, args=(self.agent, tgt), daemon=True).start()

    def _reset(self):
        for s,v in zip(self.sliders, RESET_Q): s.set(v)
        threading.Thread(target=move_to_qpos, args=(self.agent, RESET_Q), daemon=True).start()

@dataclass
class CliArgs:
    uid: str = "so101"

if __name__ == "__main__":
    args = tyro.cli(CliArgs)
    root = tk.Tk(); JointControlApp(root, args.uid); root.mainloop()
