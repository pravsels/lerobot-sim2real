# core.py
"""
Contains inverse-kinematics utilities and motion helper for LeRobot.
"""
import numpy as np
import collections
import time
from dm_control.mujoco import Physics
from dm_control.mujoco.wrapper import mjbindings
from typing import Optional

# Constants
JOINT_LIMITS = [(-1.92, 1.92), (-1.75, 1.75), (-1.75, 1.57),
                (-1.66, 1.66), (-2.79, 2.79), (-0.17, 1.75)]
RESET_Q = np.array([0.0054, -0.0069, 0.0069, 1.598, 1.5789, 0.0177], dtype=np.float32)

# IK result tuple
IKResult = collections.namedtuple("IKResult", ["qpos", "err_norm", "steps", "success"])


def nullspace_method(jac: np.ndarray, delta: np.ndarray, regularization_strength: float = 0.0) -> np.ndarray:
    H = jac.T @ jac
    g = jac.T @ delta
    if regularization_strength:
        H += np.eye(H.shape[0]) * regularization_strength
        return np.linalg.solve(H, g)
    return np.linalg.lstsq(H, g, rcond=None)[0]


def qpos_from_site_pose(
    physics: Physics,
    site_name: str,
    target_pos: Optional[np.ndarray] = None,
    target_quat: Optional[np.ndarray] = None,
    tol: float = 1e-5,
    rot_weight: float = 0.1,
    reg_thresh: float = 0.1,
    reg_strength: float = 3e-2,
    max_norm: float = 2.0,
    max_steps: int = 100
) -> IKResult:
    sim = physics.copy(share_model=True)
    mj = mjbindings.mjlib
    site_id = sim.model.name2id(site_name, 'site')

    for i in range(max_steps):
        mj.mj_fwdPosition(sim.model.ptr, sim.data.ptr)
        err, jac, idx = [], [], 0
        if target_pos is not None:
            pos = sim.named.data.site_xpos[site_name]
            e = target_pos - pos
            err.extend(e)
        if target_quat is not None:
            mat = sim.named.data.site_xmat[site_name]
            q = np.empty(4); nq = np.empty(4); vel = np.empty(3)
            mj.mju_mat2Quat(q, mat)
            mj.mju_negQuat(nq, q); mj.mju_mulQuat(q, target_quat, nq)
            mj.mju_quat2Vel(vel, q, 1)
            err.extend((vel * rot_weight))
        err = np.array(err)
        if np.linalg.norm(err) < tol:
            return IKResult(sim.data.qpos.copy(), np.linalg.norm(err), i, True)

        jac = np.empty((len(err), sim.model.nv), dtype=err.dtype)
        mj.mj_jacSite(sim.model.ptr, sim.data.ptr,
                      jac[:3] if target_pos is not None else None,
                      jac[3:] if target_quat is not None else None,
                      site_id)
        reg = reg_strength if np.linalg.norm(err) > reg_thresh else 0.0
        upd = nullspace_method(jac, err, reg)
        norm = np.linalg.norm(upd)
        if norm > max_norm:
            upd *= max_norm / norm
        sim.data.qpos[:] += upd

    return IKResult(sim.data.qpos.copy(), np.linalg.norm(err), max_steps, False)


def euler_to_quat(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr, sr = np.cos(roll/2), np.sin(roll/2)
    cp, sp = np.cos(pitch/2), np.sin(pitch/2)
    cy, sy = np.cos(yaw/2), np.sin(yaw/2)
    return np.array([cr*cp*cy + sr*sp*sy,
                     sr*cp*cy - cr*sp*sy,
                     cr*sp*cy + sr*cp*sy,
                     cr*cp*sy - sr*sp*cy], dtype=np.float64)


def quat_to_euler(q: np.ndarray) -> np.ndarray:
    w,x,y,z = q
    roll = np.arctan2(2*(w*x+y*z), 1-2*(x*x+y*y))
    pitch = np.arcsin(np.clip(2*(w*y-z*x), -1, 1))
    yaw = np.arctan2(2*(w*z+x*y), 1-2*(y*y+z*z))
    return np.array([roll, pitch, yaw], dtype=np.float64)


def move_to_qpos(agent, target_qpos: np.ndarray, freq: int =30, max_step: float =0.025, timeout: float =20.0):
    start = time.perf_counter()
    target = target_qpos.astype(np.float32)
    current = agent.qpos.squeeze().cpu().numpy()
    while True:
        delta = np.clip(target-current, -max_step, max_step)
        if np.linalg.norm(delta)<1e-4: break
        current += delta; agent.set_target_qpos(current)
        time.sleep(max(0, 1/freq - (time.perf_counter()-start)))
        if time.perf_counter()-start>timeout: break

