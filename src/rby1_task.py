# SPDX-FileCopyrightText: Copyright (c) 2020-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Isaac Sim ``BaseTask`` implementing the RBY1 simulation loop (model M/A)."""
from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
from isaacsim.core.api.scenes.scene import Scene
from isaacsim.core.api.tasks import BaseTask
from isaacsim.core.utils.rotations import quat_to_rot_matrix
from isaacsim.core.utils.stage import add_reference_to_stage

from rby1_controller import PDController
from rby1_robot import RBY1Robot
from rby1_udp_bridge import RBY1UdpBridge
from sim_gripper_bridge import (
    SimGripperServer,
    closeness_to_finger_meter,
    finger_meter_to_closeness,
)


# ============================================================
# Per-model configuration
# ============================================================

@dataclass(frozen=True)
class RBY1ModelConfig:
    model: str
    usd_file_name: str
    cpp_joint_names: tuple[str, ...]
    joint_kp: tuple[float, ...]
    joint_kd: tuple[float, ...]
    mobility_dof: int
    reference_wheel_target: tuple[float, ...]


_BODY_JOINT_NAMES = (
    "torso_0", "torso_1", "torso_2", "torso_3", "torso_4", "torso_5",
    "right_arm_0", "right_arm_1", "right_arm_2", "right_arm_3", "right_arm_4", "right_arm_5", "right_arm_6",
    "left_arm_0", "left_arm_1", "left_arm_2", "left_arm_3", "left_arm_4", "left_arm_5", "left_arm_6",
    "head_0", "head_1",
)

_BODY_JOINT_KP = (
    3911.0, 3911.0, 3911.0, 573.6, 573.6, 573.6,
    208.6, 208.6, 208.6, 91.27, 39.11, 39.11, 39.11,
    208.6, 208.6, 208.6, 91.27, 39.11, 39.11, 39.11,
    39.11, 39.11,
)

_BODY_JOINT_KD = (
    3520.0, 3520.0, 3520.0, 1043.0, 1043.0, 1043.0,
    521.5, 521.5, 321.5, 208.6, 91.26, 91.26, 61.26,
    521.5, 521.5, 321.5, 208.6, 91.26, 91.26, 61.26,
    91.26, 91.26,
)

RBY1_MODEL_CONFIGS = {
    "a": RBY1ModelConfig(
        model="a",
        usd_file_name="model_v_1_2_a.usd",
        cpp_joint_names=("right_wheel", "left_wheel", *_BODY_JOINT_NAMES),
        joint_kp=(100.0, 100.0, *_BODY_JOINT_KP),
        joint_kd=(10.0, 10.0, *_BODY_JOINT_KD),
        mobility_dof=2,
        reference_wheel_target=(-0.5, -0.5),
    ),
    "m": RBY1ModelConfig(
        model="m",
        usd_file_name="model_v_1_2_m_rev.usd",
        cpp_joint_names=("wheel_fr", "wheel_fl", "wheel_rr", "wheel_rl", *_BODY_JOINT_NAMES),
        joint_kp=(100.0, 100.0, 100.0, 100.0, *_BODY_JOINT_KP),
        joint_kd=(10.0, 10.0, 10.0, 10.0, *_BODY_JOINT_KD),
        mobility_dof=4,
        reference_wheel_target=(0.5, -0.5, -0.5, 0.5),
    ),
}

JOINT_VELOCITY_LPF_ALPHA = 0.5


def normalize_robot_model(robot_model: str) -> str:
    model = str(robot_model).lower()
    if model not in RBY1_MODEL_CONFIGS:
        valid_models = ", ".join(sorted(RBY1_MODEL_CONFIGS))
        raise ValueError(f"Unsupported RBY1 model '{robot_model}'. Valid models: {valid_models}")
    return model


def _resolve_usd_path(usd_file_name: str) -> str:
    """Locate the RBY1 USD asset for the selected model.

    Priority:
      1. ``RBY1_USD_PATH`` environment variable
      2. ``<repo>/assets/<model usd>``
      3. ``<src/>/<model usd>`` (legacy fallback)
    """
    env_path = os.environ.get("RBY1_USD_PATH")
    if env_path:
        return env_path
    src_dir = os.path.dirname(os.path.abspath(__file__))
    repo_assets = os.path.join(os.path.dirname(src_dir), "assets", usd_file_name)
    if os.path.isfile(repo_assets):
        return repo_assets
    return os.path.join(src_dir, usd_file_name)


# ============================================================
# Task definition
# ============================================================

class RBY1Task(BaseTask):
    """Drives the RBY1 articulation from external UDP commands (or a built-in test trajectory)."""

    GRIPPER_JOINT_NAMES = ["gripper_finger_l1", "gripper_finger_r1"]

    def __init__(
        self,
        udp_bridge: Optional[RBY1UdpBridge] = None,
        robot_model: str = "m",
        gripper_server: Optional[SimGripperServer] = None,
    ):
        super().__init__(name="rby1_task", offset=None)
        self.robot_model = normalize_robot_model(robot_model)
        self.model_config = RBY1_MODEL_CONFIGS[self.robot_model]
        self.robot = None
        self.robot_prim_path = "/World/RBY1"
        self.usd_path = _resolve_usd_path(self.model_config.usd_file_name)
        self.step_counter = 0
        self.udp_bridge = udp_bridge          # None -> standalone mode (no UDP)
        self.gripper_server = gripper_server  # optional sim gripper bridge
        self.force_ui = None                  # optional ExternalForceUI

        self._last_command_seq = 0
        self._new_command_count_since_log = 0
        self._command_wait_timeout: Optional[float] = None
        self._command_wait_miss_count_since_log = 0
        self._state_sent_once = False
        self._command_stream_started = False
        self._last_pre_step_sim_time: Optional[float] = None
        self._prev_diff_position: Optional[np.ndarray] = None
        self._prev_diff_time: Optional[float] = None
        self._joint_velocity_lpf_alpha = JOINT_VELOCITY_LPF_ALPHA
        self._filtered_joint_velocity: Optional[np.ndarray] = None
        self._joint_indices_np: Optional[np.ndarray] = None
        self._ready_flags: Optional[np.ndarray] = None
        self._zero_ctrl_state: Optional[np.ndarray] = None
        self._state_q: Optional[np.ndarray] = None
        self._state_dq: Optional[np.ndarray] = None
        self._state_tau: Optional[np.ndarray] = None
        self._efforts_full: Optional[np.ndarray] = None
        self._external_force_error_logged = False
        self._last_log_wall_time = time.monotonic()
        self._last_log_sim_time: Optional[float] = None

    # ------------------------------------------------------------------
    # Scene setup
    # ------------------------------------------------------------------

    def set_up_scene(self, scene: Scene) -> None:
        """Spawn the ground plane and the RBY1 robot in the world."""
        super().set_up_scene(scene)

        ground_plane = scene.add_default_ground_plane(
            name="default_ground_plane",
            prim_path="/World/defaultGroundPlane",
            static_friction=0.0,
            dynamic_friction=0.0,
            restitution=0.0,
        )
        ground_plane._collision_prim.set_contact_offset(0.002)
        ground_plane._collision_prim.set_rest_offset(0.0)

        print(f"[RBY1Task] model={self.robot_model}, usd_path={self.usd_path}")
        add_reference_to_stage(usd_path=self.usd_path, prim_path=self.robot_prim_path)

        self.rby_robot = RBY1Robot(
            prim_path=self.robot_prim_path,
            name="RBY1",
            position=np.array([0.0, 0.0, 0.0]),
            orientation=np.array([1.0, 0.0, 0.0, 0.0]),
            robot_model=self.robot_model,
        )
        self.robot = scene.add(self.rby_robot)
        self.rby_robot.set_joint_properties(scene.stage)
        self.rby_robot.set_properties(scene.stage)
        # Self-collision left disabled — enabling it with the current USD collision
        # geometry causes joint instability. A USD collision fix is planned.
        self.robot.set_enabled_self_collisions(False)
        self.robot.set_solver_position_iteration_count(4)
        self.robot.set_solver_velocity_iteration_count(1)

    # ------------------------------------------------------------------
    # Physics step
    # ------------------------------------------------------------------

    def pre_step(self, time_step_index: int, simulation_time: float) -> None:
        """Callback invoked before each PhysX step."""
        self.step_counter += 1
        self._last_pre_step_sim_time = simulation_time
        if self._last_log_sim_time is None:
            self._last_log_sim_time = simulation_time

        if self.udp_bridge is None:
            # Standalone mode: drive a built-in test trajectory (C++ DOF order).
            self._update_full_state(simulation_time, update_velocity=True)
            velocity_mode, target = self._build_reference_target(simulation_time)
            self.pd_controller.update_target(target, velocity_mode)
        else:
            # UDP mode: consume the latest command (waiting for a fresh one once the stream started).
            if self._state_sent_once and self._command_stream_started:
                cmd, new_command_received, command_seq = self.udp_bridge.wait_for_command_after(
                    self._last_command_seq,
                    timeout=self._command_wait_timeout,
                )
            else:
                cmd, new_command_received, command_seq = self.udp_bridge.get_latest_command_with_status(
                    self._last_command_seq
                )
            if new_command_received and cmd is not None:
                self._command_stream_started = True
                self._new_command_count_since_log += command_seq - self._last_command_seq
                self._last_command_seq = command_seq
                self._apply_udp_command(cmd)
            elif self._state_sent_once and self._command_stream_started:
                self._command_wait_miss_count_since_log += 1

        # External PD torque → max_efforts clipping → expand to the full Isaac Sim DOF buffer.
        self.robot.set_joint_efforts(self._compute_reordered_efforts())

        # Optional sim gripper bridge.
        if self.gripper_server is not None:
            self._apply_gripper_commands(simulation_time)

        self._apply_external_force_ui()

        if self.step_counter % 500 == 0:
            now_wall = time.monotonic()
            wall_dt = now_wall - self._last_log_wall_time
            sim_dt = simulation_time - self._last_log_sim_time
            realtime_factor = sim_dt / wall_dt if wall_dt > 0.0 else 0.0
            print("wall_dt={:.3f}s, sim_dt={:.3f}s, rtf={:.3f}".format(wall_dt, sim_dt, realtime_factor))
            self._last_log_wall_time = now_wall
            self._last_log_sim_time = simulation_time

    def _apply_external_force_ui(self) -> None:
        """Apply force/torque from the ExternalForceUI to the selected body only."""
        force_ui = getattr(self, "force_ui", None)
        if force_ui is None:
            return

        if hasattr(force_ui, "consume_apply_once"):
            apply_once = force_ui.consume_apply_once()
        else:
            apply_once = bool(getattr(force_ui, "apply_trigger", False))
            force_ui.apply_trigger = False

        continuous_enabled = bool(getattr(force_ui, "continuous_enabled", False))
        if not apply_once and not continuous_enabled:
            return

        body_name = getattr(force_ui, "selected_body", "link_torso_5")
        body_prims = {
            "link_torso_5": self.robot.torso_5,
            "ee_left": self.robot.ee_left,
            "ee_right": self.robot.ee_right,
        }
        rigid_prim = body_prims.get(body_name)
        if rigid_prim is None:
            self._disable_external_force_ui(force_ui, f"selected body '{body_name}' is not initialized")
            return

        try:
            force = np.asarray(force_ui.forces[body_name], dtype=np.float32).reshape(1, 3)
            torque = np.asarray(force_ui.torques[body_name], dtype=np.float32).reshape(1, 3)
            if not np.any(force) and not np.any(torque):
                return

            rigid_prim_view = rigid_prim._rigid_prim_view
            if hasattr(rigid_prim_view, "is_physics_handle_valid") and not rigid_prim_view.is_physics_handle_valid():
                self._disable_external_force_ui(force_ui, f"selected body '{body_name}' physics handle is not valid")
                return

            rigid_prim_view.apply_forces_and_torques_at_pos(
                forces=force,
                torques=torque,
                is_global=True,
            )
        except Exception as exc:
            self._disable_external_force_ui(force_ui, f"failed for {body_name}: {exc}")

    def _disable_external_force_ui(self, force_ui, reason: str) -> None:
        """Disable continuous force application and avoid repeating the same error."""
        force_ui.continuous_enabled = False
        force_ui.apply_trigger = False
        if hasattr(force_ui, "apply_once_requested"):
            force_ui.apply_once_requested = False
        if hasattr(force_ui, "_update_continuous_status"):
            force_ui._update_continuous_status()
        if not self._external_force_error_logged:
            print(f"[RBY1Task] External force disabled: {reason}")
            self._external_force_error_logged = True

    def post_physics_step(self, physics_dt: float) -> None:
        """After a physics step, send the state to C++ and prepare the next command."""
        if self.udp_bridge is None or self.robot is None or not hasattr(self, "joint_indices"):
            return

        state_time = 0.0
        if self._last_pre_step_sim_time is not None:
            state_time = self._last_pre_step_sim_time + physics_dt

        self._update_udp_state(state_time, update_velocity=True)

        self.udp_bridge.send_state(
            t=state_time,
            is_ready=self._ready_flags,
            position=self._state_q,
            velocity=self._state_dq,
            current=self._zero_ctrl_state,  # Isaac Sim does not expose motor current
            torque=self._state_tau,
        )
        self._state_sent_once = True

    def _apply_udp_command(self, cmd) -> None:
        """Store a received RobotCommand as the PD reference used by the next pre_step."""
        mode, cmd_target, feedback_gain, feedforward_torque, _finished, _kp, _kd = cmd
        mode = np.array(mode, dtype=bool)
        mode[: self.model_config.mobility_dof] = True
        # C++ mode: True = velocity control, False = position control (matches MuJoCo).
        self.pd_controller.update_target(cmd_target, mode)
        self.pd_controller.feedback_gain = np.clip(feedback_gain.astype(float) / 10.0, 0.0, 1.0)
        self.pd_controller.feedforward_term = feedforward_torque

    def _apply_gripper_commands(self, simulation_time: float) -> None:
        """Drive finger joints from ``SimGripperServer`` closeness targets."""
        cmd = self.gripper_server.get_targets()
        # Only apply commands after homing is complete (safety guard).
        if cmd["homed"]:
            left_m = closeness_to_finger_meter(cmd["left"])
            right_m = closeness_to_finger_meter(cmd["right"])
            targets = np.array([left_m, right_m], dtype=np.float64)
            self.robot._articulation_view.set_joint_position_targets(
                targets, joint_indices=self.gripper_command_indices
            )

        # Report present positions back as closeness (best-effort).
        try:
            joints_state = self.robot.get_joints_state()
            positions = np.asarray(joints_state.positions, dtype=np.float64)
            l_idx, r_idx = self.gripper_command_indices
            self.gripper_server.set_present(
                finger_meter_to_closeness(positions[l_idx]),
                finger_meter_to_closeness(positions[r_idx]),
                simulation_time,
            )
        except Exception:
            pass

    def _build_reference_target(self, simulation_time: float) -> tuple[np.ndarray, np.ndarray]:
        """Generate a built-in test trajectory (C++ model DOF order) for standalone mode."""
        velocity_mode = np.zeros(len(self.joint_indices), dtype=bool)
        target = np.zeros(len(self.joint_indices))

        velocity_mode[: self.model_config.mobility_dof] = True
        target[: self.model_config.mobility_dof] = self.model_config.reference_wheel_target
        target[: self.model_config.mobility_dof] *= np.clip(simulation_time * 0.1, a_min=0.0, a_max=1.0)

        if simulation_time < 4.0:
            gripper_target = np.array([[0.0, -0.0]], dtype=np.float32)
            self.robot._articulation_view.set_joint_position_targets(
                positions=gripper_target,
                joint_indices=self.gripper_command_indices,
            )
        elif simulation_time < 8.0:
            gripper_target = np.array([[-0.05, -0.05]], dtype=np.float32)
            self.robot._articulation_view.set_joint_position_targets(
                positions=gripper_target,
                joint_indices=self.gripper_command_indices,
            )

        return velocity_mode, target

    def _compute_reordered_efforts(self) -> np.ndarray:
        """Expand the C++ model-DOF torques into the full Isaac Sim DOF buffer."""
        if self._state_q is None or self._state_dq is None or self._efforts_full is None:
            raise RuntimeError("[RBY1Task] joint state buffer not initialised.")

        efforts = self.pd_controller.compute_torque(self._state_q, self._state_dq)
        efforts = np.clip(efforts, -self.max_efforts, self.max_efforts)
        self._efforts_full.fill(0.0)
        self._efforts_full[self.joint_indices] = efforts
        return self._efforts_full

    def coswave(self, t):
        return 0.5 * (1 - np.cos(2 * np.pi / 4.0 * t))

    def sinwave(self, t):
        return np.sin(2 * np.pi / 4.0 * t)

    # ------------------------------------------------------------------
    # State collection
    # ------------------------------------------------------------------

    def _update_full_state(self, sample_time: Optional[float] = None, update_velocity: bool = True) -> None:
        """Update the member state buffers needed by the standalone path."""
        joints_state = self.robot.get_joints_state()
        joints_tau = self.robot.get_measured_joint_efforts()
        pos_IB, quat_IB = self.robot.get_world_pose()
        base_lin_vel_b, base_ang_vel_b = self._get_base_velocities_in_body_frame(quat_IB)
        q = np.array(joints_state.positions)[self.joint_indices]
        np.copyto(self._state_q, q)
        self._update_joint_velocity(sample_time, update_velocity)

        raw_tau = joints_tau
        if raw_tau is None or np.ndim(raw_tau) == 0:
            self._state_tau.fill(0.0)
        else:
            tau = np.array(raw_tau)[self.joint_indices]
            np.copyto(self._state_tau, tau)

        _ = pos_IB, base_lin_vel_b, base_ang_vel_b

    def _update_udp_state(self, sample_time: Optional[float] = None, update_velocity: bool = True) -> None:
        """Update only the minimal joint state needed by the UDP lockstep path."""
        q = np.asarray(self.robot.get_joint_positions(joint_indices=self._joint_indices_np))
        np.copyto(self._state_q, q)
        self._update_joint_velocity(sample_time, update_velocity)
        joints_tau = self.robot.get_measured_joint_efforts()
        raw_tau = joints_tau
        if raw_tau is None or np.ndim(raw_tau) == 0:
            self._state_tau.fill(0.0)
        else:
            tau = np.array(raw_tau)[self.joint_indices]
            np.copyto(self._state_tau, tau)

    def _update_joint_velocity(self, sample_time: Optional[float], update_velocity: bool) -> None:
        """Derive a finite-difference velocity from ``_state_q`` and apply a low-pass filter."""
        if self._state_q is None or self._state_dq is None:
            raise RuntimeError("[RBY1Task] joint state buffer not initialised.")

        if sample_time is None or not update_velocity:
            return

        if self._prev_diff_position is None or self._prev_diff_time is None:
            self._prev_diff_position = self._state_q.copy()
            self._prev_diff_time = sample_time
            self._state_dq.fill(0.0)
            if self._filtered_joint_velocity is not None:
                self._filtered_joint_velocity.fill(0.0)
            return

        dt = sample_time - self._prev_diff_time
        if dt <= 1e-9:
            return

        np.subtract(self._state_q, self._prev_diff_position, out=self._state_dq)
        self._state_dq /= dt

        if self._filtered_joint_velocity is None or self._filtered_joint_velocity.shape != self._state_dq.shape:
            self._filtered_joint_velocity = np.zeros_like(self._state_dq)

        alpha = self._joint_velocity_lpf_alpha
        self._filtered_joint_velocity += alpha * (self._state_dq - self._filtered_joint_velocity)
        np.copyto(self._state_dq, self._filtered_joint_velocity)

        np.copyto(self._prev_diff_position, self._state_q)
        self._prev_diff_time = sample_time

    def _get_base_velocities_in_body_frame(self, quat_IB: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        lin_I = self.robot.get_linear_velocity()
        ang_I = self.robot.get_angular_velocity()
        R_BI = quat_to_rot_matrix(quat_IB).transpose()
        return np.matmul(R_BI, lin_I), np.matmul(R_BI, ang_I)

    # ------------------------------------------------------------------
    # BaseTask lifecycle
    # ------------------------------------------------------------------

    def calculate_metrics(self) -> dict:
        raise NotImplementedError

    def is_done(self) -> bool:
        return False

    def cleanup(self) -> None:
        if self._state_q is not None:
            self._state_q.fill(0.0)
        if self._state_dq is not None:
            self._state_dq.fill(0.0)
        if self._filtered_joint_velocity is not None:
            self._filtered_joint_velocity.fill(0.0)
        if self._state_tau is not None:
            self._state_tau.fill(0.0)

    def post_reset(self) -> None:
        """Initialise joint index mapping and PD controller after a reset."""
        self.step_counter = 0
        self._last_pre_step_sim_time = None
        self._prev_diff_position = None
        self._prev_diff_time = None
        self._filtered_joint_velocity = None
        self._new_command_count_since_log = 0
        self._command_wait_miss_count_since_log = 0
        self._external_force_error_logged = False
        self._state_sent_once = False
        self._command_stream_started = False
        self._last_log_wall_time = time.monotonic()
        self._last_log_sim_time = None
        if self.udp_bridge is not None:
            _, _, self._last_command_seq = self.udp_bridge.get_latest_command_with_status(self._last_command_seq)

        self.num_joints = self.robot.num_dof
        self.dof_names = self.robot.dof_names
        print(f"[RBY1Task] num_joints={self.num_joints} dof_names={list(self.dof_names)}")

        # Map C++ joint names → Isaac Sim DOF indices.
        self.joint_indices = []
        for name in self.model_config.cpp_joint_names:
            idx = self.robot._articulation_view.get_dof_index(name)
            if idx < 0:
                raise RuntimeError(
                    f"[RBY1Task] C++ joint '{name}' not found in Isaac Sim DOF. "
                    f"dof_names: {list(self.dof_names)}"
                )
            self.joint_indices.append(idx)
        self.isaac_to_cpp_idx = self.joint_indices  # alias kept for UDP-bridge compatibility
        print(f"[RBY1Task] joint_indices: {self.joint_indices}")
        self._joint_indices_np = np.asarray(self.joint_indices, dtype=np.int64)

        n_ctrl = len(self.joint_indices)
        self._ready_flags = np.ones(n_ctrl, dtype=bool)
        self._zero_ctrl_state = np.zeros(n_ctrl, dtype=float)
        self._state_q = np.zeros(n_ctrl, dtype=float)
        self._state_dq = np.zeros(n_ctrl, dtype=float)
        self._filtered_joint_velocity = np.zeros(n_ctrl, dtype=float)
        self._state_tau = np.zeros(n_ctrl, dtype=float)
        self._efforts_full = np.zeros(self.num_joints, dtype=float)

        self.gripper_command_indices = []
        for name in self.GRIPPER_JOINT_NAMES:
            idx = self.robot._articulation_view.get_dof_index(name)
            if idx < 0:
                raise RuntimeError(
                    f"[RBY1Task] gripper joint '{name}' not found in Isaac Sim DOF. "
                    f"dof_names: {list(self.dof_names)}"
                )
            self.gripper_command_indices.append(idx)
        print(f"[RBY1Task] gripper_command_indices: {self.gripper_command_indices}")

        # Effort mode for the controlled joints; position mode for the gripper.
        self.robot._articulation_view.switch_control_mode("effort", joint_indices=self.joint_indices)
        self.robot._articulation_view.switch_control_mode("position", joint_indices=self.gripper_command_indices)

        self.max_efforts = self.robot._articulation_view.get_max_efforts()[0, self.joint_indices]

        self._initialize_pd_controller()
        self._set_default_robot_state()

    def _initialize_pd_controller(self) -> None:
        """Create the PD controller using per-joint gains (C++ model DOF order)."""
        joint_kp = np.asarray(self.model_config.joint_kp, dtype=np.float32)
        joint_kd = np.asarray(self.model_config.joint_kd, dtype=np.float32)
        if len(joint_kp) != len(self.joint_indices) or len(joint_kd) != len(self.joint_indices):
            raise RuntimeError("[RBY1Task] PD gain array length does not match the C++ model joint count.")

        self.pd_controller = PDController(kp=joint_kp, kd=joint_kd, num_joints=len(self.joint_indices))

        # Initialise the controller to a neutral state to avoid a sudden torque step on reset.
        self.pd_controller.velocity_mode.fill(False)
        self.pd_controller.target_pos.fill(0.0)
        self.pd_controller.target_vel.fill(0.0)
        self.pd_controller.feedback_gain.fill(1.0)
        self.pd_controller.feedforward_term.fill(0.0)
        self.pd_controller.target_torque.fill(0.0)

    def _set_default_robot_state(self) -> None:
        """Reset the robot pose/joint state explicitly right after a reset."""
        self.robot.set_world_pose(position=np.array([0.0, 0.0, 1.31]))
        self.robot.set_joint_positions(np.zeros(self.num_joints, dtype=float))
        self.robot.set_joint_velocities(np.zeros(self.num_joints, dtype=float))
        self.robot.set_joint_efforts(np.zeros(self.num_joints, dtype=float))
