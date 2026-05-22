# SPDX-FileCopyrightText: Copyright (c) 2020-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Isaac Sim ``BaseTask`` implementing the RBY1 simulation loop."""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np
from isaacsim.core.api.scenes.scene import Scene
from isaacsim.core.api.tasks import BaseTask
from isaacsim.core.utils.rotations import quat_to_rot_matrix
from isaacsim.core.utils.stage import add_reference_to_stage

from config import (
    CPP_JOINT_NAMES,
    JOINT_KD_BASE,
    JOINT_KP_BASE,
    PD_GAIN_SCALE,
    resolve_usd_path,
)
from rby1_controller import PDController
from rby1_robot import RBY1Robot
from rby1_udp_bridge import RBY1UdpBridge
from sim_gripper_bridge import (
    SimGripperServer,
    closeness_to_finger_meter,
    finger_meter_to_closeness,
)

log = logging.getLogger(__name__)


class RBY1Task(BaseTask):
    """Drives the RBY1 articulation from external UDP commands.

    Each physics step:
      1. Collect joint states (reordered to C++ 24-DOF layout).
      2. Send ``RobotState`` to rby1-sdk and apply the latest ``RobotCommand``.
      3. Compute PD torques and write them to the articulation.
      4. (Optional) Forward gripper commands and report present positions.
    """

    GRIPPER_JOINT_NAMES = ["gripper_finger_l1", "gripper_finger_r1"]

    def __init__(
        self,
        udp_bridge: Optional[RBY1UdpBridge] = None,
        gripper_server: Optional[SimGripperServer] = None,
    ):
        super().__init__(name="rby1_task", offset=None)
        self.robot = None
        self.robot_prim_path = "/World/RBY1"
        self.usd_path = resolve_usd_path()
        self.step_counter = 0
        self.observations: dict = {}
        self.udp_bridge = udp_bridge
        self.gripper_server = gripper_server

    # ------------------------------------------------------------------
    # Scene setup
    # ------------------------------------------------------------------

    def set_up_scene(self, scene: Scene) -> None:
        """Spawn the ground plane and the RBY1 robot in the world."""
        super().set_up_scene(scene)

        scene.add_default_ground_plane(
            name="default_ground_plane",
            prim_path="/World/defaultGroundPlane",
            static_friction=3.0,
            dynamic_friction=2.0,
            restitution=0.1,
        )

        add_reference_to_stage(usd_path=self.usd_path, prim_path=self.robot_prim_path)

        self.rby_robot = RBY1Robot(
            prim_path=self.robot_prim_path,
            name="RBY1",
            position=np.array([0.0, 0.0, 0.0]),
            orientation=np.array([1.0, 0.0, 0.0, 0.0]),
        )
        self.robot = scene.add(self.rby_robot)
        self.rby_robot.set_joint_properties(scene.stage)
        self.rby_robot.set_properties(scene.stage)
        # Self-collision left disabled — enabling it with the current USD collision
        # geometry causes joint instability. A USD collision fix is planned.
        self.robot.set_enabled_self_collisions(False)
        self.robot.set_solver_position_iteration_count(8)
        self.robot.set_solver_velocity_iteration_count(2)

    # ------------------------------------------------------------------
    # Physics step
    # ------------------------------------------------------------------

    def pre_step(self, time_step_index: int, simulation_time: float) -> None:
        """Callback invoked before each PhysX step."""
        self.step_counter += 1

        # 1) Observe.
        self.observations = self.get_states()

        # 2) Robot UDP I/O.
        if self.udp_bridge is not None:
            self._exchange_robot_udp(simulation_time)

        # 3) Compute & apply PD torque.
        self.robot.set_joint_efforts(self._compute_reordered_efforts())
        self._record_reference()

        # 4) Gripper bridge (optional).
        if self.gripper_server is not None:
            self._apply_gripper_commands(simulation_time)

    def _exchange_robot_udp(self, simulation_time: float) -> None:
        """Send the latest state to C++ and consume any new command."""
        obs = self.observations["RBY1"]
        self.udp_bridge.send_state(
            t=simulation_time,
            is_ready=self._is_ready_buf,
            position=obs["q"],
            velocity=obs["dq"],
            current=self._zero_current,   # Isaac Sim does not expose motor current
            torque=obs["tau"],
        )

        cmd = self.udp_bridge.get_latest_command()
        if cmd is None:
            return

        mode, cmd_target, feedback_gain, feedforward_torque, _finished, _kp, _kd = cmd
        # C++ mode: True = velocity control, False = position control (matches MuJoCo).
        self.pd_controller.update_target(cmd_target, mode.astype(bool))

        # Only update feedback_gain when the SDK sends a non-zero value. During
        # power-on ramp-up the SDK sends 0; keeping the previous gain (initialised
        # to 1.0) prevents joints from sagging under gravity and triggering a
        # tracking-error MajorFault on the first RobotCommand.
        gain = np.clip(feedback_gain.astype(float) / 10.0, 0.0, 1.0)
        if np.any(gain > 0):
            self.pd_controller.feedback_gain = gain
        self.pd_controller.feedforward_term = feedforward_torque

    def _record_reference(self) -> None:
        """Store controller setpoints alongside the observation dict."""
        self.observations["RBY1"]["ref_q"] = self.pd_controller.target_pos
        self.observations["RBY1"]["ref_dq"] = self.pd_controller.target_vel
        self.observations["RBY1"]["ref_tau"] = self.pd_controller.target_torque

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
        except Exception as exc:
            log.debug("Skipping gripper present-state send: %s", exc)

    def _compute_reordered_efforts(self) -> np.ndarray:
        """PD torques (C++ 24-DOF order) → full Isaac Sim DOF buffer."""
        efforts = self.pd_controller.compute_torque(self.observations)  # (24,)
        np.clip(efforts, -self.max_efforts, self.max_efforts, out=efforts)
        self._efforts_full.fill(0.0)
        self._efforts_full[self.joint_indices] = efforts
        return self._efforts_full

    # ------------------------------------------------------------------
    # State collection
    # ------------------------------------------------------------------

    def get_states(self) -> dict:
        """Read joint and base state, reordered to the C++ 24-DOF layout."""
        joints_state = self.robot.get_joints_state()
        joints_tau = self.robot.get_measured_joint_efforts()
        pos_IB, quat_IB = self.robot.get_world_pose()
        base_lin_vel_b, base_ang_vel_b = self._get_base_velocities_in_body_frame(quat_IB)

        # Defensive: ``get_measured_joint_efforts`` may return ``None`` or a 0-d array.
        if joints_tau is None or np.ndim(joints_tau) == 0:
            tau = np.zeros(len(self.joint_indices))
        else:
            tau = np.array(joints_tau)[self.joint_indices]

        return {
            "RBY1": {
                "q":   np.array(joints_state.positions)[self.joint_indices],
                "dq":  np.array(joints_state.velocities)[self.joint_indices],
                "tau": tau,
                "base_world_position": pos_IB,
                "base_world_orientation": quat_IB,
                "base_linear_velocity": base_lin_vel_b,
                "base_angular_velocity": base_ang_vel_b,
            }
        }

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
        self.observations = {}

    def post_reset(self) -> None:
        """Initialise joint index mapping and PD controller after a reset."""
        self.step_counter = 0
        self.num_joints = self.robot.num_dof
        self.dof_names = self.robot.dof_names
        print(f"[RBY1Task] num_joints={self.num_joints} dof_names={list(self.dof_names)}")

        self.joint_indices = self._resolve_dof_indices(CPP_JOINT_NAMES, "control")
        # Alias kept for compatibility with the UDP bridge.
        self.isaac_to_cpp_idx = self.joint_indices
        self.gripper_command_indices = self._resolve_dof_indices(
            self.GRIPPER_JOINT_NAMES, "gripper"
        )

        # Effort mode for the 24 controlled joints; position mode for the gripper.
        self.robot._articulation_view.switch_control_mode("effort", joint_indices=self.joint_indices)
        self.robot._articulation_view.switch_control_mode("position", joint_indices=self.gripper_command_indices)

        self.max_efforts = self.robot._articulation_view.get_max_efforts()[0, self.joint_indices]

        # Pre-allocate hot-path buffers to avoid per-step allocation.
        n_ctrl = len(self.joint_indices)
        self._is_ready_buf = np.ones(n_ctrl, dtype=bool)
        self._zero_current = np.zeros(n_ctrl, dtype=np.float64)
        self._efforts_full = np.zeros(self.num_joints, dtype=np.float64)

        self._initialize_pd_controller()
        self._set_default_robot_state()

    def _resolve_dof_indices(self, joint_names: list[str], label: str) -> list[int]:
        """Map joint names to Isaac Sim DOF indices, raising if any are missing."""
        indices = []
        for name in joint_names:
            idx = self.robot._articulation_view.get_dof_index(name)
            if idx < 0:
                raise RuntimeError(
                    f"[RBY1Task] {label} joint '{name}' not found in Isaac Sim DOF. "
                    f"dof_names: {list(self.dof_names)}"
                )
            indices.append(idx)
        print(f"[RBY1Task] {label}_indices: {indices}")
        return indices

    def _initialize_pd_controller(self) -> None:
        """Create the PD controller using per-joint gains in C++ 24-DOF order."""
        if len(JOINT_KP_BASE) != len(self.joint_indices) or len(JOINT_KD_BASE) != len(self.joint_indices):
            raise RuntimeError("[RBY1Task] PD gain array length does not match the C++ 24-DOF joint count.")

        self.pd_controller = PDController(
            kp=JOINT_KP_BASE * PD_GAIN_SCALE,
            kd=JOINT_KD_BASE * PD_GAIN_SCALE,
            num_joints=len(self.joint_indices),
        )

    def _set_default_robot_state(self) -> None:
        """Reset pose/joint state to a known initial configuration."""
        self.robot.set_world_pose(position=np.array([0.0, 0.0, 1.31]))
        zeros = np.zeros(self.num_joints, dtype=float)
        self.robot.set_joint_positions(zeros)
        self.robot.set_joint_velocities(zeros)
        self.robot.set_joint_efforts(zeros)
