# SPDX-FileCopyrightText: Copyright (c) 2020-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""PD torque controller used by ``RBY1Task``."""
from __future__ import annotations

import numpy as np

from config import TORQUE_DAMPING_SCALE


class PDController:
    """Per-joint PD controller with selectable position/velocity mode.

    For each joint the computed torque is::

        # position mode
        torque = feedback_gain * (kp * pos_err + kd * vel_err * TORQUE_DAMPING_SCALE)
                 + feedforward_term

        # velocity mode
        torque = feedback_gain * (kp * vel_err * TORQUE_DAMPING_SCALE)
                 + feedforward_term

    The ``TORQUE_DAMPING_SCALE`` factor matches the MuJoCo control callback.
    """

    def __init__(self, kp: np.ndarray, kd: np.ndarray, num_joints: int):
        self.kp = kp
        self.kd = kd
        self.num_joints = num_joints
        self.velocity_mode    = np.zeros(num_joints, dtype=bool)
        self.target_pos       = np.zeros(num_joints, dtype=float)
        self.target_vel       = np.zeros(num_joints, dtype=float)
        self.feedback_gain    = np.ones(num_joints, dtype=float)
        self.feedforward_term = np.zeros(num_joints, dtype=float)
        self.target_torque    = np.zeros(num_joints, dtype=float)

    def compute_torque(self, observations: dict) -> np.ndarray:
        """Compute the per-joint torque for the current observation."""
        obs = observations["RBY1"]
        pos_error = self.target_pos - obs["q"]
        vel_error = self.target_vel - obs["dq"]

        torque_pos = self.kp * pos_error + self.kd * vel_error * TORQUE_DAMPING_SCALE
        torque_vel = self.kp * vel_error * TORQUE_DAMPING_SCALE

        self.target_torque = (
            self.feedback_gain
            * np.where(self.velocity_mode, torque_vel, torque_pos)
            + self.feedforward_term
        )
        return self.target_torque

    def update_target(self, target: np.ndarray, velocity_mode: np.ndarray) -> None:
        self.velocity_mode = velocity_mode
        self.target_pos = np.where(~self.velocity_mode, target, 0)
        self.target_vel = np.where(self.velocity_mode, target, 0)

    def update_gains(self, kp: np.ndarray, kd: np.ndarray) -> None:
        self.kp = kp
        self.kd = kd

    def update_feedforward_term(self, feedforward_term: np.ndarray) -> None:
        self.feedforward_term = feedforward_term
