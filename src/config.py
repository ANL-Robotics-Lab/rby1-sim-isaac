# SPDX-FileCopyrightText: Copyright (c) 2020-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Central configuration constants for rby1-sim-isaac.

All hardcoded constants (physics rates, UDP ports, PD gains, gripper calibration)
live here so they can be tuned in one place.
"""
from __future__ import annotations

import os
from typing import Final

import numpy as np


# ============================================================
# Simulation configuration
# ============================================================

PHYSICS_HZ: Final[float] = 500.0
RENDER_HZ: Final[float] = 30.0

PHYSICS_DT: Final[float] = 1.0 / PHYSICS_HZ
RENDER_DT: Final[float] = 1.0 / RENDER_HZ
STAGE_UNITS_IN_METERS: Final[float] = 1.0

# Note: the RBY1 USD asset is selected automatically per robot model.
# See ``RBY1_MODEL_CONFIGS`` and ``_resolve_usd_path`` in ``rby1_task.py``.


# ============================================================
# UDP bridge defaults (rby1-sdk integration)
# ============================================================

# Robot state/command bridge (matches rby1-sdk real_time_control_protocol.h)
ROBOT_STATE_HOST: Final[str] = "127.0.0.1"   # Isaac Sim → C++ rby1-sdk
ROBOT_STATE_PORT: Final[int] = 5005
ROBOT_CMD_PORT: Final[int] = 5006            # C++ rby1-sdk → Isaac Sim

# Gripper bridge (SimDynamixelBus ↔ SimGripperServer)
GRIPPER_HOST: Final[str] = os.environ.get("RBY1_SIM_GRIPPER_HOST", "127.0.0.1")
GRIPPER_CMD_PORT: Final[int] = int(os.environ.get("RBY1_SIM_GRIPPER_CMD_PORT", "5007"))
GRIPPER_STATE_PORT: Final[int] = int(os.environ.get("RBY1_SIM_GRIPPER_STATE_PORT", "5008"))


# ============================================================
# Control configuration (PD controller)
# ============================================================

# C++ kRobotJointNames order (rby1-sdk/include/rby1-sdk/model.h, class A).
# Isaac Sim controls these 24 DOF; gripper joints are handled separately.
CPP_JOINT_NAMES: Final[list[str]] = [
    "right_wheel", "left_wheel",
    "torso_0", "torso_1", "torso_2", "torso_3", "torso_4", "torso_5",
    "right_arm_0", "right_arm_1", "right_arm_2", "right_arm_3",
    "right_arm_4", "right_arm_5", "right_arm_6",
    "left_arm_0", "left_arm_1", "left_arm_2", "left_arm_3",
    "left_arm_4", "left_arm_5", "left_arm_6",
    "head_0", "head_1",
]
N_CTRL: Final[int] = len(CPP_JOINT_NAMES)  # 24

# Per-joint PD gains in CPP_JOINT_NAMES order. Multiplied by 10 when applied
# (kept here as base values so they read identically to the original source).
JOINT_KP_BASE: Final[np.ndarray] = np.array([
    100.0, 100.0,
    391.1, 391.1, 391.1, 57.36, 13.03, 57.36,
    20.86, 20.86, 20.86, 9.127, 3.911, 3.911, 3.911,
    20.86, 20.86, 20.86, 9.127, 3.911, 3.911, 3.911,
    50.0, 50.0,
], dtype=np.float32)

JOINT_KD_BASE: Final[np.ndarray] = np.array([
    1.0, 1.0,
    352.0, 352.0, 352.0, 104.3, 104.3, 104.3,
    52.15, 52.15, 52.15, 20.86, 9.126, 9.126, 9.126,
    52.15, 52.15, 52.15, 20.86, 9.126, 9.126, 9.126,
    2.0, 2.0,
], dtype=np.float32)

PD_GAIN_SCALE: Final[float] = 10.0

# Damping term scaling used inside PDController.compute_torque (matches MuJoCo callback).
TORQUE_DAMPING_SCALE: Final[float] = 0.002


# ============================================================
# Gripper calibration & finger joint kinematics
# ============================================================

# Fake homing range for SimDynamixelBus (emulates motor radian span at ~±45°).
GRIPPER_HOME_MIN_RAD: Final[float] = -0.785398
GRIPPER_HOME_MAX_RAD: Final[float] = +0.785398
GRIPPER_HOMING_STEP_RAD: Final[float] = 0.1

# URDF finger prismatic joint travel: lower=-0.05 m (open), upper=0.0 m (closed).
GRIPPER_FINGER_TRAVEL_M: Final[float] = 0.05
