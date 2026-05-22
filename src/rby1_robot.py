# SPDX-FileCopyrightText: Copyright (c) 2020-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""RBY1 articulation wrapper with PhysX joint/motor configuration."""
from __future__ import annotations

from typing import Optional, Sequence

import carb
import numpy as np
import omni
from isaacsim.core.prims import SingleArticulation
from pxr import PhysxSchema, Sdf, UsdPhysics

from motor_profiles import get_profile_for_joint


class RBY1Robot(SingleArticulation):
    """Articulation wrapper that applies PhysX drive/motor properties at setup."""

    def __init__(
        self,
        prim_path: str,
        name: str = "rby1_robot",
        position: Optional[Sequence[float]] = None,
        orientation: Optional[Sequence[float]] = None,
    ) -> None:
        self.base_prim_path = prim_path + "/base"
        SingleArticulation.__init__(
            self,
            prim_path=prim_path,
            name=name,
            position=position,
            orientation=orientation,
            articulation_controller=None,
        )

    def post_reset(self) -> None:
        SingleArticulation.post_reset(self)

    def initialize(self, physics_sim_view: omni.physics.tensors.SimulationView = None) -> None:
        SingleArticulation.initialize(self, physics_sim_view=physics_sim_view)

    # ------------------------------------------------------------------
    # PhysX property helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _set_float_attr(prim, attr_name: str, value: float) -> None:
        attr = prim.GetAttribute(attr_name)
        if not attr:
            attr = prim.CreateAttribute(attr_name, Sdf.ValueTypeNames.Float)
        attr.Set(float(value))

    def _apply_joint_motor_model(self, prim, joint_api, drive_type: str, profile: dict) -> None:
        joint_api.CreateJointFrictionAttr().Set(profile["joint_friction"])
        joint_api.CreateArmatureAttr().Set(profile["armature"])
        prim.ApplyAPI("PhysxJointAxisAPI", drive_type)

        viscous = profile["viscous_nm_s_per_rad"]
        if drive_type == "angular":
            viscous = np.deg2rad(viscous)

        prefix = f"physxJointAxis:{drive_type}"
        self._set_float_attr(prim, f"{prefix}:staticFrictionEffort", profile["static_effort"])
        self._set_float_attr(prim, f"{prefix}:dynamicFrictionEffort", profile["dynamic_effort"])
        self._set_float_attr(prim, f"{prefix}:viscousFrictionCoefficient", viscous)

    # ------------------------------------------------------------------
    # Public configuration entry points
    # ------------------------------------------------------------------

    def set_properties(self, stage) -> None:
        """Apply default PhysX properties to every rigid body in the stage."""
        for prim in stage.Traverse():
            if prim.GetTypeName() != "Xform":
                continue
            if prim.HasAPI(UsdPhysics.RigidBodyAPI) and not prim.HasAPI(PhysxSchema.PhysxRigidBodyAPI):
                PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
            if not prim.HasAPI(PhysxSchema.PhysxRigidBodyAPI):
                continue
            rb_api = PhysxSchema.PhysxRigidBodyAPI.Get(stage, prim.GetPath())
            if not rb_api:
                continue
            rb_api.CreateDisableGravityAttr().Set(False)
            rb_api.CreateRetainAccelerationsAttr().Set(False)
            rb_api.CreateLinearDampingAttr().Set(0.0)
            rb_api.CreateAngularDampingAttr().Set(0.0)
            rb_api.CreateMaxLinearVelocityAttr().Set(200.0)
            rb_api.CreateMaxAngularVelocityAttr().Set(1000.0)

    def set_joint_properties(self, stage, stiffness: float = 5e4, damping: float = 1e2) -> None:
        """Apply PhysX/Drive properties to all revolute and prismatic joints in the stage."""
        for prim in stage.Traverse():
            type_name = prim.GetTypeName()
            if type_name == "PhysicsPrismaticJoint":
                self._configure_prismatic_joint(prim)
            elif type_name == "PhysicsRevoluteJoint":
                self._configure_revolute_joint(prim)

    def _configure_prismatic_joint(self, prim) -> None:
        """Configure two-finger gripper prismatic joints."""
        joint = UsdPhysics.PrismaticJoint(prim)
        joint.CreateBreakForceAttr().Set(float("inf"))
        joint.CreateBreakTorqueAttr().Set(float("inf"))

        joint_api = PhysxSchema.PhysxJointAPI(prim)
        joint_api.CreateMaxJointVelocityAttr().Set(0.04)
        joint_api.CreateJointFrictionAttr().Set(0.0)
        joint_api.CreateArmatureAttr().Set(0.0)

        if prim.HasAPI(UsdPhysics.DriveAPI):
            drive = UsdPhysics.DriveAPI(prim, "linear")
            drive.CreateStiffnessAttr().Set(100.0)
            drive.CreateDampingAttr().Set(1.0)
            drive.CreateTargetPositionAttr(0.0)
            drive.CreateTargetVelocityAttr(0.0)

    def _configure_revolute_joint(self, prim) -> None:
        joint = UsdPhysics.RevoluteJoint(prim)
        if not joint:
            carb.log_warn(f"[RBY1Robot] Failed to retrieve RevoluteJoint at: {prim.GetPath()}")
            return

        joint.CreateBreakForceAttr().Set(float("inf"))
        joint.CreateBreakTorqueAttr().Set(float("inf"))

        joint_api = PhysxSchema.PhysxJointAPI(prim)
        profile = get_profile_for_joint(prim.GetName())

        if prim.HasAPI(UsdPhysics.DriveAPI):
            drive = UsdPhysics.DriveAPI(prim, "angular")
            drive.CreateStiffnessAttr().Set(0.0)
            drive.CreateDampingAttr().Set(0.0)
            self._apply_joint_motor_model(prim, joint_api, "angular", profile)
        else:
            joint_api.CreateMaxJointVelocityAttr().Set(1000.0)
            joint_api.CreateJointFrictionAttr().Set(0.0)
            joint_api.CreateArmatureAttr().Set(0.0)
