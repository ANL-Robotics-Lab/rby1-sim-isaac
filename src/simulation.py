# SPDX-FileCopyrightText: Copyright (c) 2020-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Top-level Isaac Sim entry point for the RBY1 simulator."""
from __future__ import annotations

from isaacsim import SimulationApp

from config import (
    GRIPPER_CMD_PORT,
    GRIPPER_HOST,
    GRIPPER_STATE_PORT,
    PHYSICS_DT,
    RENDER_DT,
    ROBOT_CMD_PORT,
    ROBOT_STATE_HOST,
    ROBOT_STATE_PORT,
    STAGE_UNITS_IN_METERS,
)


class Simulation:
    """Owns the Isaac Sim ``World`` and the optional UDP bridges."""

    def __init__(
        self,
        sim_app,
        udp_state_send_ip: str = ROBOT_STATE_HOST,
        udp_state_send_port: int = ROBOT_STATE_PORT,
        udp_cmd_recv_port: int = ROBOT_CMD_PORT,
        use_sim_gripper: bool = False,
        sim_gripper_cmd_port: int = GRIPPER_CMD_PORT,
        sim_gripper_state_host: str = GRIPPER_HOST,
        sim_gripper_state_port: int = GRIPPER_STATE_PORT,
    ):
        # Imports kept lazy: many of these depend on SimulationApp being initialised.
        from isaacsim.core.api import World

        from rby1_task import RBY1Task
        from rby1_udp_bridge import RBY1UdpBridge

        self.simulator = sim_app

        # Robot UDP bridge (always on — the SDK driver lives outside the sim process).
        self.udp_bridge = RBY1UdpBridge(
            state_send_ip=udp_state_send_ip,
            state_send_port=udp_state_send_port,
            cmd_recv_port=udp_cmd_recv_port,
        )

        # Gripper sim bridge (optional).
        self.gripper_server = None
        if use_sim_gripper:
            from sim_gripper_bridge import SimGripperServer
            self.gripper_server = SimGripperServer(
                cmd_bind_port=sim_gripper_cmd_port,
                state_send_host=sim_gripper_state_host,
                state_send_port=sim_gripper_state_port,
            )
            self.gripper_server.start()

        self.world = World(
            physics_dt=PHYSICS_DT,
            rendering_dt=RENDER_DT,
            stage_units_in_meters=STAGE_UNITS_IN_METERS,
            sim_params={"solver_type": "PGS"},
        )
        self.task = RBY1Task(udp_bridge=self.udp_bridge, gripper_server=self.gripper_server)
        self.world.add_task(self.task)
        self.robot = None
        self.reset_needed = False

        self._carb_input = None
        self._input = None
        self._keyboard = None
        self._keyboard_sub = None
        self._register_keyboard_callbacks()

    # ------------------------------------------------------------------
    # Keyboard handling (Backspace = request reset)
    # ------------------------------------------------------------------

    def _register_keyboard_callbacks(self) -> None:
        import carb.input as carb_input
        import omni.appwindow

        self._carb_input = carb_input
        self._input = carb_input.acquire_input_interface()
        self._keyboard = omni.appwindow.get_default_app_window().get_keyboard()
        self._keyboard_sub = self._input.subscribe_to_keyboard_events(
            self._keyboard, self._on_keyboard_event
        )

    def _cleanup_keyboard_callbacks(self) -> None:
        if self._keyboard_sub is not None:
            self._input.unsubscribe_to_keyboard_events(self._keyboard, self._keyboard_sub)
            self._keyboard_sub = None
        self._keyboard = None
        self._input = None

    def _on_keyboard_event(self, event) -> bool:
        if event.type != self._carb_input.KeyboardEventType.KEY_PRESS:
            return False
        if event.input == self._carb_input.KeyboardInput.BACKSPACE:
            self.reset_needed = True
            return True
        return False

    # ------------------------------------------------------------------
    # World lifecycle
    # ------------------------------------------------------------------

    def initial_reset(self) -> None:
        self.world.reset()
        self.world.initialize_physics()
        self.robot = self.world.scene.get_object("RBY1")
        self.world.pause()

    def reset_simulation(self) -> None:
        self.world.reset()
        self.world.initialize_physics()
        self.robot = self.world.scene.get_object("RBY1")
        self.reset_needed = False
        self.world.pause()
        print("[Simulation] Reset")

    def simulation_loop(self) -> None:
        self.world.play()
        try:
            while self.simulator.is_running():
                if self.reset_needed:
                    self.reset_simulation()
                if self.world.is_playing():
                    self.world.step()
                else:
                    self.world.render()
        finally:
            self._cleanup_keyboard_callbacks()
            if self.udp_bridge is not None:
                self.udp_bridge.close()
            if self.gripper_server is not None:
                self.gripper_server.close()
            self.simulator.close()


def _build_argparser():
    import argparse

    p = argparse.ArgumentParser(description="RBY1 Isaac Sim Simulation")
    p.add_argument("--state-ip", default=ROBOT_STATE_HOST,
                   help="Target IP for RobotState transmission")
    p.add_argument("--state-port", type=int, default=ROBOT_STATE_PORT,
                   help="Target port for RobotState transmission")
    p.add_argument("--cmd-port", type=int, default=ROBOT_CMD_PORT,
                   help="Port for receiving RobotCommand")
    p.add_argument("--sim-gripper", action="store_true",
                   help="Enable receiving gripper commands from SimDynamixelBus")
    p.add_argument("--no-sim-gripper", action="store_true",
                   help="Ignore --sim-gripper (used to override the docker default)")
    p.add_argument("--gripper-cmd-port", type=int, default=GRIPPER_CMD_PORT,
                   help="Port for receiving gripper commands")
    p.add_argument("--gripper-state-ip", default=GRIPPER_HOST,
                   help="IP for sending gripper state")
    p.add_argument("--gripper-state-port", type=int, default=GRIPPER_STATE_PORT,
                   help="Port for sending gripper state")
    return p


def main() -> None:
    args, _unknown = _build_argparser().parse_known_args()

    # --no-sim-gripper overrides --sim-gripper (used by docker default args).
    sim_gripper_enabled = args.sim_gripper and not args.no_sim_gripper

    simulation_app = SimulationApp({"headless": False})


    simulation = Simulation(
        simulation_app,
        udp_state_send_ip=args.state_ip,
        udp_state_send_port=args.state_port,
        udp_cmd_recv_port=args.cmd_port,
        use_sim_gripper=sim_gripper_enabled,
        sim_gripper_cmd_port=args.gripper_cmd_port,
        sim_gripper_state_host=args.gripper_state_ip,
        sim_gripper_state_port=args.gripper_state_port,
    )
    simulation.initial_reset()

    mode = f"UDP(→ {args.state_ip}:{args.state_port}, ← :{args.cmd_port})"
    if sim_gripper_enabled:
        mode += (f" + SimGripper(cmd←:{args.gripper_cmd_port}, "
                 f"state→{args.gripper_state_ip}:{args.gripper_state_port})")
    elif args.no_sim_gripper:
        mode += " + SimGripper(off)"
    print(f"[Simulation] Starting... [{mode}]")

    simulation.simulation_loop()


if __name__ == "__main__":
    main()
