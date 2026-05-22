# SPDX-FileCopyrightText: Copyright (c) 2020-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""UDP transport for the rby1-sdk real-time control protocol.

Wraps :mod:`udp_protocol` with socket I/O and a background receive thread.
"""
from __future__ import annotations

import logging
import socket
import threading
from typing import Optional

import numpy as np

from udp_protocol import (
    RobotCommand,
    build_robot_state_packet,
    parse_robot_command_packet,
)

log = logging.getLogger(__name__)


class RBY1UdpBridge:
    """UDP bridge between Isaac Sim (Python) and rby1 Core (C++).

    * Sends ``RobotState`` packets to ``state_send_addr``
    * Receives ``RobotCommand`` packets on ``cmd_recv_port`` (background thread)
    """

    def __init__(
        self,
        state_send_ip: str = "127.0.0.1",
        state_send_port: int = 5005,
        cmd_recv_port: int = 5006,
    ):
        self.state_send_addr = (state_send_ip, state_send_port)
        self.cmd_recv_port = cmd_recv_port

        self._send_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._recv_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._recv_sock.bind(("0.0.0.0", cmd_recv_port))

        self._lock = threading.Lock()
        self._command: Optional[RobotCommand] = None

        self._running = True
        self._recv_thread = threading.Thread(target=self._recv_loop, daemon=True)
        self._recv_thread.start()

        print(f"[RBY1UdpBridge] State send → {state_send_ip}:{state_send_port}")
        print(f"[RBY1UdpBridge] Command recv ← 0.0.0.0:{cmd_recv_port}")

    def send_state(
        self,
        t: float,
        is_ready: np.ndarray,
        position: np.ndarray,
        velocity: np.ndarray,
        current: np.ndarray,
        torque: np.ndarray,
    ) -> None:
        """Serialise and send a ``RobotState`` packet to C++."""
        pkt = build_robot_state_packet(t, is_ready, position, velocity, current, torque)
        try:
            self._send_sock.sendto(pkt, self.state_send_addr)
        except OSError as exc:
            log.warning("send_state failed: %s", exc)

    def get_latest_command(self) -> Optional[RobotCommand]:
        """Return the most recently received ``RobotCommand``, or ``None``."""
        with self._lock:
            return self._command

    def close(self) -> None:
        self._running = False
        self._send_sock.close()
        self._recv_sock.close()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _recv_loop(self) -> None:
        """Blocking recv loop. Exits cleanly when ``close()`` closes the socket."""
        buf = bytearray(4096)
        while self._running:
            try:
                n, _ = self._recv_sock.recvfrom_into(buf)
            except OSError:
                break  # socket closed
            result = parse_robot_command_packet(bytes(buf[:n]))
            if result is None:
                log.debug("Dropped malformed command packet (%d bytes)", n)
                continue
            with self._lock:
                self._command = result
