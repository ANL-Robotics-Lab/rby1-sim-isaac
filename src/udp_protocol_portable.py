"""Portable RBY1 real-time UDP packet codec.

This is a byte-compatible fallback for the Linux-only ``udp_protocol`` module
bundled in the vendor simulator image. The layout follows
``rby1-sdk/net/real_time_control_protocol.h``. Scalar fields are little-endian,
matching the RBY1 SDK implementation and supported Windows/Linux hosts.
"""

from __future__ import annotations

import struct
from typing import Final, Optional, Tuple

import numpy as np


_HEADER: Final[bytes] = b"$$"
_FOOTER: Final[bytes] = b"%%"
_CRC8_POLYNOMIAL: Final[int] = 0x31

RobotCommand = Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    bool,
    Optional[np.ndarray],
    Optional[np.ndarray],
]


def _build_crc8_table() -> tuple[int, ...]:
    table: list[int] = []
    for value in range(256):
        crc = value
        for _ in range(8):
            if crc & 0x80:
                crc = ((crc << 1) ^ _CRC8_POLYNOMIAL) & 0xFF
            else:
                crc = (crc << 1) & 0xFF
        table.append(crc)
    return tuple(table)


_CRC8_TABLE: Final[tuple[int, ...]] = _build_crc8_table()


def crc8(data: bytes | bytearray | memoryview) -> int:
    """Return the RBY1 protocol CRC-8 for ``data``."""
    crc = 0xFF
    for value in memoryview(data).cast("B"):
        crc = _CRC8_TABLE[crc ^ value]
    return crc


def _one_dimensional(name: str, value, dtype) -> np.ndarray:
    array = np.asarray(value, dtype=dtype)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional, got shape {array.shape}")
    return array


def _finish_packet(body: bytes | bytearray) -> bytes:
    # The uint16 length covers the body, CRC byte, and two-byte footer. It does
    # not include the two-byte header or the uint16 length field itself.
    declared_length = len(body) + 3
    if declared_length > 0xFFFF:
        raise ValueError("RBY1 UDP packet exceeds the uint16 protocol length")

    prefix = _HEADER + struct.pack("<H", declared_length) + bytes(body)
    return prefix + bytes((crc8(prefix),)) + _FOOTER


def build_robot_state_packet(
    t: float,
    is_ready: np.ndarray,
    position: np.ndarray,
    velocity: np.ndarray,
    current: np.ndarray,
    torque: np.ndarray,
) -> bytes:
    """Serialise an RBY1 ``ControlState`` packet."""
    ready = _one_dimensional("is_ready", is_ready, np.bool_)
    arrays = [
        _one_dimensional("position", position, np.dtype("<f8")),
        _one_dimensional("velocity", velocity, np.dtype("<f8")),
        _one_dimensional("current", current, np.dtype("<f8")),
        _one_dimensional("torque", torque, np.dtype("<f8")),
    ]

    dof = ready.size
    if dof > 0xFF:
        raise ValueError(f"RBY1 UDP protocol supports at most 255 DOF, got {dof}")
    for name, array in zip(("position", "velocity", "current", "torque"), arrays):
        if array.size != dof:
            raise ValueError(f"{name} has {array.size} entries; expected {dof}")

    body = bytearray(struct.pack("<Bd", dof, float(t)))
    body.extend(ready.astype(np.uint8, copy=False).tobytes())
    for array in arrays:
        body.extend(array.tobytes())
    return _finish_packet(body)


def _validated_packet(data: bytes) -> tuple[bytes, int] | None:
    try:
        packet = bytes(data)
    except (TypeError, ValueError):
        return None

    if len(packet) < 7 or packet[:2] != _HEADER:
        return None

    declared_length = struct.unpack_from("<H", packet, 2)[0]
    total_length = 4 + declared_length
    if declared_length < 3 or len(packet) < total_length:
        return None

    packet = packet[:total_length]
    if packet[-2:] != _FOOTER:
        return None

    crc_position = total_length - 3
    if packet[crc_position] != crc8(packet[:crc_position]):
        return None
    return packet, crc_position


def parse_robot_command_packet(data: bytes) -> Optional[RobotCommand]:
    """Parse an RBY1 ``ControlInput`` packet, returning ``None`` if malformed.

    Recent simulator packets may append optional float64 ``kp`` and ``kd``
    arrays after the ``finished`` byte. Older SDK packets omit both arrays.
    """
    validated = _validated_packet(data)
    if validated is None:
        return None
    packet, crc_position = validated

    index = 4
    if index >= crc_position:
        return None
    dof = packet[index]
    index += 1

    standard_size = dof + (8 * dof) + (4 * dof) + (8 * dof) + 1
    remaining = crc_position - index
    if remaining not in (standard_size, standard_size + 16 * dof):
        return None

    mode = np.frombuffer(packet, dtype=np.uint8, count=dof, offset=index).astype(np.bool_)
    index += dof
    target = np.frombuffer(packet, dtype="<f8", count=dof, offset=index).copy()
    index += 8 * dof
    feedback_gain = np.frombuffer(packet, dtype="<u4", count=dof, offset=index).copy()
    index += 4 * dof
    feedforward_torque = np.frombuffer(packet, dtype="<f8", count=dof, offset=index).copy()
    index += 8 * dof
    finished = packet[index] == 1
    index += 1

    kp = None
    kd = None
    if index < crc_position:
        kp = np.frombuffer(packet, dtype="<f8", count=dof, offset=index).copy()
        index += 8 * dof
        kd = np.frombuffer(packet, dtype="<f8", count=dof, offset=index).copy()
        index += 8 * dof

    if index != crc_position:
        return None
    return mode, target, feedback_gain, feedforward_torque, finished, kp, kd
