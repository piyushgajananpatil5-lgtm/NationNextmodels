"""Compact, framed serial transport for LoRa telemetry."""
from __future__ import annotations

import json
import struct
import zlib
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import serial

MAGIC = b"SAR1"
_HEADER = struct.Struct("!4sH")
_CRC = struct.Struct("!I")


@dataclass
class LoRaSender:
    """Send length-limited JSON packets through a pyserial-compatible port."""

    port: str
    baudrate: int = 115200
    timeout_seconds: float = 1.0
    max_packet_bytes: int = 512

    def __post_init__(self) -> None:
        self._serial: serial.Serial | None = None

    def open(self) -> None:
        """Open the configured serial device."""
        if self._serial is None or not self._serial.is_open:
            self._serial = serial.Serial(self.port, self.baudrate, timeout=self.timeout_seconds)

    def close(self) -> None:
        """Close the serial device if it is open."""
        if self._serial is not None and self._serial.is_open:
            self._serial.close()

    def __enter__(self) -> "LoRaSender":
        self.open()
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def encode(self, payload: Mapping[str, Any]) -> bytes:
        """Encode a compact UTF-8 JSON payload with a CRC32 trailer."""
        body = json.dumps(payload, separators=(",", ":"), allow_nan=False).encode("utf-8")
        if len(body) > self.max_packet_bytes:
            raise ValueError(f"payload is {len(body)} bytes; limit is {self.max_packet_bytes}")
        return _HEADER.pack(MAGIC, len(body)) + body + _CRC.pack(zlib.crc32(body) & 0xFFFFFFFF)

    def send(self, payload: Mapping[str, Any]) -> None:
        """Encode and transmit one packet, opening the port on first use."""
        self.open()
        assert self._serial is not None
        packet = self.encode(payload)
        written = self._serial.write(packet)
        if written != len(packet):
            raise IOError(f"serial write incomplete: {written}/{len(packet)} bytes")

    def send_event(self, event: str, gps: Sequence[float], detections: list[Mapping[str, Any]], waypoints: list[Sequence[float]]) -> None:
        """Transmit the standard SAR event schema."""
        if len(gps) != 2:
            raise ValueError("gps must be [latitude, longitude]")
        self.send({"event": event, "gps": list(gps), "detections": detections, "waypoints": waypoints})
