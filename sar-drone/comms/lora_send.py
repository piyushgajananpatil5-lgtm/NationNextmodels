"""Compact, framed serial transport for LoRa telemetry."""
from __future__ import annotations

import json
import base64
import struct
import uuid
import zlib
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import serial

MAGIC = b"SAR1"
_HEADER = struct.Struct("!4sH")
_CRC = struct.Struct("!I")
_FRAGMENT_EVENT = "fragment"


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
        """Encode and transmit a payload, fragmenting oversized messages."""
        self.open()
        assert self._serial is not None
        body = json.dumps(payload, separators=(",", ":"), allow_nan=False).encode("utf-8")
        if len(body) <= self.max_packet_bytes:
            self._write(self.encode(payload))
            return

        for packet in self._fragment_packets(body):
            self._write(packet)

    def _write(self, packet: bytes) -> None:
        assert self._serial is not None
        written = self._serial.write(packet)
        if written != len(packet):
            raise IOError(f"serial write incomplete: {written}/{len(packet)} bytes")

    def _fragment_packets(self, body: bytes) -> list[bytes]:
        """Build numbered, base64-encoded fragments that fit the body limit."""
        message_id = uuid.uuid4().hex
        chunk_size = max(1, int(self.max_packet_bytes * 0.55))
        while True:
            chunks = [body[offset:offset + chunk_size] for offset in range(0, len(body), chunk_size)]
            total = len(chunks)
            try:
                packets = [
                    self.encode({
                        "event": _FRAGMENT_EVENT,
                        "message_id": message_id,
                        "sequence": sequence,
                        "total": total,
                        "data": base64.b64encode(chunk).decode("ascii"),
                    })
                    for sequence, chunk in enumerate(chunks)
                ]
            except ValueError:
                packets = []
            if packets and all(len(packet) <= self.max_packet_bytes + _HEADER.size + _CRC.size for packet in packets):
                return packets
            if chunk_size == 1:
                raise ValueError("max_packet_bytes is too small for a fragment envelope")
            chunk_size -= 1


def decode_packet(packet: bytes) -> Mapping[str, Any]:
    """Validate a framed packet and return its JSON payload."""
    if len(packet) < _HEADER.size + _CRC.size:
        raise ValueError("packet is too short")
    magic, body_length = _HEADER.unpack(packet[:_HEADER.size])
    if magic != MAGIC:
        raise ValueError("invalid packet magic")
    body_start = _HEADER.size
    body_end = body_start + body_length
    if len(packet) != body_end + _CRC.size:
        raise ValueError("packet length does not match its header")
    body = packet[body_start:body_end]
    expected_crc = _CRC.unpack(packet[body_end:])[0]
    if zlib.crc32(body) & 0xFFFFFFFF != expected_crc:
        raise ValueError("packet CRC mismatch")
    decoded = json.loads(body.decode("utf-8"))
    if not isinstance(decoded, dict):
        raise ValueError("packet payload must be a JSON object")
    return decoded


class LoRaReassembler:
    """Reassemble fragmented packets; return complete payloads from ``add``."""

    def __init__(self) -> None:
        self._messages: dict[str, dict[str, Any]] = {}

    def add(self, packet: bytes) -> Mapping[str, Any] | None:
        payload = decode_packet(packet)
        if payload.get("event") != _FRAGMENT_EVENT:
            return payload
        message_id = payload.get("message_id")
        sequence = payload.get("sequence")
        total = payload.get("total")
        encoded_data = payload.get("data")
        if not isinstance(message_id, str) or not isinstance(sequence, int) or not isinstance(total, int) or not isinstance(encoded_data, str):
            raise ValueError("invalid fragment envelope")
        if total < 1 or sequence < 0 or sequence >= total:
            raise ValueError("invalid fragment sequence")

        message = self._messages.setdefault(message_id, {"total": total, "parts": {}})
        if message["total"] != total:
            raise ValueError("fragment total changed during reassembly")
        message["parts"][sequence] = base64.b64decode(encoded_data, validate=True)
        if len(message["parts"]) != total:
            return None

        body = b"".join(message["parts"][index] for index in range(total))
        del self._messages[message_id]
        result = json.loads(body.decode("utf-8"))
        if not isinstance(result, dict):
            raise ValueError("reassembled payload must be a JSON object")
        return result

    def send_event(self, event: str, gps: Sequence[float], detections: list[Mapping[str, Any]], waypoints: list[Sequence[float]]) -> None:
        """Transmit the standard SAR event schema."""
        if len(gps) != 2:
            raise ValueError("gps must be [latitude, longitude]")
        self.send({"event": event, "gps": list(gps), "detections": detections, "waypoints": waypoints})
