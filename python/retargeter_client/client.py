"""ZeroMQ client of the teleop server: one DEALER socket carries StreamStart / HumanFrame / StreamEnd out and
StreamInfo / RobotFrame back (the server's ROUTER routes replies to this socket)."""

from __future__ import annotations

import time

import numpy as np
import zmq

from . import messages
from .bvh import Skeleton


class TeleopError(RuntimeError):
    pass


class TeleopClient:
    def __init__(self, endpoint: str = "tcp://127.0.0.1:5560", context: zmq.Context | None = None):
        self.context = context or zmq.Context.instance()
        self.socket = self.context.socket(zmq.DEALER)
        self.socket.setsockopt(zmq.LINGER, 0)
        self.socket.setsockopt(zmq.SNDHWM, 100000)
        self.socket.setsockopt(zmq.RCVHWM, 100000)
        self.socket.connect(endpoint)
        self.pending = []  # messages received while waiting for something else
        self.infos = {}

    def close(self) -> None:
        self.socket.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def open_stream(self, stream_id: int, skeleton: Skeleton, fps: float, timeout: float = 60.0) -> messages.StreamInfo:
        """Opens (or restarts) a stream; waits for the server's StreamInfo (the first stream also builds the
        server's retargeter, about a second)."""
        self.socket.send(messages.StreamStart(stream_id=stream_id, fps=fps, skeleton=skeleton.name,
                                              joint_names=list(skeleton.joint_names), parents=skeleton.parents,
                                              rest_offsets=skeleton.rest_offsets).encode())
        deadline = time.monotonic() + timeout
        while True:
            left = deadline - time.monotonic()
            if left <= 0:
                raise TeleopError(f"no reply to StreamStart {stream_id} within {timeout} s (is `retarget teleop` running?)")
            if self.socket.poll(int(left * 1000)) == 0:
                continue
            message = messages.decode(self.socket.recv())
            if isinstance(message, messages.StreamInfo) and message.stream_id == stream_id:
                if message.status != 0:
                    raise TeleopError(message.message)
                self.infos[stream_id] = message
                return message
            self.pending.append(message)

    def send_frame(self, stream_id: int, seq: int, t_source: float, joint_pos, joint_quat) -> None:
        """`joint_pos` (J, 3) local translations, `joint_quat` (J, 4) local rotations w x y z (canonical frame)."""
        self.socket.send(messages.HumanFrame(stream_id=stream_id, seq=seq, t_source=t_source,
                                             joint_pos=np.asarray(joint_pos, np.float32),
                                             joint_quat=np.asarray(joint_quat, np.float32)).encode())

    def close_stream(self, stream_id: int) -> None:
        self.socket.send(messages.StreamEnd(stream_id=stream_id).encode())

    def receive(self, timeout: float | None = None):
        """Next message from the server (RobotFrame, or StreamInfo for errors); None on timeout."""
        if self.pending:
            return self.pending.pop(0)
        if self.socket.poll(None if timeout is None else int(timeout * 1000)) == 0:
            return None
        return messages.decode(self.socket.recv())
