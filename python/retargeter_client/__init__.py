"""Client for the retargeter teleop server (`retarget teleop`): stream human frames, receive robot frames.

    from retargeter_client import TeleopClient, load_bvh
    clip = load_bvh("walk.bvh")
    with TeleopClient("tcp://127.0.0.1:5560") as client:
        info = client.open_stream(0, clip.skeleton, clip.fps)
        client.send_frame(0, seq=0, t_source=time.monotonic(), joint_pos=clip.joint_pos[0], joint_quat=clip.joint_quat[0])
        frame = client.receive(timeout=1.0)   # RobotFrame
"""

from .bvh import Clip, Skeleton, load_bvh
from .client import TeleopClient, TeleopError
from .messages import HumanFrame, RobotFrame, StreamEnd, StreamInfo, StreamStart

FLAGS = {1: "velocity_clamped", 2: "cold_start", 4: "unsupported", 8: "penetration", 16: "environment_support",
         32: "flight", 64: "frames_dropped"}

__all__ = ["Clip", "Skeleton", "load_bvh", "TeleopClient", "TeleopError", "HumanFrame", "RobotFrame", "StreamEnd",
           "StreamInfo", "StreamStart", "FLAGS"]
