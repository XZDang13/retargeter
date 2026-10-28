"""End-to-end check of `retarget teleop` and the Python client: starts the server on a spare port (causal, then
with a 40 ms fixed lag), streams two clips in real time, and checks replies, ordering, latency and error handling.
Run from the repository root:

    build/tools/venv/bin/python python/tests/test_teleop.py
"""

import os
import subprocess
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "python"))

from retargeter_client import TeleopClient, TeleopError, load_bvh, messages  # noqa: E402

BVH = os.path.expanduser("~/Downloads/soma_uniform/bvh/")
ENDPOINT = "tcp://127.0.0.1:5597"


def main() -> int:
    failures = []

    def check(condition, what):
        print(("ok   " if condition else "FAIL ") + what)
        if not condition:
            failures.append(what)

    for lag, budget in [(0.0, 50), (0.04, 120)]:
        print(f"-- lag {lag} s")
        run(lag, budget, check, failures)
    if failures:
        print(f"{len(failures)} check(s) failed")
        return 1
    print("all checks passed")
    return 0


def run(lag, budget, check, failures) -> None:
    """Streams two clips through a server with the given lag; end-to-end p95 must stay under `budget` ms."""
    server = subprocess.Popen([os.path.join(ROOT, "build/linux/x86_64/release/retarget"), "teleop", "--bind", ENDPOINT,
                               "--slots", "2", "--quiet", "--lag", str(lag)],
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    before = len(failures)

    try:
        clips = [load_bvh(BVH + "210531/walk_forward_amateur_001__A001.bvh"),
                 load_bvh(BVH + "221010/jump_ff_180_R_001__A045.bvh")]
        with TeleopClient(ENDPOINT) as client:
            infos = [client.open_stream(10 + k, clip.skeleton, clip.fps, timeout=60) for k, clip in enumerate(clips)]
            check(all(i.status == 0 and i.robot == "g1_29dof" and len(i.dof_names) == 29 for i in infos), "streams opened")
            try:
                client.open_stream(99, clips[0].skeleton, clips[0].fps, timeout=10)
                check(False, "a third stream is refused when both slots are in use")
            except TeleopError as e:
                check("slots" in str(e), f"a third stream is refused when both slots are in use ({e})")

            frames = 240  # 2 s at 120 Hz
            start = time.monotonic()
            got = {10: [], 11: []}
            for f in range(frames):
                while time.monotonic() < start + f / 120:
                    m = client.receive(timeout=max(0.0, start + f / 120 - time.monotonic()))
                    if isinstance(m, messages.RobotFrame):
                        got[m.stream_id].append((m, time.monotonic()))
                for k, clip in enumerate(clips):
                    client.send_frame(10 + k, f, time.monotonic(), clip.joint_pos[f], clip.joint_quat[f])
            for sid in got:   # closing answers the frames still in a fixed-lag window
                client.close_stream(sid)
            deadline = time.monotonic() + 5
            while sum(map(len, got.values())) < 2 * frames and time.monotonic() < deadline:
                m = client.receive(timeout=0.5)
                if isinstance(m, messages.RobotFrame):
                    got[m.stream_id].append((m, time.monotonic()))
            for sid, items in got.items():
                seqs = [m.seq for m, _ in items]
                check(seqs == list(range(frames)), f"stream {sid}: every frame answered, in order")
                latency = np.array([(t - m.t_source) * 1000 for m, t in items[1:]])  # first frame: cold start
                check(np.percentile(latency, 95) < budget, f"stream {sid}: end-to-end p95 {np.percentile(latency, 95):.2f} ms < {budget} ms")
                check(not any(m.flags & 64 for m, _ in items), f"stream {sid}: no frames dropped")
                check(items[0][0].flags & 2 and not any(m.flags & 2 for m, _ in items[1:]), f"stream {sid}: one cold start")
                check(all(np.isfinite(m.dof_pos).all() and len(m.dof_pos) == 29 for m, _ in items), f"stream {sid}: finite poses")

            # Errors are reported, not fatal.
            client.send_frame(42, 0, time.monotonic(), clips[0].joint_pos[0], clips[0].joint_quat[0])
            m = client.receive(timeout=5)
            check(isinstance(m, messages.StreamInfo) and m.status == 1 and "not open" in m.message, "frame for an unknown stream rejected")
            client.open_stream(10, clips[0].skeleton, clips[0].fps)
            client.send_frame(10, frames, time.monotonic(), clips[0].joint_pos[0][:5], clips[0].joint_quat[0][:5])
            m = client.receive(timeout=5)
            check(isinstance(m, messages.StreamInfo) and m.status == 1, "malformed frame rejected")
            client.socket.send(b"garbage")
            m = client.receive(timeout=5)
            check(isinstance(m, messages.StreamInfo) and m.status == 1, "undecodable message rejected")
            client.close_stream(10)
            check(server.poll() is None, "server still running")
    finally:
        server.terminate()
        output = server.communicate(timeout=10)[0]
    if len(failures) > before:
        print(output)


if __name__ == "__main__":
    raise SystemExit(main())
