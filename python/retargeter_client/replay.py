"""Replays BVH clips to the teleop server in real time (one stream per clip) and reports latency.

    python -m retargeter_client.replay clip.bvh [more.bvh ...] [--rate 1] [--out frames.npz]
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from . import FLAGS, messages
from .bvh import load_bvh
from .client import TeleopClient


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("bvh", nargs="+")
    p.add_argument("--endpoint", default="tcp://127.0.0.1:5560")
    p.add_argument("--rate", type=float, default=1.0, help="playback speed (1: real time; 0: as fast as possible)")
    p.add_argument("--seconds", type=float, default=None, help="replay at most this much of each clip")
    p.add_argument("--loop", action="store_true", help="repeat the clips until interrupted")
    p.add_argument("--first-id", type=int, default=0, help="stream id of the first clip")
    p.add_argument("--out", help="write the received robot frames (per stream) to this .npz")
    args = p.parse_args(argv)

    clips = [load_bvh(f) for f in args.bvh]
    with TeleopClient(args.endpoint) as client:
        ids = [args.first_id + i for i in range(len(clips))]
        for i, clip in zip(ids, clips):
            info = client.open_stream(i, clip.skeleton, clip.fps)
            print(f"stream {i}: {clip.frames} frames at {clip.fps:g} Hz -> {info.robot} ({len(info.dof_names)} dofs)")
        frames = [clip.frames if args.seconds is None else min(clip.frames, int(args.seconds * clip.fps)) for clip in clips]
        schedule = sorted((f / clip.fps, k, f) for k, clip in enumerate(clips) for f in range(frames[k]))
        received = {i: [] for i in ids}
        sent = 0
        start = time.monotonic()
        try:
            while True:
                for t, k, f in schedule:
                    if args.rate > 0:
                        wait = start + t / args.rate - time.monotonic()
                        # Collect replies while waiting for the next frame's time.
                        while wait > 0:
                            m = client.receive(timeout=wait)
                            if isinstance(m, messages.RobotFrame):
                                received[m.stream_id].append((m, time.monotonic()))
                            elif isinstance(m, messages.StreamInfo) and m.status:
                                print("server:", m.message)
                            wait = start + t / args.rate - time.monotonic()
                    clip = clips[k]
                    client.send_frame(ids[k], sent, time.monotonic(), clip.joint_pos[f], clip.joint_quat[f])
                    sent += 1
                if not args.loop:
                    break
                start = time.monotonic()
        except KeyboardInterrupt:
            pass
        # Closing a stream answers the frames still in a fixed-lag window.
        for i in ids:
            client.close_stream(i)
        deadline = time.monotonic() + 5
        while sum(len(v) for v in received.values()) < sent and time.monotonic() < deadline:
            m = client.receive(timeout=0.5)
            if isinstance(m, messages.RobotFrame):
                received[m.stream_id].append((m, time.monotonic()))
        wall = time.monotonic() - start

    total = sum(len(v) for v in received.values())
    print(f"sent {sent} frames, received {total} robot frames in {wall:.2f} s")
    for i, items in received.items():
        if not items:
            continue
        e2e = np.array([(arrival - m.t_source) * 1000 for m, arrival in items])
        server = np.array([(m.t_emit - m.t_receive) * 1000 for m, _ in items])
        flags = {}
        for m, _ in items:
            for bit, name in FLAGS.items():
                if m.flags & bit:
                    flags[name] = flags.get(name, 0) + 1
        print(f"stream {i}: {len(items)} frames; end-to-end latency ms p50 {np.percentile(e2e, 50):.2f} "
              f"p95 {np.percentile(e2e, 95):.2f} max {e2e.max():.2f}; server receive->emit p50 {np.percentile(server, 50):.2f}"
              f"; flags {flags or 'none'}")
    if args.out:
        arrays = {}
        for i, items in received.items():
            if items:
                arrays[f"s{i}_seq"] = np.array([m.seq for m, _ in items])
                arrays[f"s{i}_root_pos"] = np.stack([m.root_pos for m, _ in items])
                arrays[f"s{i}_root_quat"] = np.stack([m.root_quat for m, _ in items])
                arrays[f"s{i}_dof_pos"] = np.stack([m.dof_pos for m, _ in items])
                arrays[f"s{i}_flags"] = np.array([m.flags for m, _ in items])
                arrays[f"s{i}_latency_ms"] = np.array([(a - m.t_source) * 1000 for m, a in items])
        np.savez(args.out, **arrays)
        print("wrote", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
