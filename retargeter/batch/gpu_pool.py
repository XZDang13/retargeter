from __future__ import annotations


def parse_gpu_ids(raw: str | None) -> list[int]:
    if raw is None:
        return []
    value = str(raw).strip()
    if not value or value.lower() == "cpu":
        return []

    gpu_ids: list[int] = []
    for part in value.split(","):
        token = part.strip()
        if not token:
            continue
        try:
            gpu_id = int(token)
        except ValueError as exc:
            raise ValueError(f"Invalid GPU id {token!r} in {raw!r}.") from exc
        if gpu_id < 0:
            raise ValueError(f"GPU ids must be non-negative, got {gpu_id}.")
        gpu_ids.append(gpu_id)
    return gpu_ids


def assign_device(worker_index: int, gpu_ids: list[int], processes_per_gpu: int) -> str:
    if worker_index < 0:
        raise ValueError("worker_index must be non-negative.")
    if processes_per_gpu <= 0:
        raise ValueError("processes_per_gpu must be positive.")
    if not gpu_ids:
        return "cpu"
    gpu_index = (worker_index // int(processes_per_gpu)) % len(gpu_ids)
    return f"cuda:{gpu_ids[gpu_index]}"
