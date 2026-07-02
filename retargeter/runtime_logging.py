from __future__ import annotations


def configure_native_runtime_logging(*, quiet: bool = True) -> None:
    """Configure optional native runtimes without making them required imports."""
    try:
        import warp
    except ImportError:
        return
    config = getattr(warp, "config", None)
    if config is not None and hasattr(config, "quiet"):
        config.quiet = bool(quiet)
