"""ANSI color constants and per-setup label helpers."""

GREEN  = "\033[32m"
RED    = "\033[31m"
DIM    = "\033[2m"
OFF    = "\033[0m"
PURPLE = "\033[38;5;141m"
ORANGE = "\033[38;5;208m"


def setup_short(setup: str) -> str:
    if "mtier" in setup: return "Mtier"
    if "cpu"   in setup: return "CPU"
    if "hbm"   in setup: return "HBM"
    return setup


def setup_color(setup: str) -> str:
    if "mtier" in setup: return PURPLE
    if "cpu"   in setup: return ORANGE
    return ""


def vllm_label(port: int, gpus: str | None, setup: str) -> str:
    """`[vllm:8001:GPU0+Mtier]` colored by setup."""
    g = f"GPU{gpus}" if gpus is not None else "GPU?"
    return f"{setup_color(setup)}[vllm:{port}:{g}+{setup_short(setup)}]{OFF}"
