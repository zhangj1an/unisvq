"""Device-agnostic utilities for NPU/CUDA.

Import from this module instead of using torch.cuda / torch.npu directly.
"""

import torch


def is_npu_available() -> bool:
    """Check if Ascend NPU is available."""
    try:
        import torch_npu  # noqa: F401
        return torch.npu.is_available()
    except (ImportError, AttributeError):
        return False


def get_device() -> torch.device:
    """Return the best available device (NPU > CUDA > CPU)."""
    if is_npu_available():
        return torch.device("npu:0")
    elif torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


def get_device_name() -> str:
    """Return device type string for torch.autocast."""
    if is_npu_available():
        return "npu"
    elif torch.cuda.is_available():
        return "cuda"
    return "cpu"


def empty_cache():
    """Free unused memory on the current device."""
    import gc
    gc.collect()
    if is_npu_available():
        torch.npu.empty_cache()
    elif torch.cuda.is_available():
        torch.cuda.empty_cache()


def autocast(enabled: bool = True, dtype=None):
    """Return an autocast context manager for the current device."""
    if dtype is None:
        if is_npu_available() or torch.cuda.is_available():
            dtype = torch.bfloat16
        else:
            dtype = torch.float32
    device_type = get_device_name()
    if device_type == "npu":
        return torch.npu.amp.autocast(enabled=enabled, dtype=dtype)
    elif device_type == "cuda":
        return torch.cuda.amp.autocast(enabled=enabled, dtype=dtype)
    else:
        # CPU: no autocast, just a no-op context
        import contextlib
        return contextlib.nullcontext()


def mem_get_info(device=None):
    """Return (free, total) memory in bytes for the device."""
    if is_npu_available():
        return torch.npu.mem_get_info(device)
    elif torch.cuda.is_available():
        return torch.cuda.mem_get_info(device)
    raise RuntimeError("No NPU or CUDA device available")


def synchronize(device=None):
    """Synchronize device execution."""
    if is_npu_available():
        torch.npu.synchronize(device)
    elif torch.cuda.is_available():
        torch.cuda.synchronize(device)


def reset_peak_memory_stats(device=None):
    """Reset peak memory stats."""
    if is_npu_available():
        torch.npu.reset_peak_memory_stats(device)
    elif torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)


def max_memory_allocated(device=None):
    """Return peak allocated memory in bytes."""
    if is_npu_available():
        return torch.npu.max_memory_allocated(device)
    elif torch.cuda.is_available():
        return torch.cuda.max_memory_allocated(device)
    return 0


def to_device(tensor_or_module, device=None):
    """Move tensor/module to the best available device."""
    if device is None:
        device = get_device()
    return tensor_or_module.to(device)
