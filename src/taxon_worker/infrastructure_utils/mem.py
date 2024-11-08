import logging
import time
import traceback
from typing import Optional, Union

import psutil
import torch

logger = logging.getLogger()


def get_torch_cuda_device_if_available(device: Union[int, str] = 0) -> torch.device:
    """Set device if available."""
    logger.debug(f"requested device: {device}")
    # logger.debug(f"{traceback.format_stack()=}")
    if isinstance(device, str):
        if device.startswith("cuda"):
            device = int(device.split(":")[-1])
        elif device == "cpu":
            device = "cpu"
        else:
            logger.warning(f"Unknown device: {device}")
            device = 0
    if torch.cuda.is_available():
        new_device = torch.device(device)
    else:
        new_device = torch.device("cpu")
    logger.debug(f"new_device: {new_device}")
    print(f"new_device: {new_device}")
    return new_device


def get_ram():
    """Get visualized RAM usage in GB."""
    mem = psutil.virtual_memory()
    free = mem.available / 1024**3
    total = mem.total / 1024**3
    total_cubes = 24
    free_cubes = int(total_cubes * free / total)
    return (
        f"RAM:  {total - free:.1f}/{total:.1f}GB  RAM: ["
        + (total_cubes - free_cubes) * "▮"
        + free_cubes * "▯"
        + "]"
    )


def get_vram(device: Optional[torch.device] = None):
    """Get visualized VRAM usage in GB."""
    device = device if device else torch.cuda.current_device()
    if torch.device(device).type == "cpu":
        return "No GPU available"
    try:
        free = torch.cuda.mem_get_info(device)[0] / 1024**3
        total = torch.cuda.mem_get_info(device)[1] / 1024**3
        total_cubes = 24
        free_cubes = int(total_cubes * free / total)
        return (
            f"device:{device}    VRAM: {total - free:.1f}/{total:.1f}GB  VRAM:["
            + (total_cubes - free_cubes) * "▮"
            + free_cubes * "▯"
            + "]"
        )
    except ValueError:
        logger.debug(f"device: {device}, {torch.cuda.is_available()=}")
        logger.error(f"Error: {traceback.format_exc()}")
        return "No GPU available"


def wait_for_gpu_memory(required_memory_gb: float = 1.0, device: Union[int, str] = 0):
    """Wait until GPU memory is below threshold."""
    device = get_torch_cuda_device_if_available(device)

    # check if device is cpu
    if device.type == "cpu":
        logger.debug("No need to wait for CPU")
        return
    while torch.cuda.mem_get_info(device)[0] / 1024**3 > required_memory_gb:
        logger.debug(f"Waiting for {required_memory_gb} GB of GPU memory. " + get_vram(device))
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        time.sleep(5)
