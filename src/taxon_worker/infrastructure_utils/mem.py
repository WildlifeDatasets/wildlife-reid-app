import torch
import psutil
import logging
import traceback
from typing import Union, Optional

logger = logging.getLogger(__name__)


def get_torch_cuda_device_if_available(device: Union[int, str] = 0) -> torch.device:
    """Set device if available."""
    logger.debug(f"requested device: {device}")
    # logger.debug(f"{traceback.format_stack()=}")
    print(f"requested device: {device}")
    print(f"{traceback.format_stack()=}")
    if isinstance(device, str):
        device = int(device.split(":")[-1])
    if torch.cuda.is_available():
        new_device = torch.device(device)
    else:
        new_device = torch.device("cpu")
    logger.debug(f"new_device: {new_device}")
    print(f"new_device: {new_device}")
    return new_device


def get_ram():
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
    try:
        device = device if device else torch.cuda.current_device()
        if torch.device(device).type == "cpu":
            return "No GPU available"
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
    except (ValueError, RuntimeError):
        logger.debug(f"device: {device}, {torch.cuda.is_available()=}")
        logger.error(f"Error: {traceback.format_exc()}")
        return "No GPU available"
