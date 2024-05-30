import torch
import psutil
import logging
import traceback

logger = logging.getLogger()

def get_ram():
    mem = psutil.virtual_memory()
    free = mem.available / 1024 ** 3
    total = mem.total / 1024 ** 3
    total_cubes = 24
    free_cubes = int(total_cubes * free / total)
    return f'RAM:  {total - free:.1f}/{total:.1f}GB\t RAM: [' + (total_cubes - free_cubes) * '▮' + free_cubes * '▯' + ']'


def get_vram(device=None):
    device = device if device else torch.cuda.current_device()
    if device == "cpu":
        return 'No GPU available'
    try:
        free = torch.cuda.mem_get_info(device)[0] / 1024 ** 3
        total = torch.cuda.mem_get_info(device)[1] / 1024 ** 3
        total_cubes = 24
        free_cubes = int(total_cubes * free / total)
        return f'device:{device}\nVRAM: {total - free:.1f}/{total:.1f}GB\t VRAM:[' + (
                total_cubes - free_cubes) * '▮' + free_cubes * '▯' + ']'
    except ValueError:
        logger.debug(f"device: {device}, {torch.cuda.is_available()=}")
        logger.error(f"Error: {traceback.format_exc()}")
        return 'No GPU available'
