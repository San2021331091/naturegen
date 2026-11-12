"""Read-only environment report. Can run before installing project dependencies."""
import ctypes
import importlib.metadata
import json
import platform
import shutil
import subprocess
import sys

report = {'python': sys.version, 'platform': platform.platform(), 'cpu': platform.processor()}
for command in ('node', 'npm.cmd', 'nvidia-smi'):
    path = shutil.which(command)
    if path:
        result = subprocess.run([path, '--version'] if command != 'nvidia-smi' else [path], capture_output=True, text=True)
        report[command] = (result.stdout or result.stderr).strip()
    else:
        report[command] = 'not available'
if sys.platform == 'win32':
    buffer = ctypes.create_string_buffer(64)
    ctypes.cast(buffer, ctypes.POINTER(ctypes.c_ulong))[0] = 64
    if ctypes.windll.kernel32.GlobalMemoryStatusEx(buffer):
        report['ram_gib'] = round(int.from_bytes(buffer.raw[8:16], 'little') / 2**30, 2)
try:
    import torch
    report['torch'] = str(torch.__version__)
    report['cuda'] = torch.cuda.is_available()
    report['gpu'] = torch.cuda.get_device_name(0) if report['cuda'] else None
    report['vram_gib'] = torch.cuda.get_device_properties(0).total_memory / 2**30 if report['cuda'] else 0
except (ImportError, OSError) as error:
    report['torch_error'] = str(error)
print(json.dumps(report, indent=2))
