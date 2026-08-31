import ctypes
import getpass
import re
import sys
from pathlib import Path


def _effective_username() -> str:
    """Return the process-token user, which may differ inside a Windows sandbox."""
    if sys.platform != "win32":
        return getpass.getuser()

    size = ctypes.c_ulong(0)
    ctypes.windll.advapi32.GetUserNameW(None, ctypes.byref(size))
    buffer = ctypes.create_unicode_buffer(size.value)
    if not ctypes.windll.advapi32.GetUserNameW(buffer, ctypes.byref(size)):
        return getpass.getuser()
    return buffer.value


def pytest_configure(config) -> None:
    """Keep pytest temp trees separate for normal and sandboxed Windows users."""
    if config.option.basetemp is not None:
        return

    context = re.sub(r"[^a-z0-9_.-]", "-", _effective_username().lower())
    temp_root = Path("work")
    temp_root.mkdir(parents=True, exist_ok=True)
    config.option.basetemp = str(temp_root / f"pytest-temp-{context}")
