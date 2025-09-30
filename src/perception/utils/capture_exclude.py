import ctypes
from ctypes import wintypes

# Constants from Winuser.h
WDA_NONE = 0x00000000
WDA_MONITOR = 0x00000001      # legacy, not what we want
WDA_EXCLUDEFROMCAPTURE = 0x00000011  # Windows 10 2004+

user32 = ctypes.windll.user32
SetWindowDisplayAffinity = user32.SetWindowDisplayAffinity
SetWindowDisplayAffinity.argtypes = [wintypes.HWND, wintypes.DWORD]
SetWindowDisplayAffinity.restype = wintypes.BOOL

def set_window_exclude_from_capture(hwnd: int, enable: bool = True) -> bool:
    """Returns True on success. Requires Windows 10 2004+."""
    mode = WDA_EXCLUDEFROMCAPTURE if enable else WDA_NONE
    ok = SetWindowDisplayAffinity(hwnd, mode)
    return bool(ok)
