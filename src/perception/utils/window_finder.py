import win32gui
from typing import Optional, Tuple

def _match(title: str, needle: str) -> bool:
    return needle.lower() in title.lower()

def find_window_rect(title_contains: str = "Mahjong Soul") -> Optional[Tuple[int,int,int,int]]:
    """Return (left, top, right, bottom) of the first matching window’s RECT."""
    target = {"rect": None}
    def enum_handler(hwnd, _):
        if win32gui.IsWindowVisible(hwnd):
            title = win32gui.GetWindowText(hwnd)
            if _match(title, title_contains):
                rect = win32gui.GetWindowRect(hwnd)
                target["rect"] = rect
                # stop enumeration
                return
    win32gui.EnumWindows(enum_handler, None)
    return target["rect"]
