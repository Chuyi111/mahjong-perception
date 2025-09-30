import win32gui
import win32con
from typing import Optional, Tuple, Sequence
import ctypes

def set_process_dpi_aware() -> None:
    try:
        ctypes.windll.user32.SetProcessDPIAware()
    except Exception:
        pass

def _title_matches(title: str, needles: Sequence[str]) -> bool:
    t = title.lower()
    return any(n.lower() in t for n in needles)

def get_client_rect_in_screen(hwnd) -> Tuple[int, int, int, int]:
    cl = win32gui.GetClientRect(hwnd)
    w, h = cl[2] - cl[0], cl[3] - cl[1]
    left, top = win32gui.ClientToScreen(hwnd, (0, 0))
    return (left, top, left + w, top + h)

def find_window_by_titles(*title_hints: str) -> Optional[Tuple[int, Tuple[int,int,int,int]]]:
    """Return (hwnd, client_rect_in_screen) for the first matching window."""
    res = {"pair": None}
    def enum_handler(hwnd, _):
        if not win32gui.IsWindowVisible(hwnd) or win32gui.IsIconic(hwnd):
            return
        title = win32gui.GetWindowText(hwnd)
        if title and _title_matches(title, title_hints):
            rect = get_client_rect_in_screen(hwnd)
            res["pair"] = (hwnd, rect)
            raise StopIteration
    try:
        win32gui.EnumWindows(enum_handler, None)
    except StopIteration:
        pass
    return res["pair"]

def set_window_topmost(hwnd: int, topmost: bool) -> None:
    insert_after = win32con.HWND_TOPMOST if topmost else win32con.HWND_NOTOPMOST
    # Keep current position/size; only change z-order
    win32gui.SetWindowPos(
        hwnd, insert_after, 0, 0, 0, 0,
        win32con.SWP_NOMOVE | win32con.SWP_NOSIZE | win32con.SWP_NOACTIVATE
    )
