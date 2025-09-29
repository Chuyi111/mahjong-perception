from perception.utils.window_finder import find_window_rect

def test_find_window_rect_runs():
    # Should not crash even if window doesn't exist
    rect = find_window_rect("DefinitelyNotARealWindowTitle")
    assert rect is None or (len(rect) == 4)
