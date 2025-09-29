import time
import cv2
from perception.utils.window_finder import find_window_rect
from perception.capture.screen_capture import ScreenCapturer, annotate_hud

def main():
    rect = find_window_rect("Mahjong Soul") or find_window_rect("雀魂麻將") or find_window_rect("Chrome")  # fallback
    if not rect:
        print("Mahjong Soul window not found. Open it and try again.")
        return

    print("Capturing region:", rect)
    cap = ScreenCapturer(rect)

    last = time.time()
    frames = 0
    fps = 0.0

    while True:
        img = cap.grab()
        frames += 1
        now = time.time()
        if now - last >= 1.0:
            fps = frames / (now - last)
            frames = 0
            last = now

        hud = {"FPS": f"{fps:.1f}", "Size": f"{img.shape[1]}x{img.shape[0]}"}
        vis = annotate_hud(img, hud)

        cv2.imshow("Mahjong Perception - Capture", vis)
        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord('q'):
            break
        elif key == ord('s'):
            ts = int(now * 1000)
            cv2.imwrite(f"data/snap_{ts}.png", img)
            print(f"Saved data/snap_{ts}.png")

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
