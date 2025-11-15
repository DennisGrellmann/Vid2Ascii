import cv2
import time
import numpy as np

# =========================
# CONFIG
# =========================
ASCII_CHARS_BASE = "@%#*+=-:. "
NEW_WIDTH = 200      # starting width (slider overrides this)

SHOW_ORIGINAL = False
SHOW_TERMINAL = False
SHOW_ASCII_WINDOW = True
SHOW_FPS = True

# Toggles (can change live)
INVERT_GRAYSCALE_MAP = False
INVERT_DISPLAY_ONLY = False
INVERT_ASCII_CHARS = False

# Profile index (for 'p' key)
PROFILE_INDEX = 0

# Custom profiles:
# p = cycle through these
PROFILES = [
    {
        "name": "Cinematic Low-Res",   # chunky, stable, film-y
        "width": 120,
        "invert_map": False,
        "invert_display": False,
        "invert_chars": False,
    },
    {
        "name": "Balanced Live",       # your “do everything” default
        "width": 200,
        "invert_map": False,
        "invert_display": False,
        "invert_chars": False,
    },
    {
        "name": "Hi-Detail Portrait",  # maximum detail, heavier load
        "width": 300,
        "invert_map": False,
        "invert_display": False,
        "invert_chars": False,
    },
    {
        "name": "Bright Edge Sketch",  # “super bright edge mode”
        # width still decent for detail, but a bit under max
        "width": 220,
        # invert mapping + display + chars for glowy edge look
        "invert_map": True,
        "invert_display": True,
        "invert_chars": True,
    },
]


# =========================
# Helpers
# =========================
def nothing(x):
    pass


def get_ascii_chars():
    return "".join(reversed(ASCII_CHARS_BASE)) if INVERT_ASCII_CHARS else ASCII_CHARS_BASE


def apply_profile(idx):
    """Apply a profile by index: set width slider + invert flags."""
    global INVERT_GRAYSCALE_MAP, INVERT_DISPLAY_ONLY, INVERT_ASCII_CHARS
    profile = PROFILES[idx % len(PROFILES)]
    print(f"\n[PROFILE] {profile['name']}")
    INVERT_GRAYSCALE_MAP = profile["invert_map"]
    INVERT_DISPLAY_ONLY = profile["invert_display"]
    INVERT_ASCII_CHARS = profile["invert_chars"]
    # Update the slider if it exists
    try:
        cv2.setTrackbarPos("Size", "ASCII Webcam", profile["width"])
    except cv2.error:
        # If window/trackbar not ready yet, ignore
        pass


def frame_to_ascii_and_depth(gray, new_width):
    h, w = gray.shape
    aspect_ratio = h / w
    new_height = int(aspect_ratio * new_width * 0.55)

    resized = cv2.resize(gray, (new_width, new_height), interpolation=cv2.INTER_AREA)
    resized = cv2.GaussianBlur(resized, (3, 3), 0)

    if INVERT_GRAYSCALE_MAP:
        resized = 255 - resized

    ascii_chars = get_ascii_chars()
    chars = np.array(list(ascii_chars))

    indices = (resized.astype(np.float32) / 255 * (len(ascii_chars) - 1)).astype(np.int32)
    ascii_array = chars[indices]
    lines = ["".join(row) for row in ascii_array]

    depth_gray = resized
    return lines, depth_gray


def ascii_to_image_per_char(lines, depth_gray, char_w=8, char_h=12, fps=None):
    h = len(lines)
    if h == 0:
        return None
    w = len(lines[0])

    dh, dw = depth_gray.shape
    if dh != h or dw != w:
        depth_gray = cv2.resize(depth_gray, (w, h), interpolation=cv2.INTER_AREA)

    canvas = np.zeros((h * char_h, w * char_w, 3), dtype=np.uint8)

    for y, line in enumerate(lines):
        for x, ch in enumerate(line):
            v = int(depth_gray[y, x])
            color = (v, v, v)
            org = (x * char_w, (y + 1) * char_h - 2)
            cv2.putText(canvas, ch, org, cv2.FONT_HERSHEY_PLAIN, 0.8, color, 1, cv2.LINE_AA)

    if SHOW_FPS and fps is not None:
        cv2.putText(canvas, f"FPS: {fps:.1f}", (5, 15),
                    cv2.FONT_HERSHEY_PLAIN, 1.0, (220, 220, 220), 1)

    if INVERT_DISPLAY_ONLY:
        canvas = cv2.bitwise_not(canvas)

    return canvas


# =========================
# MAIN
# =========================
def main():
    global INVERT_GRAYSCALE_MAP, INVERT_DISPLAY_ONLY, INVERT_ASCII_CHARS
    global SHOW_TERMINAL, SHOW_ORIGINAL, PROFILE_INDEX

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        return

    if SHOW_ASCII_WINDOW:
        cv2.namedWindow("ASCII Webcam", cv2.WINDOW_NORMAL)
        cv2.createTrackbar("Size", "ASCII Webcam", NEW_WIDTH, 400, nothing)
        # Apply initial profile to sync slider & flags
        apply_profile(PROFILE_INDEX)

    prev_time = time.time()
    fps = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Dynamic width from slider
        size_val = cv2.getTrackbarPos("Size", "ASCII Webcam")
        CURRENT_WIDTH = max(30, size_val)

        ascii_lines, depth_gray = frame_to_ascii_and_depth(gray, CURRENT_WIDTH)

        now = time.time()
        dt = now - prev_time
        prev_time = now
        if dt > 0:
            instant_fps = 1.0 / dt
            fps = 0.9 * fps + 0.1 * instant_fps if fps > 0 else instant_fps

        if SHOW_TERMINAL:
            print("\033[H\033[J", end="")
            print("\n".join(ascii_lines))

        if SHOW_ASCII_WINDOW:
            ascii_img = ascii_to_image_per_char(ascii_lines, depth_gray, fps=fps)
            if ascii_img is not None:
                cv2.imshow("ASCII Webcam", ascii_img)

        if SHOW_ORIGINAL:
            cv2.imshow("Webcam", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('m'):
            INVERT_GRAYSCALE_MAP = not INVERT_GRAYSCALE_MAP
            print(f"INVERT_GRAYSCALE_MAP = {INVERT_GRAYSCALE_MAP}")
        elif key == ord('i'):
            INVERT_DISPLAY_ONLY = not INVERT_DISPLAY_ONLY
            print(f"INVERT_DISPLAY_ONLY = {INVERT_DISPLAY_ONLY}")
        elif key == ord('c'):
            INVERT_ASCII_CHARS = not INVERT_ASCII_CHARS
            print(f"INVERT_ASCII_CHARS = {INVERT_ASCII_CHARS}")
        elif key == ord('t'):
            SHOW_TERMINAL = not SHOW_TERMINAL
            print(f"SHOW_TERMINAL = {SHOW_TERMINAL}")
        elif key == ord('o'):
            SHOW_ORIGINAL = not SHOW_ORIGINAL
            print(f"SHOW_ORIGINAL = {SHOW_ORIGINAL}")
        elif key == ord('p'):
            # Profile button: cycle profiles
            PROFILE_INDEX = (PROFILE_INDEX + 1) % len(PROFILES)
            apply_profile(PROFILE_INDEX)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
