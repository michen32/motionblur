import cv2
import numpy as np
import random

# ---------- Filter Functions ----------
def apply_motion_blur(image, kernel_size=15):
    kernel = np.zeros((kernel_size, kernel_size))
    kernel[int((kernel_size - 1)/2), :] = np.ones(kernel_size)
    kernel = kernel / kernel_size
    return cv2.filter2D(image, -1, kernel)

def apply_vhs_filter(image):
    height, width = image.shape[:2]
    for y in range(0, height, 2):
        image[y:y+1, :] = image[y:y+1, :] * 0.5
    b, g, r = cv2.split(image)
    r = np.roll(r, 1, axis=1)
    b = np.roll(b, -1, axis=1)
    image = cv2.merge([b, g, r])
    noise = np.random.randint(0, 30, (height, width, 3), dtype='uint8')
    image = cv2.add(image, noise)
    offset = random.randint(-3, 3)
    image = np.roll(image, offset, axis=1)
    return image

def apply_glitch_effects(frame, glitch_intensity):
    if random.random() < glitch_intensity:
        frame = np.roll(frame, random.randint(-5, 5), axis=1)
    if random.random() < glitch_intensity:
        frame = apply_vhs_filter(frame)
    return frame

def simulate_eyesight_blur(image, vision="20/20"):
    vision_map = {
        "20/20": 1,
        "20/40": 5,
        "20/60": 10,
        "20/80": 15,
        "20/100": 20,
        "20/200": 30
    }
    blur_strength = vision_map.get(vision, 1)
    kernel = np.ones((blur_strength, blur_strength), np.float32) / (blur_strength ** 2)
    return cv2.filter2D(image, -1, kernel)

def apply_fisheye_effect(image):
    height, width = image.shape[:2]
    x = np.linspace(-1, 1, width)
    y = np.linspace(-1, 1, height)
    x, y = np.meshgrid(x, y)
    r = np.sqrt(x**2 + y**2)
    factor = 1.0 / (1 + 0.5 * r)
    x_distorted = x * factor
    y_distorted = y * factor
    x_distorted = np.clip(x_distorted * (width // 2) + width // 2, 0, width - 1).astype(np.float32)
    y_distorted = np.clip(y_distorted * (height // 2) + height // 2, 0, height - 1).astype(np.float32)
    return cv2.remap(image, x_distorted, y_distorted, interpolation=cv2.INTER_LINEAR)

def apply_snake_vision(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.applyColorMap(gray, cv2.COLORMAP_JET)

def apply_green_teal_filter(image):
    b, g, r = cv2.split(image)
    g = cv2.add(g, 60)
    b = cv2.add(b, 60)
    r = cv2.subtract(r, 60)
    return cv2.merge([b, g, r])

def simulate_vision_blur(image, vision_level='20/20'):
    blur_map = {
        '20/20': 1,
        '20/40': 5,
        '20/60': 10,
        '20/100': 15,
        '20/200': 25
    }
    kernel_size = blur_map.get(vision_level, 1)
    return cv2.GaussianBlur(image, (kernel_size*2+1, kernel_size*2+1), 0)

def apply_sharpening(img):
    sharpening_kernel = np.array([[-1, -1, -1],
                                  [-1, 9, -1],
                                  [-1, -1, -1]])
    return cv2.filter2D(img, -1, sharpening_kernel)

def bat_vision(img):
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    img_hsv[..., 0] = np.clip(img_hsv[..., 0], 0, 100)
    img_hsv[..., 1] = np.clip(img_hsv[..., 1], 40, 255)
    bat_img = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)
    bat_img = cv2.GaussianBlur(bat_img, (7, 7), 0)
    rows, cols = bat_img.shape[:2]
    x, y = np.meshgrid(np.linspace(-1, 1, cols), np.linspace(-1, 1, rows))
    d = np.sqrt(x**2 + y**2)
    vignette = 1 - np.clip(d, 0, 1)
    vignette = cv2.merge([vignette]*3)
    bat_img = cv2.multiply(bat_img.astype(np.float32), vignette.astype(np.float32))
    return np.clip(bat_img, 0, 255).astype(np.uint8)

# ---------- Menu Overlay Function ----------
def draw_menu_overlay(frame, active_filters):
    height, width = frame.shape[:2]
    menu_width = 300
    overlay = frame.copy()
    
    cv2.rectangle(overlay, (0, 0), (menu_width, height), (0, 0, 0), -1)
    alpha = 0.5
    frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

    y = 30
    spacing = 30
    cv2.putText(frame, "VHS Art Menu", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    y += spacing

    menu_items = [
        ("[m] Motion Blur", active_filters["motion"]),
        ("[v] Short-Sightedness", active_filters["vblur"]),
        ("[1-4] Vision Levels", None),
        ("[g] Glitch", active_filters["glitch"]),
        ("[f] Fisheye", active_filters["fisheye"]),
        ("[s] Sharpen", active_filters["sharpen"]),
        ("[p] Green/Teal", active_filters["green_teal"]),
        ("[d] Bat Vision", active_filters["bat"]),
        ("[h] Snake Vision", active_filters["snake_vision"]),
        ("[q] Quit", None),
    ]

    for item, active in menu_items:
        color = (0, 255, 0) if active else (255, 255, 255)
        cv2.putText(frame, item, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)
        y += spacing

    return frame

# ---------- Main Application ----------
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

print("Press 'q' to quit.")
print("Press 'o' to toggle menu.")  # This is the toggle option

apply_motion = False
apply_vhs = False
apply_glitch = False
apply_fisheye = False
apply_sharpen = False
apply_green_teal = False
apply_bat = False
apply_vblur = False
apply_snake_vision_live = False

glitch_intensity = 0.1
vision_mode = '20/20'
show_menu = True  # Menu is enabled by default

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to read from camera.")
        break

    # Apply active filters
    if apply_motion:
        frame = apply_motion_blur(frame, kernel_size=25)
    if apply_vhs:
        frame = apply_vhs_filter(frame)
    if apply_glitch:
        frame = apply_glitch_effects(frame, glitch_intensity)
    if apply_fisheye:
        frame = apply_fisheye_effect(frame)
    if apply_sharpen:
        frame = apply_sharpening(frame)
    if apply_green_teal:
        frame = apply_green_teal_filter(frame)
    if apply_bat:
        frame = bat_vision(frame)
    if apply_vblur:
        frame = simulate_vision_blur(frame, vision_mode)
    if apply_snake_vision_live:
        frame = apply_snake_vision(frame)

    active_filters = {
        "motion": apply_motion,
        "glitch": apply_glitch,
        "fisheye": apply_fisheye,
        "sharpen": apply_sharpen,
        "green_teal": apply_green_teal,
        "bat": apply_bat,
        "vblur": apply_vblur,
        "snake_vision": apply_snake_vision_live
    }

    if show_menu:
        frame = draw_menu_overlay(frame, active_filters)

    # Show frame with applied filters and menu overlay
    cv2.imshow("VHS Art Experience", frame)

    # Wait for key press
    key = cv2.waitKey(1) & 0xFF

    # Toggle menu on/off with 'o'
    if key == ord('o'):
        show_menu = not show_menu

    # Other keys for toggling filters and vision modes
    elif key == ord('q'):
        break
    elif key == ord('m'):
        apply_motion = not apply_motion
    elif key == ord('g'):
        apply_glitch = not apply_glitch
    elif key == ord('f'):
        apply_fisheye = not apply_fisheye
    elif key == ord('s'):
        apply_sharpen = not apply_sharpen
    elif key == ord('p'):
        apply_green_teal = not apply_green_teal
    elif key == ord('d'):
        apply_bat = not apply_bat
    elif key == ord('v'):
        apply_vblur = not apply_vblur
        print(f"Short-sightedness simulation {'enabled' if apply_vblur else 'disabled'}")
    elif key == ord('1'):
        vision_mode = '20/40'
        print("Vision mode set to 20/40")
    elif key == ord('2'):
        vision_mode = '20/60'
        print("Vision mode set to 20/60")
    elif key == ord('3'):
        vision_mode = '20/100'
        print("Vision mode set to 20/100")
    elif key == ord('4'):
        vision_mode = '20/200'
        print("Vision mode set to 20/200")
    elif key == ord('h'):
        apply_snake_vision_live = not apply_snake_vision_live
        print(f"snake_vision effect {'enabled' if apply_snake_vision_live else 'disabled'}")

cap.release()
cv2.destroyAllWindows()
