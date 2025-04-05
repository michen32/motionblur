import cv2
import numpy as np
import random

# Add a neon-like text overlay
def draw_neon_text(image, text, position, color, thickness=2, font_size=3):
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(image, text, position, font, font_size, color, thickness, cv2.LINE_AA)

def apply_motion_blur(image, kernel_size=15):
    kernel = np.zeros((kernel_size, kernel_size))
    kernel[int((kernel_size - 1)/2), :] = np.ones(kernel_size)
    kernel = kernel / kernel_size
    return cv2.filter2D(image, -1, kernel)

def apply_vhs_filter(image):
    height, width = image.shape[:2]

    # Add scan lines
    for y in range(0, height, 2):
        image[y:y+1, :] = image[y:y+1, :] * 0.5

    # Add slight RGB shift (chromatic aberration)
    b, g, r = cv2.split(image)
    r = np.roll(r, 1, axis=1)
    b = np.roll(b, -1, axis=1)
    image = cv2.merge([b, g, r])

    # Add noise
    noise = np.random.randint(0, 30, (height, width, 3), dtype='uint8')
    image = cv2.add(image, noise)

    # Slight horizontal wiggle
    offset = random.randint(-3, 3)
    image = np.roll(image, offset, axis=1)

    return image

def apply_glitch_effects(frame, glitch_intensity):
    if random.random() < glitch_intensity:
        frame = np.roll(frame, random.randint(-5, 5), axis=1)  # Horizontal wiggle
    if random.random() < glitch_intensity:
        frame = apply_vhs_filter(frame)
    return frame

# Fisheye lens effect
def apply_fisheye_effect(image):
    height, width = image.shape[:2]
    # Generate a grid of coordinates
    x = np.linspace(-1, 1, width)
    y = np.linspace(-1, 1, height)
    x, y = np.meshgrid(x, y)

    # Apply the fisheye effect (radial distortion)
    r = np.sqrt(x**2 + y**2)
    factor = 1.0 / (1 + 0.5 * r)  # Radial distortion factor
    x_distorted = x * factor
    y_distorted = y * factor

    # Map the distorted coordinates to the image pixels
    x_distorted = np.clip(x_distorted * (width // 2) + width // 2, 0, width - 1).astype(np.float32)
    y_distorted = np.clip(y_distorted * (height // 2) + height // 2, 0, height - 1).astype(np.float32)

    # Remap the image based on the distorted coordinates
    fisheye_image = cv2.remap(image, x_distorted, y_distorted, interpolation=cv2.INTER_LINEAR)
    return fisheye_image

# Start video capture
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

print("Press 'q' to quit.")
print("Press 'm' for Motion Blur, 'v' for VHS Filter, 'g' for Glitch Effects, 'f' for Fisheye, 't' for Neon Text.")

# Initialize all filters as off
apply_motion = False
apply_vhs = False
apply_glitch = False
apply_fisheye = False
apply_neon_text = False

glitch_intensity = 0.1  # Default glitch intensity

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to read from camera.")
        break

    # Apply filters based on user selection
    if apply_motion:
        frame = apply_motion_blur(frame, kernel_size=25)
    
    if apply_vhs:
        frame = apply_vhs_filter(frame)

    if apply_glitch:
        frame = apply_glitch_effects(frame, glitch_intensity)

    if apply_fisheye:
        frame = apply_fisheye_effect(frame)

    if apply_neon_text:
        # Add overlay text (Neon Flicker effect)
        draw_neon_text(frame, "ERROR", (100, 100), (random.randint(100, 255), random.randint(100, 255), random.randint(100, 255)))

    # Display the final frame
    cv2.imshow("VHS Art Experience", frame)

    # Handle key inputs
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break
    elif key == ord('m'):
        apply_motion = not apply_motion  # Toggle motion blur
    elif key == ord('v'):
        apply_vhs = not apply_vhs  # Toggle VHS effect
    elif key == ord('g'):
        apply_glitch = not apply_glitch  # Toggle glitch effect
    elif key == ord('f'):
        apply_fisheye = not apply_fisheye  # Toggle fisheye lens
    elif key == ord('t'):
        apply_neon_text = not apply_neon_text  # Toggle neon text overlay

cap.release()
cv2.destroyAllWindows()
