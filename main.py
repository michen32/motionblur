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

# Start video capture
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

print("Press 'q' to quit, 'g' to increase glitch intensity.")

glitch_intensity = 0.1  # Default glitch intensity

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to read from camera.")
        break

    # Apply motion blur
    blurred = apply_motion_blur(frame, kernel_size=25)

    # Apply glitch effects (based on intensity)
    frame_with_glitches = apply_glitch_effects(blurred, glitch_intensity)

    # Add overlay text (Neon Flicker effect)
    draw_neon_text(frame_with_glitches, "ERROR", (100, 100), (random.randint(100, 255), random.randint(100, 255), random.randint(100, 255)))

    # Display the final frame
    cv2.imshow("VHS Art Experience", frame_with_glitches)

    # Handle key inputs
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break
    elif key == ord('g'):
        glitch_intensity = min(glitch_intensity + 0.05, 1.0)  # Increase glitch intensity

cap.release()
cv2.destroyAllWindows()
