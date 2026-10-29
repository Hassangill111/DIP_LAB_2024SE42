import os
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# STEP 2: LOAD THE IMAGE
# ==========================================
image_path = "image.jpg"

if not os.path.exists(image_path):
    raise FileNotFoundError(f"Error: The image file '{image_path}' was not found.")

# OpenCV loads images in BGR format
bgr_img = cv.imread(image_path)

print("--- STEP 2 & 3: IMAGE INSPECTION ---")
print(f"Image Shape (Height, Width, Channels): {bgr_img.shape}")
print(f"Image Data Type: {bgr_img.dtype}")

# ==========================================
# STEP 3: INSPECT THE IMAGE
# ==========================================
height, width, channels = bgr_img.shape
print(f"Height: {height} pixels")
print(f"Width: {width} pixels")
print(f"Number of Channels: {channels}")

# Access pixels [row, column]
top_left_pixel = bgr_img[0, 0]
center_pixel = bgr_img[height // 2, width // 2]

print(f"Top-Left Pixel (BGR): {top_left_pixel}")
print(f"Center Pixel (BGR): {center_pixel}")

# Min and Max values
print(f"Global Minimum Pixel Value: {np.min(bgr_img)}")
print(f"Global Maximum Pixel Value: {np.max(bgr_img)}")

for i, color_name in enumerate(['Blue', 'Green', 'Red']):
    c_min = np.min(bgr_img[:, :, i])
    c_max = np.max(bgr_img[:, :, i])
    print(f"{color_name} Channel - Min: {c_min}, Max: {c_max}")

# ==========================================
# STEP 4: COLOR CONVERSION & CHANNELS
# ==========================================
# Convert BGR to RGB for Matplotlib
rgb_img = cv.cvtColor(bgr_img, cv.COLOR_BGR2RGB)

# Convert to Grayscale
gray_img = cv.cvtColor(bgr_img, cv.COLOR_BGR2GRAY)

# Separate channels
r_channel = rgb_img[:, :, 0]
g_channel = rgb_img[:, :, 1]
b_channel = rgb_img[:, :, 2]

# Center Pixel to HEX conversion
c_row, c_col = height // 2, width // 2
center_rgb = rgb_img[c_row, c_col]
hex_val = "#{:02X}{:02X}{:02X}".format(center_rgb[0], center_rgb[1], center_rgb[2])
print(f"\nCenter Pixel RGB: {center_rgb}")
print(f"Center Pixel HEX Value: {hex_val}")

# Plotting 2x3 grid
plt.figure(figsize=(12, 8))

plt.subplot(2, 3, 1)
plt.imshow(rgb_img)
plt.title("RGB Image")
plt.axis('off')

plt.subplot(2, 3, 2)
plt.imshow(r_channel, cmap='gray')
plt.title("Red Channel")
plt.axis('off')

plt.subplot(2, 3, 3)
plt.imshow(g_channel, cmap='gray')
plt.title("Green Channel")
plt.axis('off')

plt.subplot(2, 3, 4)
plt.imshow(b_channel, cmap='gray')
plt.title("Blue Channel")
plt.axis('off')

plt.subplot(2, 3, 5)
plt.imshow(gray_img, cmap='gray')
plt.title("Grayscale Image")
plt.axis('off')

plt.subplot(2, 3, 6)
plt.imshow(bgr_img)
plt.title("Unconverted BGR Display")
plt.axis('off')

plt.tight_layout()
plt.show()

# ==========================================
# STEP 5: CROP, RESIZE, AND SAVE
# ==========================================
# Crop exact 100x100 region from center
y_start = height // 2 - 50
y_end = height // 2 + 50
x_start = width // 2 - 50
x_end = width // 2 + 50

crop_img = rgb_img[y_start:y_end, x_start:x_end]
print(f"\nCropped Image Shape: {crop_img.shape}")

# Resize to 200x200
resized_img = cv.resize(crop_img, (200, 200))
print(f"Resized Image Shape: {resized_img.shape}")

# Display original, crop, resized side-by-side
plt.figure(figsize=(10, 4))

plt.subplot(1, 3, 1)
plt.imshow(rgb_img)
plt.title("Original RGB")
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(crop_img)
plt.title("Cropped (100x100)")
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(resized_img)
plt.title("Resized (200x200)")
plt.axis('off')

plt.tight_layout()
plt.show()

# Save resized crop as BGR
resized_bgr = cv.cvtColor(resized_img, cv.COLOR_RGB2BGR)
save_status = cv.imwrite("resized_crop.png", resized_bgr)
print(f"Was 'resized_crop.png' saved successfully? {save_status}")

# ==========================================
# STEP 6: CREATE A HISTOGRAM
# ==========================================
plt.figure(figsize=(8, 5))
plt.hist(gray_img.ravel(), bins=256, range=[0, 256], color='gray')
plt.title("Grayscale Image Histogram")
plt.xlabel("Pixel Intensity (0-255)")
plt.ylabel("Frequency")
plt.grid(True, linestyle='--', alpha=0.5)
plt.savefig("grayscale_histogram.png")
plt.show()

# ==========================================
# STEP 7: DEMONSTRATE BIT DEPTH
# ==========================================
# 8-bit gradient (0 to 255)
gradient_8bit = np.tile(np.linspace(0, 255, 256, dtype=np.uint8), (50, 1))

# Quantize levels
gradient_1bit = (gradient_8bit // 128) * 255
gradient_2bit = (gradient_8bit // 64) * 85
gradient_4bit = (gradient_8bit // 16) * 17

print("\n--- BIT DEPTH UNIQUE VALUE COUNTS ---")
print(f"8-bit Unique Values: {len(np.unique(gradient_8bit))}")
print(f"4-bit Unique Values: {len(np.unique(gradient_4bit))}")
print(f"2-bit Unique Values: {len(np.unique(gradient_2bit))}")
print(f"1-bit Unique Values: {len(np.unique(gradient_1bit))}")

plt.figure(figsize=(10, 6))

plt.subplot(4, 1, 1)
plt.imshow(gradient_8bit, cmap='gray')
plt.title("8-bit (256 Levels)")
plt.axis('off')

plt.subplot(4, 1, 2)
plt.imshow(gradient_4bit, cmap='gray')
plt.title("4-bit (16 Levels)")
plt.axis('off')

plt.subplot(4, 1, 3)
plt.imshow(gradient_2bit, cmap='gray')
plt.title("2-bit (4 Levels)")
plt.axis('off')

plt.subplot(4, 1, 4)
plt.imshow(gradient_1bit, cmap='gray')
plt.title("1-bit (2 Levels)")
plt.axis('off')

plt.tight_layout()
plt.show()