# ==========================================
# Digital Image Processing - Lab Task 01
# File: basics.py
# ==========================================

# ------------------------------------------
# Exercise 1: Variables and Data Types
# ------------------------------------------
print("--- Exercise 1: Variables and Data Types ---")
width = 640
height = 480
image_name = "sample.png"
is_color = True

# Printing value and data type for each variable
print("width:", width, "| Type:", type(width))
print("height:", height, "| Type:", type(height))
print("image_name:", image_name, "| Type:", type(image_name))
print("is_color:", is_color, "| Type:", type(is_color))

# Calculating total number of pixels in the image
total_pixels = width * height
print("Total Pixels:", total_pixels)
print()

# ------------------------------------------
# Exercise 2: if, elif, and else
# ------------------------------------------
print("--- Exercise 2: if, elif, and else ---")
test_values = [-1, 0, 127, 128, 255, 256]

for pixel in test_values:
    # Check for invalid range outside [0, 255]
    if pixel < 0 or pixel > 255:
        label = "Invalid"
    # Check if pixel is bright (128 or above)
    elif pixel >= 128:
        label = "Bright pixel"
    # Otherwise pixel is dark
    else:
        label = "Dark pixel"
    print(f"Pixel Value: {pixel:3d} -> Label: {label}")
print()


# ------------------------------------------
# Exercise 3: for and while Loops
# ------------------------------------------
print("--- Exercise 3: for and while Loops ---")
# 1. Using a for loop with threshold checking
pixel_list = [20, 130, 255]
threshold = 128

print("For Loop Results:")
for val in pixel_list:
    if val >= threshold:
        label = "Bright pixel"
    else:
        label = "Dark pixel"
    print(f"Value: {val} -> {label}")

# 2. Using a while loop to print numbers 0, 1, and 2
print("While Loop Results:")
counter = 0
while counter < 3:
    print("Counter:", counter)
    counter += 1  # Increment counter to avoid infinite loop
print()


# ------------------------------------------
# Exercise 4: Functions
# ------------------------------------------
print("--- Exercise 4: Functions ---")

def is_bright(pixel, threshold=128):
    """
    Checks if a pixel intensity is at or above a given threshold.
    Returns True if bright, False otherwise.
    """
    if pixel >= threshold:
        return True
    else:
        return False

# Testing function calls
res1 = is_bright(127)           # Default threshold 128 -> False
res2 = is_bright(128)           # Default threshold 128 -> True
res3 = is_bright(200, 210)      # Custom threshold 210 -> False

print("is_bright(127)      ->", res1)
print("is_bright(128)      ->", res2)
print("is_bright(200, 210) ->", res3)
print()


# ------------------------------------------
# Exercise 5: Lists and Tuples
# ------------------------------------------
print("--- Exercise 5: Lists and Tuples ---")
# List operations
pixels = [0, 128, 255]
print("Original list:", pixels)
print("First item:", pixels[0])

# Replace middle item (index 1) with 150
pixels[1] = 150

# Append 64 to the end of the list
pixels.append(64)

print("Updated list:", pixels)
print("Length of list:", len(pixels))

# Tuple operations
size = (640, 480)
w, h = size  # Unpacking tuple
print(f"Unpacked Tuple -> Width: {w}, Height: {h}")

# Explanation of Mutability:
# List items can be changed (mutable) because lists are designed as dynamic containers.
# Tuple items cannot be changed (immutable) because tuples are fixed sequence records designed for safety.
print()


# ------------------------------------------
# Exercise 6: Dictionaries and Sets
# ------------------------------------------
print("--- Exercise 6: Dictionaries and Sets ---")
# Dictionary operations
image = {"width": 640, "height": 480}
image["mode"] = "RGB"  # Adding new key-value pair

print("Dictionary Width:", image["width"])
print("Dictionary Mode:", image["mode"])

# Set operations (removing duplicate labels)
label_list = ["cat", "dog", "cat", "bird"]
unique_labels = set(label_list)

print("Unique labels set:", unique_labels)
print("Number of unique labels:", len(unique_labels))
