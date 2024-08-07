import numpy as np
import matplotlib.pyplot as plt


def calculate_dual_thresholds(image_data):
    # Calculate the histogram of the image
    histogram, _ = np.histogram(image_data.flatten(), bins=256, range=(0, 256))

    # Calculate the total number of pixels
    total_pixels = width * height

    # Calculate the cumulative sum and cumulative sum of squares
    sum_t = np.sum(np.arange(256) * histogram)
    sum_t2 = np.sum((np.arange(256) ** 2) * histogram)

    # Initialize variables for thresholds and between-class variance
    optimal_threshold_1 = 0
    optimal_threshold_2 = 0
    max_variance = 0

    # Iterate over all possible thresholds to find the optimal ones
    for t in range(1, 256):
        # Calculate the weight and mean of the background
        w0 = np.sum(histogram[:t]) / total_pixels
        if w0 == 0:
            continue  # Skip if the weight is zero to avoid division by zero
        u0 = np.sum(np.arange(t) * histogram[:t]) / np.sum(histogram[:t])

        # Calculate the weight and mean of the foreground
        w1 = np.sum(histogram[t:]) / total_pixels
        if w1 == 0:
            break  # Stop iterating if the weight is zero
        u1 = np.sum(np.arange(t, 256) * histogram[t:]) / np.sum(histogram[t:])

        # Calculate between-class variance
        variance = w0 * w1 * ((u1 - u0) ** 2)

        # Update optimal thresholds if variance is higher
        if variance > max_variance:
            optimal_threshold_1 = t
            max_variance = variance
        elif optimal_threshold_1 != 0:
            optimal_threshold_2 = t
            break

    return optimal_threshold_1, optimal_threshold_2


# Read the image data
image_path = "/Users/reenabardeskar/Downloads/CVProjects/Project3/test2.img"
with open(image_path, 'rb') as f:
    raw_data = f.read()

# Convert the raw data to a NumPy array
image_data = np.frombuffer(raw_data, dtype=np.uint8)

# Reshape the array to represent the image dimensions
width = 512
height = 512
image_data = image_data[:width * height]
image_data = image_data.reshape((height, width))

# Thresholds
T1, T2 = calculate_dual_thresholds(image_data)
print("Threshold 1:", T1, "Threshold 2: ", T2)

# Initialize the segmented image
segmented_image = np.zeros_like(image_data, dtype=np.uint8)

# Perform region growing
size = (height // 2, width // 2)  # center of the image
stack = [size]

while stack:
    y, x = stack.pop()
    if segmented_image[y, x] == 0:
        if T1 <= image_data[y, x] <= T2:
            segmented_image[y, x] = 255
            # Check and add neighboring pixels to the stack using ternary operator
            stack.push((y - 1, x)) if y > 0 else None
            stack.push((y + 1, x)) if y < height - 1 else None
            stack.push((y, x - 1)) if x > 0 else None
            stack.push((y, x + 1)) if x < width - 1 else None

# Assign any remaining pixels in Region R2 to Region R3
segmented_image[(image_data > T2) & (segmented_image == 0)] = 255

# Display the original and segmented images
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.imshow(image_data, cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(segmented_image, cmap='gray')
plt.title('Image Using Region Growing Thresholding')
plt.axis('off')

plt.show()


