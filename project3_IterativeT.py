import matplotlib.pyplot as plt
from skimage import io, filters
import numpy as np

# Open the image file in binary mode
with open('/Users/reenabardeskar/Downloads/CVProjects/Project3/test2.img', 'rb') as f:
    # Read the image data
    image_data = f.read()

# Convert the image data to a NumPy array
# Assuming the image is grayscale
image = np.frombuffer(image_data, dtype=np.uint8)

# Reshape the array to represent the image dimensions
# Replace 'width' and 'height' with the actual dimensions of your image
width = 512
height = 512
image = image[:width * height]  # Truncate excess data if necessary
image = image.reshape((height, width))

# Calculate the histogram
histogram, _ = np.histogram(image.flatten(), bins=256, range=(0, 256))

# Normalize the histogram to obtain probabilities
histogram_norm = histogram.astype(np.float32) / np.sum(histogram)

# Initialize variables
max_variance = 0
optimal_threshold = 0

# Iterate through all possible threshold values
for threshold in range(1, 256):
    # Calculate probabilities of the two classes separated by the threshold
    prob_background = np.sum(histogram_norm[:threshold])
    prob_foreground = np.sum(histogram_norm[threshold:])
    
    # Calculate mean intensities of the two classes
    mean_background = np.sum(np.arange(threshold) * histogram_norm[:threshold]) / prob_background
    mean_foreground = np.sum(np.arange(threshold, 256) * histogram_norm[threshold:]) / prob_foreground
    
    # Calculate between-class variance
    variance = prob_background * prob_foreground * (mean_background - mean_foreground)**2
    
    # Update max_variance and optimal_threshold if necessary
    if variance > max_variance:
        max_variance = variance
        optimal_threshold = threshold

# Apply the optimal threshold to the image
binary_image = (image > optimal_threshold).astype(np.uint8) * 255

# Display the original and segmented images
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.imshow(image, cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(binary_image, cmap='gray')
plt.title('Image Using Iterative Thresholding')
plt.axis('off')

plt.show()
