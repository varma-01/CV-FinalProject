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

import numpy as np

# Assuming 'image' is your 2D image array
# Calculate the histogram of the image
histogram, _ = np.histogram(image.flatten(), bins=256, range=(0, 256))

# Normalize the histogram to obtain probabilities
histogram_norm = histogram.astype(np.float32) / np.sum(histogram)

# Select an initial estimate of the threshold (average intensity of the image)
threshold = np.mean(image)

# Initialize variables for mean values
u1 = 0
u2 = 0

# Repeat until convergence
while True:
    # Partition the histogram into two groups R1 and R2 using the current threshold
    R1 = histogram_norm[:int(threshold)]
    R2 = histogram_norm[int(threshold):]
    
    # Calculate the mean gray values u1 and u2 of the partitions R1 and R2
    u1_new = np.sum(np.arange(len(R1)) * R1)
    u2_new = np.sum(np.arange(len(R2)) * R2)
    
    # Update the threshold
    new_threshold = (u1_new + u2_new) / 2
    
    # Check for convergence
    if np.abs(threshold - new_threshold) < 100:  # Adjust the convergence threshold as needed
        break
    
    threshold = new_threshold
    u1 = u1_new
    u2 = u2_new

print("The threshold value using Iterative thresholding is : ",threshold)

# Apply the final threshold to the image to obtain the binary image
binary_image = (image > threshold).astype(np.uint8) * 255

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