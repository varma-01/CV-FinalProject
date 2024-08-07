import matplotlib.pyplot as plt
from skimage import io, filters
import numpy as np

# Open the image file in binary mode
with open('/Users/reenabardeskar/Downloads/CVProjects/Project3/test.img', 'rb') as f:
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

# Find the two highest local maxima
sorted_indices = np.argsort(histogram_norm)[::-1]
maxima_indices = sorted_indices[:2]
maxima_values = histogram_norm[maxima_indices]

# Find the minimum distance between the two maxima
min_distance = np.abs(maxima_indices[0] - maxima_indices[1])

# Find the lowest point between the two maxima
if maxima_indices[0] < maxima_indices[1]:
    region = histogram_norm[maxima_indices[0]:maxima_indices[1]+1]
else:
    region = histogram_norm[maxima_indices[1]:maxima_indices[0]+1]
lowest_point_index = np.argmin(region)
if maxima_indices[0] < maxima_indices[1]:
    lowest_point_index = maxima_indices[0] + lowest_point_index
else:
    lowest_point_index = maxima_indices[1] + lowest_point_index
lowest_point_value = histogram_norm[lowest_point_index]

# Calculate peakiness
peakiness = min(maxima_values[0],( maxima_values[1]/ lowest_point_value))

# Use the combination of thresholds with highest peakiness
if peakiness > 1:
    threshold = lowest_point_index
else:
    threshold = (maxima_indices[0] + maxima_indices[1]) // 2

print("The optimal threshold value is : ",threshold)
# Apply the threshold to the image to obtain the binary image
binary_image1 = (image > threshold).astype(np.uint8) * 255


# Display the original and segmented images
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.imshow(image, cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(binary_image1, cmap='gray')
plt.title('Image Using Peakiness Detection')
plt.axis('off')

plt.show()
