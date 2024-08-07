import numpy as np
import matplotlib.pyplot as plt

def distance_transform(img):
    # Initialize the result img with a large value
    result = np.full_like(img, 1000000)
    
    # Initialize the queue with boundary cells
    queue = []
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if img[i, j] == 0:
                result[i, j] = 0
                queue.append((i, j))
    
    # Perform breadth-first search
    while queue:
        i, j = queue.pop(0)
        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            x, y = i + dx, j + dy
            if 0 <= x < img.shape[0] and 0 <= y < img.shape[1] and result[x, y] > result[i, j] + 1:
                result[x, y] = result[i, j] + 1
                queue.append((x, y))

    # Find local maxima
    Skeleton = np.zeros_like(img, dtype=bool)
    for i in range(1, img.shape[0] - 1):
        for j in range(1, img.shape[1] - 1):
            if result[i, j] > 0:
                neighbors = result[i-1:i+2, j-1:j+2]
                if result[i, j] == np.max(neighbors):
                    Skeleton[i, j] = True
    
    
    return result, Skeleton

def reconstruct_image_from_skeleton(skeleton):
    # Initialize the reconstructed image with zeros
    reconimg = np.zeros_like(skeleton, dtype=bool)

    # Define the neighborhood size 
    neighborhood_size = 13

    # Iterate over the skeleton pixels
    for i in range(skeleton.shape[0]):
        for j in range(skeleton.shape[1]):
            if skeleton[i, j]:
                # Start from the current pixel and move along the skeleton path
                x, y = i, j
                while skeleton[x, y]:
                    # Set the neighborhood pixels to True
                    for dx in range(-neighborhood_size, neighborhood_size + 1):
                        for dy in range(-neighborhood_size, neighborhood_size + 1):
                            if 0 <= x + dx < skeleton.shape[0] and 0 <= y + dy < skeleton.shape[1]:
                                reconimg[x + dx, y + dy] = True
                    # Move to the next pixel in the skeleton path
                    neighbors = skeleton[max(0, x-1):min(skeleton.shape[0], x+2), max(0, y-1):min(skeleton.shape[1], y+2)]
                    x, y = np.unravel_index(np.argmax(neighbors), neighbors.shape)
                    x, y = x + max(0, x-1), y
    return reconimg

# Load image
input_file_path = 'comb.img'
header_size = 512

with open(input_file_path, 'rb') as input_file:
    # Seek to the position after the header
    input_file.seek(header_size)

    # Read the remaining content
    image_content = input_file.read()

# Convert the binary data to a NumPy array of unsigned characters (uint8)
image_array = np.frombuffer(image_content, dtype=np.uint8)

# Assuming the image is grayscale, you can reshape it into a 2D array
# Adjust the shape based on the dimensions of your image (e.g., height, width)
image_height = 512  # Replace with the actual height of your image
image_width = 512  # Replace with the actual width of your image
image_array = image_array.reshape((image_height, image_width))

# Threshold the image
threshold_value = 128
binary_image = np.where(image_array >= threshold_value, 0, 1)

# Perform distance transform
distance_transformed_image, Skeleton = distance_transform(binary_image)

# Perform Reconstruction
recon = reconstruct_image_from_skeleton(Skeleton)

# Plot the Original image
plt.imshow(binary_image, cmap='gray')
plt.title('Original Binary Image')
plt.show()

# Plot the Skeletonized image
plt.imshow(Skeleton, cmap='gray')
plt.title('Skeletonized Image')
plt.show()

# Plot the reconstructed image
plt.imshow(recon, cmap='gray')
plt.title('Deskeletonized Image')
plt.show()




