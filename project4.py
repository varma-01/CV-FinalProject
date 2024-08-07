import numpy as np
import matplotlib.pyplot as plt

# A function to create a Gaussian kernel
def gaussian_kernel(size, sigma):
    # Calculate the center of the kernel
    center = size // 2
    # Create the kernel using a lambda function
    kernel = np.fromfunction(lambda x, y: (1/(2*np.pi*sigma**2)) * np.exp(-((x-center)**2 + (y-center)**2)/(2*sigma**2)), (size, size))
    # Normalize the kernel
    return kernel / np.sum(kernel)

# A function to perform 2D convolution
def conv(image, kernel):
    # Get the size of the kernel
    kernel_size = kernel.shape[0]
    # Calculate the padding size
    pad_size = kernel_size // 2
    # Pad the image
    padded_image = np.pad(image, pad_size, mode='reflect')
    # Initialize the result array
    result = np.zeros_like(image, dtype=np.float32)
    # Perform convolution
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            result[i, j] = np.sum(padded_image[i:i+kernel_size, j:j+kernel_size] * kernel)
    return result

# A function to apply the Laplacian operator
def laplacian(image):
    # Define the Laplacian kernel
    laplacian_kernel = np.array([[0, 1, 0],
                                  [1, -4, 1],
                                  [0, 1, 0]])
    # Apply convolution
    laplacian = conv(image, laplacian_kernel)
    return laplacian

# A function to find zero crossings in an image
def zero_crossings(image):
    # Get the dimensions of the image
    rows, cols = image.shape
    # Initialize an array to store the zero crossings
    zero_crossing = np.zeros_like(image)
    # Iterate over the image pixels
    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            # Get the values of the neighboring pixels
            neighbors = [image[i - 1, j], image[i + 1, j], image[i, j - 1], image[i, j + 1],
                         image[i - 1, j - 1], image[i - 1, j + 1], image[i + 1, j - 1], image[i + 1, j + 1]]
            # Check if there is a sign change in the neighbors
            if any(np.sign(image[i, j]) != np.sign(neighbor) for neighbor in neighbors):
                zero_crossing[i, j] = 255
    return zero_crossing

# Open the image file in binary mode
with open('/Users/reenabardeskar/Downloads/CVProjects/Project3/test1.img', 'rb') as f:
    # Read the image data
    image_data = f.read()

# Convert the image data to a NumPy array
image = np.frombuffer(image_data, dtype=np.uint8)

# Reshape the array to represent the image dimensions
width = 512
height = 512
image = image[:width * height]  
image = image.reshape((height, width))

# Multi-scale LoG Edge Detection for specific sigma values with delta sigma = -0.5
print(f"Processing image")
for sigma in np.arange(5, 0, -0.5):  # Sigma values from 5 to 1 with delta sigma = -0.5
    # Determine kernel size based on sigma
    kernel_size = int(8*sigma)+1 if int(8*sigma)%2 == 0 else int(8*sigma)
    
    # Apply Gaussian Blur
    gaussian_kernel1 = gaussian_kernel(kernel_size, sigma)
    smoothed = conv(image, gaussian_kernel1)

    # Compute Laplacian 
    laplacian_image = laplacian(smoothed)

    # Find zero-crossings
    ed = zero_crossings(laplacian_image)

    if sigma in {1.0, 2.0, 3.0, 4.0, 5.0}:
        # Display and save edge map
        
        plt.figure(figsize=(8, 8))
        plt.imshow(image, cmap='gray')
        plt.title("Original Image")
        plt.axis('off')
        plt.show()
    break
# Close all figures
plt.close('all')
