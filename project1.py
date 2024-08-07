import numpy as np
import matplotlib.pyplot as plt


def label_connected_components(image, min_size):
    rows, cols = image.shape
    labels = np.zeros((rows, cols), dtype=np.uint32)
    label_counter = 0

    for i in range(rows):
        for j in range(cols):
            if image[i, j] == 1:
                neighbors_labels = [labels[ni, nj] for ni in range(i-1, i+2) for nj in range(j-1, j+2)
                                    if 0 <= ni < rows and 0 <= nj < cols and labels[ni, nj] != 0]
                if len(neighbors_labels) == 0:
                    label_counter += 1
                    labels[i, j] = label_counter
                else:
                    labels[i, j] = min(neighbors_labels)
                    for neighbor_label in neighbors_labels:
                        if neighbor_label != labels[i, j]:
                            min_label = min(labels[i, j], neighbor_label)
                            max_label = max(labels[i, j], neighbor_label)
                            # Merge equivalent labels
                            for r in range(rows):
                                for c in range(cols):
                                    if labels[r, c] == max_label:
                                        labels[r, c] = min_label

    # Second pass: relabel the components to have consecutive labels
    relabel = {}
    next_label = 1
    for i in range(rows):
        for j in range(cols):
            if labels[i, j] != 0:
                if labels[i, j] not in relabel:
                    relabel[labels[i, j]] = next_label
                    next_label += 1
                labels[i, j] = relabel[labels[i, j]]

    # Filter out small components
    component_sizes = {label: np.sum(labels == label) for label in np.unique(labels)}
    for label, size in component_sizes.items():
        if size < min_size:
            labels[labels == label] = 0

    component_areas = {label: size for label, size in component_sizes.items() if size >= min_size}

    # Calculate centroids
    component_centroids = {}
    for label, area in component_areas.items():
        centroid_x, centroid_y = 0, 0
        for i in range(rows):
            for j in range(cols):
                if labels[i, j] == label:
                    centroid_x += j
                    centroid_y += i
        centroid_x /= area
        centroid_y /= area
        component_centroids[label] = (centroid_x, centroid_y)

    # Calculate bounding boxes
    component_bounding_boxes = {}
    for label, area in component_areas.items():
        min_x, max_x, min_y, max_y = cols, 0, rows, 0
        for i in range(rows):
            for j in range(cols):
                if labels[i, j] == label:
                    min_x = min(min_x, j)
                    max_x = max(max_x, j)
                    min_y = min(min_y, i)
                    max_y = max(max_y, i)
        component_bounding_boxes[label] = ((min_x, min_y), (max_x, max_y))

    # Calculate perimeters
    component_perimeters = {}
    for label, area in component_areas.items():
        perimeter = 0
        for i in range(rows):
            for j in range(cols):
                if labels[i, j] == label:
                    for ni, nj in [(i-1, j), (i+1, j), (i, j-1), (i, j+1)]:
                        if 0 <= ni < rows and 0 <= nj < cols and labels[ni, nj] != label:
                            perimeter += 1
        component_perimeters[label] = perimeter

    # Calculate compactness
    component_compactness = {}
    for label, area in component_areas.items():
        compactness = (component_perimeters[label] ** 2) / area
        component_compactness[label] = compactness

    # Calculate eccentricity
    component_eccentricity = {}
    for label in component_areas.keys():
        min_x, min_y = component_bounding_boxes[label][0]
        max_x, max_y = component_bounding_boxes[label][1]
        a = (max_x - min_x) / 2
        b = (max_y - min_y) / 2
        eccentricity = np.sqrt(1 - (b ** 2) / (a ** 2))
        component_eccentricity[label] = eccentricity

# Calculate orientation of elongation using PCA
    component_orientations = {}
    for label in component_areas.keys():
        # Find the coordinates of the pixels in the component
        coords = np.column_stack(np.where(labels == label))
        
        # Compute the covariance matrix
        cov_matrix = np.cov(coords, rowvar=False)
        
        # Perform PCA
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        
        # Get the orientation of elongation (angle of the first eigenvector)
        orientation = np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0])
        component_orientations[label] = orientation

    # # Calculate orientation of elongation
    # component_orientations = {}
    # for label in component_areas.keys():
    #     # Calculate second moments
    #     m00, m01, m10, m11, m20, m02 = 0, 0, 0, 0, 0, 0
    #     for i in range(rows):
    #         for j in range(cols):
    #             if labels[i, j] == label:
    #                 m00 += 1
    #                 m01 += i
    #                 m10 += j
    #                 m11 += i * j
    #                 m20 += j * j
    #                 m02 += i * i

    #     # Calculate orientation
    #     x_bar = m10 / m00
    #     y_bar = m01 / m00
    #     u11 = (m11 - x_bar * m01) / m00
    #     u20 = (m20 - x_bar * m10) / m00
    #     u02 = (m02 - y_bar * m01) / m00
    #     theta = 0.5 * np.arctan(2 * u11 / (u20 - u02))

    #     component_orientations[label] = theta

    # Print detailed information
    for label, area in component_areas.items():
        centroid_x, centroid_y = component_centroids[label]
        min_x, min_y = component_bounding_boxes[label][0]
        max_x, max_y = component_bounding_boxes[label][1]
        print(f"Component {label}:")
        print(f"  Area = {area}")
        print(f"  Centroid = ({centroid_x:.2f}, {centroid_y:.2f})")
        print(f"  Bounding Box: (({min_x},{min_y}), ({max_x},{max_y})")
        print(f"  Perimeter = {component_perimeters[label]}")
        print(f"  Compactness = {component_compactness[label]}")
        print(f"  Eccentricity = {component_eccentricity[label]}")
        print(f"  Orientation of Elongation = {component_orientations[label]:.2f} radians")

    return labels


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

# Apply connected component labeling
labels = label_connected_components(binary_image, 1000)
num_labels = np.unique(labels)
print("\n Total number of components: ",len(num_labels)-1)
print(" List of unique labels: ",num_labels)


# Display the labeled image
plt.imshow(labels, cmap='rainbow')
plt.title('Components based on the specified min size ')
plt.colorbar()
plt.show()

plt.imshow(image_array, cmap='gray')
plt.title('Original Image')
plt.colorbar()
plt.show()

plt.imshow(binary_image, cmap='gray')
plt.title('Threshold Image')
plt.colorbar()
plt.show()
