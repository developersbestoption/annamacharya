import cv2  # OpenCV for image processing
import numpy as np  # NumPy for numerical operations
from matplotlib import pyplot as plt  # For displaying images

# ---------------------- Load the Image ----------------------

image = cv2.imread('download.jpeg')  # Load image from the file system

if image is None:  # Check if image was loaded successfully
    print("Image not found.")  # If not, display error message
    exit()  # Exit the program

# ---------------------- Apply Gaussian Filter ----------------------

# Gaussian Blur: smooths the image by averaging pixels with a Gaussian kernel
# Syntax: cv2.GaussianBlur(src, ksize, sigmaX)
gaussian = cv2.GaussianBlur(image, (5, 5), 0)  # 5x5 kernel, sigma=0 (auto)

# ---------------------- Apply Median Filter ----------------------

# Median Blur: replaces each pixel with the median of neighboring pixels
# Good for removing salt-and-pepper noise
# Syntax: cv2.medianBlur(src, ksize)
median = cv2.medianBlur(image, 5)  # Kernel size must be odd and >1

# ---------------------- Apply Bilateral Filter ----------------------

# Bilateral Filter: smooths image but preserves edges
# Syntax: cv2.bilateralFilter(src, d, sigmaColor, sigmaSpace)
# d = Diameter of each pixel neighborhood
# sigmaColor = Filter sigma in color space
# sigmaSpace = Filter sigma in coordinate space
bilateral = cv2.bilateralFilter(image, 9, 75, 75)  # Edge-preserving smoothing
# ---------------------- Display Results ----------------------
# List of titles to display for each image
titles = ['Original', 'Gaussian Filter', 'Median Filter', 'Bilateral Filter']

# List of corresponding image results
images = [image, gaussian, median, bilateral]

plt.figure(figsize=(10, 8))  # Set figure size for better visibility

# Loop through and plot each image in a 2x2 grid
for i in range(4):
    plt.subplot(2, 2, i+1)  # Define subplot position (2 rows, 2 columns)
    plt.imshow(cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB))  # Convert BGR to RGB for correct display
    plt.title(titles[i])  # Set subplot title
    plt.axis('off')  # Turn off axis ticks and labels
plt.tight_layout()  # Adjust layout to prevent overlapping
plt.show()  
