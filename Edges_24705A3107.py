# Import the OpenCV library for image processing tasks like reading, transforming, and filtering images
import cv2
# Import NumPy, which is used for handling arrays, matrices, and numerical operations in image processing
import numpy as np

# Import pyplot from matplotlib, which is used to display images and visualizations in Python
from matplotlib import pyplot as plt

# ---------------------- Load the Image ----------------------

# Load the image file named 'loki.jpg' into a variable called 'image' using OpenCV's imread function
image = cv2.imread('loki.jpg')

# Check if the image is loaded successfully. If not, print an error and exit the program.
if image is None:
    print("Image not found.")  # If image is not found or path is incorrect
    exit()  # Exit the program

# ---------------------- Convert to Grayscale ----------------------

# Convert the image from BGR (default color format in OpenCV) to grayscale using cvtColor
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# ---------------------- Apply Sobel Edge Detection ----------------------

# Apply Sobel filter to detect edges in the X-direction (horizontal changes)
sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)

# Apply Sobel filter to detect edges in the Y-direction (vertical changes)
sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

# Combine the Sobel X and Y results to get a complete edge map using magnitude
sobel_combined = cv2.magnitude(sobelx, sobely)

# ---------------------- Apply Canny Edge Detection ----------------------

# Apply Canny edge detection algorithm with threshold values 100 and 200
canny = cv2.Canny(gray, 100, 200)

# ---------------------- Apply Thresholding on Grayscale ----------------------

# Apply binary thresholding on the grayscale image.
# All pixel values above 120 become 255 (white), others become 0 (black)
_, thresh_gray = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY)

# ---------------------- Apply Thresholding on Color Channels Separately ----------------------

# Split the color image into its three channels: Blue, Green, and Red
b, g, r = cv2.split(image)

# Apply binary thresholding on each channel separately
_, thresh_b = cv2.threshold(b, 120, 255, cv2.THRESH_BINARY)  # Threshold Blue
_, thresh_g = cv2.threshold(g, 120, 255, cv2.THRESH_BINARY)  # Threshold Green
_, thresh_r = cv2.threshold(r, 120, 255, cv2.THRESH_BINARY)  # Threshold Red

# Merge the thresholded channels back into a single color image
merged_thresh = cv2.merge((thresh_b, thresh_g, thresh_r))

# ---------------------- Display Results Using Matplotlib ----------------------

# Titles for each subplot image
titles = ['Original', 'Gray', 'Sobel X+Y', 'Canny', 'Threshold Gray', 'Threshold Color']

# Store all the images to be displayed in a list
images = [image, gray, sobel_combined, canny, thresh_gray, merged_thresh]

# Create a figure with a specific size (width=12 inches, height=8 inches)
plt.figure(figsize=(12, 8))

# Loop through each image and display it in a subplot
for i in range(6):
    plt.subplot(2, 3, i+1)  # Arrange images in a 2x3 grid
    if len(images[i].shape) == 2:
        # If the image is grayscale (2D), use gray colormap
        plt.imshow(images[i], cmap='gray')
    else:
        # If the image is color (3D), convert from BGR to RGB before displaying
        plt.imshow(cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB))
    plt.title(titles[i])  # Set the title for each subplot
    plt.axis('off')       # Hide axis ticks for clean view

# Adjust layout to prevent overlap of titles and plots
plt.tight_layout()

# Display all the plotted images in a single window
plt.show()
