import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the Noisy Image (Figure 4)
img = cv2.imread('images/emma_salt_pepper.jpg', cv2.IMREAD_GRAYSCALE)

if img is None:
    print("Error: Image not found. Check filename.")
else:
    # --- Part (a): Gaussian Smoothing ---
    # Gaussian creates a weighted average. 
    # We use a 5x5 kernel.
    gaussian_blur = cv2.GaussianBlur(img, (5, 5), sigmaX=2)

    # --- Part (b): Median Filtering ---
    # Median sorting replaces each pixel with the median value of its neighbors.
    # We use a kernel size of 5 (must be an odd number).
    median_blur = cv2.medianBlur(img, 5)

    # --- Visualization ---
    plt.figure(figsize=(15, 6))

    # 1. Original Noisy Image
    plt.subplot(1, 3, 1)
    plt.imshow(img, cmap='gray')
    plt.title('Original (Salt & Pepper Noise)')
    plt.axis('off')

    # 2. Gaussian Result
    plt.subplot(1, 3, 2)
    plt.imshow(gaussian_blur, cmap='gray')
    plt.title('Gaussian Smoothing')
    plt.axis('off')

    # 3. Median Result
    plt.subplot(1, 3, 3)
    plt.imshow(median_blur, cmap='gray')
    plt.title('Median Filtering')
    plt.axis('off')

    plt.tight_layout()
    plt.show()