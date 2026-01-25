import cv2
import numpy as np
import matplotlib.pyplot as plt

def get_gaussian_kernel(k_size, sigma):
    # Generates the Gaussian kernel (Same as Part A) 
    center = k_size // 2
    x, y = np.mgrid[-center:center+1, -center:center+1]
    normalizer = 1 / (2.0 * np.pi * sigma**2)
    kernel = normalizer * np.exp(-(x**2 + y**2) / (2.0 * sigma**2))
    kernel = kernel / kernel.sum()
    return kernel

# 1. Load Image
img = cv2.imread('images/highlights_and_shadows.jpg', cv2.IMREAD_GRAYSCALE)

if img is None:
    print("Error: Image not found.")
else:
    sigma = 2
    k_size = 5

    # --- Method 1: Manual Implementation (Part C) ---
    manual_kernel = get_gaussian_kernel(k_size, sigma)
    img_manual = cv2.filter2D(img, -1, manual_kernel)

    # --- Method 2: OpenCV Built-in (Part D) ---
    # cv2.GaussianBlur(src, ksize, sigmaX)
    img_opencv = cv2.GaussianBlur(img, (k_size, k_size), sigmaX=sigma)

    # --- Visualization ---
    plt.figure(figsize=(10, 5))

    # 1. Manual Result
    plt.subplot(1, 2, 1)
    plt.imshow(img_manual, cmap='gray')
    plt.title('Manual Implementation')
    plt.axis('off')

    # 2. OpenCV Result
    plt.subplot(1, 2, 2)
    plt.imshow(img_opencv, cmap='gray')
    plt.title('OpenCV Built-in')
    plt.axis('off')

    plt.tight_layout()
    plt.show()