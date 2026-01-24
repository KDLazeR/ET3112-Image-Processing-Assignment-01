import cv2
import numpy as np
import matplotlib.pyplot as plt

def get_gaussian_kernel(k_size, sigma):
    #Generates the Gaussian kernel (Same as Part A) 
    center = k_size // 2
    x, y = np.mgrid[-center:center+1, -center:center+1]
    normalizer = 1 / (2.0 * np.pi * sigma**2)
    kernel = normalizer * np.exp(-(x**2 + y**2) / (2.0 * sigma**2))
    kernel = kernel / kernel.sum()
    return kernel

# 1. Load the image in Grayscale
img = cv2.imread('images/highlights_and_shadows.jpg', cv2.IMREAD_GRAYSCALE)

if img is None:
    print("Error: Image not found. Check path.")
else:
    # 2. Compute the 5x5 Kernel manually (Sigma = 2)
    sigma = 2
    kernel_size = 5
    my_kernel = get_gaussian_kernel(kernel_size, sigma)

    # 3. Apply the custom kernel using filter2D
    # -1 means "keep the same depth as the original image"
    img_smoothed = cv2.filter2D(img, -1, my_kernel)

    # --- Visualization ---
    plt.figure(figsize=(10, 5))

    # Original
    plt.subplot(1, 2, 1)
    plt.imshow(img, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')

    # Smoothed (Manual Kernel)
    plt.subplot(1, 2, 2)
    plt.imshow(img_smoothed, cmap='gray')
    plt.title(f'Smoothed (Manual Kernel, $\sigma$={sigma})')
    plt.axis('off')

    plt.tight_layout()
    plt.show()