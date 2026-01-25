import cv2
import numpy as np
import matplotlib.pyplot as plt

def get_derivative_kernels(k_size, sigma):
    # 1. Generate Grid
    center = k_size // 2
    x, y = np.mgrid[-center:center+1, -center:center+1]
    
    # 2. Base Gaussian
    normalizer = 1 / (2.0 * np.pi * sigma**2)
    G = normalizer * np.exp(-(x**2 + y**2) / (2.0 * sigma**2))
    G = G / G.sum()
    
    # 3. Derivatives
    Gx = -(x / sigma**2) * G
    Gy = -(y / sigma**2) * G
    
    return Gx, Gy

# Load Image
img = cv2.imread('images/highlights_and_shadows.jpg', cv2.IMREAD_GRAYSCALE)

if img is None:
    print("Error: Image not found.")
else:
    # 1. Get Kernels
    sigma = 2
    Gx_kernel, Gy_kernel = get_derivative_kernels(5, sigma)
    
    # 2. Apply Kernels (Convolve)
    # MUST use cv2.CV_64F to keep negative values (transitions from bright -> dark)
    grad_x_float = cv2.filter2D(img, cv2.CV_64F, Gx_kernel)
    grad_y_float = cv2.filter2D(img, cv2.CV_64F, Gy_kernel)
    
    # 3. Convert to Absolute values for Visualization
    # (We want to see the edge strength, regardless of direction)
    grad_x_abs = cv2.convertScaleAbs(grad_x_float)
    grad_y_abs = cv2.convertScaleAbs(grad_y_float)
    
    # Optional: Combine them to see all edges (Magnitude)
    magnitude = cv2.addWeighted(grad_x_abs, 0.5, grad_y_abs, 0.5, 0)

    # --- Visualization ---
    plt.figure(figsize=(12, 5))
    
    # Vertical Edges (detected by Gx)
    plt.subplot(1, 3, 1)
    plt.imshow(grad_x_abs, cmap='gray')
    plt.title('Horizontal Gradient ($G_x$)\nDetects Vertical Edges')
    plt.axis('off')
    
    # Horizontal Edges (detected by Gy)
    plt.subplot(1, 3, 2)
    plt.imshow(grad_y_abs, cmap='gray')
    plt.title('Vertical Gradient ($G_y$)\nDetects Horizontal Edges')
    plt.axis('off')

    # Combined
    plt.subplot(1, 3, 3)
    plt.imshow(magnitude, cmap='gray')
    plt.title('Gradient Magnitude\n(All Edges)')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()