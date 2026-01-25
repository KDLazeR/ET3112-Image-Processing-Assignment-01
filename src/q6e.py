import cv2
import numpy as np
import matplotlib.pyplot as plt

def get_derivative_kernels(k_size, sigma):
    # Re-using the function from Part D
    center = k_size // 2
    x, y = np.mgrid[-center:center+1, -center:center+1]
    normalizer = 1 / (2.0 * np.pi * sigma**2)
    G = normalizer * np.exp(-(x**2 + y**2) / (2.0 * sigma**2))
    G = G / G.sum()
    Gx = -(x / sigma**2) * G
    Gy = -(y / sigma**2) * G
    return Gx, Gy

img = cv2.imread('images/highlights_and_shadows.jpg', cv2.IMREAD_GRAYSCALE)

if img is None:
    print("Error: Image not found.")
else:
    # --- Method 1: Manual Derivative of Gaussian (DoG) ---
    # Sigma=2 provides strong smoothing before edge detection
    sigma = 2
    Gx_kernel, Gy_kernel = get_derivative_kernels(5, sigma)
    
    man_x = cv2.filter2D(img, cv2.CV_64F, Gx_kernel)
    man_y = cv2.filter2D(img, cv2.CV_64F, Gy_kernel)
    
    # Calculate Magnitude (combine X and Y)
    man_mag = cv2.magnitude(man_x, man_y)
    man_final = cv2.convertScaleAbs(man_mag)

    # --- Method 2: OpenCV Sobel ---
    # standard Sobel with 5x5 kernel size
    sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5)
    sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=5)
    
    sobel_mag = cv2.magnitude(sobel_x, sobel_y)
    sobel_final = cv2.convertScaleAbs(sobel_mag)

    # --- Visualization ---
    plt.figure(figsize=(10, 6))

    # Manual Result
    plt.subplot(1, 2, 1)
    plt.imshow(man_final, cmap='gray')
    plt.title(f'Manual DoG\n($\sigma$={sigma}, Smooth Edges)')
    plt.axis('off')

    # Sobel Result
    plt.subplot(1, 2, 2)
    plt.imshow(sobel_final, cmap='gray')
    plt.title('OpenCV Sobel\n(ksize=5, Sharper/Noisier)')
    plt.axis('off')

    plt.tight_layout()
    plt.show()