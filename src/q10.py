import cv2
import numpy as np
import matplotlib.pyplot as plt

def gaussian_weight(x, sigma):
    # Helper: Computes 1D Gaussian weight 
    return (1.0 / (2 * np.pi * (sigma ** 2))) * np.exp(-(x ** 2) / (2 * (sigma ** 2)))

def manual_bilateral_filter(source, diameter, sigma_s, sigma_r):
    
    # (a) Manual Implementation of Bilateral Filter
    # img: Grayscale Image
    # diameter: Window size
    # sigma_s: Spatial (Distance) Standard Deviation
    # sigma_r: Range (Intensity) Standard Deviation
    
    img = source.astype(np.float32)
    h, w = img.shape
    output = np.zeros((h, w), dtype=np.float32)
    
    radius = diameter // 2
    
    # Pre-compute spatial weights (they don't change)
    # This speeds up the code significantly
    x, y = np.mgrid[-radius:radius+1, -radius:radius+1]
    spatial_kernel = np.exp(-(x**2 + y**2) / (2 * sigma_s**2))

    print(f"Processing Manual Bilateral Filter... (Size: {w}x{h})")
    
    # Iterate over every pixel
    for i in range(radius, h - radius):
        for j in range(radius, w - radius):
            
            # 1. Extract the local neighborhood (Region of Interest)
            # intensity_window contains the pixel values nearby
            intensity_window = img[i-radius:i+radius+1, j-radius:j+radius+1]
            
            # 2. Compute Range Weights (based on intensity difference)
            # Calculate difference between neighbor pixels and center pixel
            center_pixel_val = img[i, j]
            diff = intensity_window - center_pixel_val
            range_kernel = np.exp(-(diff**2) / (2 * sigma_r**2))
            
            # 3. Combine Weights
            weights = spatial_kernel * range_kernel
            
            # 4. Normalize
            norm_val = np.sum(weights)
            pixel_val = np.sum(weights * intensity_window) / norm_val
            
            output[i, j] = pixel_val
            
    return output.astype(np.uint8)

# --- Main Program ---

# Load Image
img_full = cv2.imread('images/highlights_and_shadows.jpg', cv2.IMREAD_GRAYSCALE)

if img_full is None:
    print("Error: Image not found.")
else:
    # Resize for speed (Manual implementation is slow in Python)
    # We reduce it to 25% size just to demonstrate it works quickly
    img = cv2.resize(img_full, (0,0), fx=0.5, fy=0.5)

    # Parameters
    d = 9          # Diameter
    sigma_s = 100  # Spatial Sigma (Large blur)
    sigma_r = 100  # Range Sigma (How strictly to keep edges)

    # (b) Apply Gaussian Blur 
    img_gaussian = cv2.GaussianBlur(img, (d, d), sigma_s)

    # (c) Apply OpenCV Bilateral Filter
    img_cv_bilateral = cv2.bilateralFilter(img, d, sigma_s, sigma_r)

    # (d) Apply Manual Bilateral Filter
    # This uses my function from Part (a)
    img_manual_bilateral = manual_bilateral_filter(img, d, sigma_s, sigma_r)

    # --- Visualization ---
    plt.figure(figsize=(12, 10))

    # 1. Original
    plt.subplot(2, 2, 1)
    plt.imshow(img, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')

    # 2. Gaussian (Blurry)
    plt.subplot(2, 2, 2)
    plt.imshow(img_gaussian, cmap='gray')
    plt.title('b)OpenCV Gaussian Blur')
    plt.axis('off')

    # 3. OpenCV Bilateral
    plt.subplot(2, 2, 3)
    plt.imshow(img_cv_bilateral, cmap='gray')
    plt.title('c)OpenCV Bilateral Filter')
    plt.axis('off')

    # 4. Manual Bilateral
    plt.subplot(2, 2, 4)
    plt.imshow(img_manual_bilateral, cmap='gray')
    plt.title('d)Manual Bilateral Implementation')
    plt.axis('off')

   # plt.tight_layout()
    plt.show()