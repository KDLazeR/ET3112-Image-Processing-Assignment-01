import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load Figure 3 (Woman in front of window)
img = cv2.imread('images/looking_out.jpg', cv2.IMREAD_GRAYSCALE)
#safety check
if img is None:
    print("Error: Could not find image 'images/looking_out.jpg'")
else:
    # --- Part A: Otsu's Thresholding ---
    
    # We want the "Woman and Room" to be the foreground (White).
    # Since they are dark pixels, we use THRESH_BINARY_INV.
    # Otsu's algorithm automatically calculates the optimal threshold value (ret).
    thresh_val, binary_mask = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    print(f"Otsu's optimal threshold value is: {thresh_val}")

    # --- Visualization ---
    plt.figure(figsize=(10, 5))
    
    # Original Grayscale
    plt.subplot(1, 2, 1)
    plt.imshow(img, cmap='gray')
    plt.title('Original Grayscale')
    plt.axis('off')
    
    # Binary Mask
    plt.subplot(1, 2, 2)
    plt.imshow(binary_mask, cmap='gray')
    plt.title(f'Otsu Binary Mask (Threshold={thresh_val})')
    plt.axis('off')
    
    plt.show()