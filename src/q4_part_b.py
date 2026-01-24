import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load Figure 3 (Woman in front of window)
img = cv2.imread('images/looking_out.jpg', cv2.IMREAD_GRAYSCALE)

if img is None:
    print("Error: check image path")
else:
    # --- Step 1: Get the Mask (from Part A) ---
    # Otsu thresholding with Inverse (Dark = Foreground)
    thresh_val, binary_mask = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # --- Step 2: Equalize ONLY the Foreground ---
    
    # Create a copy of the image to modify
    result_img = img.copy()
    
    # 1. Extract the pixels where the mask is White (255)
    # This gives us a 1D array of just the room/woman pixels
    foreground_pixels = img[binary_mask == 255]
    
    # 2. Equalize these pixels
    # cv2.equalizeHist requires a 2D array, so we reshape our 1D pixel list into a single-column "image"
    if len(foreground_pixels) > 0:
        fg_reshaped = foreground_pixels.reshape(-1, 1)
        fg_eq = cv2.equalizeHist(fg_reshaped)
        
        # Flatten back to 1D array
        fg_eq_flat = fg_eq.flatten()
        
        # 3. Put them back into the result image
        result_img[binary_mask == 255] = fg_eq_flat

    # --- Step 3: Visualization ---
    plt.figure(figsize=(12, 6))
    
    # Original
    plt.subplot(1, 2, 1)
    plt.imshow(img, cmap='gray')
    plt.title('Original (Dark Foreground)')
    plt.axis('off')
    
    # Equalized Foreground
    plt.subplot(1, 2, 2)
    plt.imshow(result_img, cmap='gray')
    plt.title('Equalized Foreground Only')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()