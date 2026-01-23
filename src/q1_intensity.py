import cv2
import numpy as np
import matplotlib.pyplot as plt

# load the runway image in grayscale
img = cv2.imread('images/runway.png', cv2.IMREAD_GRAYSCALE)

# safety Check
if img is None:
    print("error: cant find the image file")
else:
    # --- part a: gamma = 0.5 (brighten) ---
    gamma1 = 0.5
    # calculate lookup table
    table1 = np.array([((i / 255.0) ** gamma1) * 255 for i in range(256)]).astype("uint8")
    # apply transform
    res_a = cv2.LUT(img, table1)

    # --- part b: gamma = 2.0 (darken) ---
    gamma2 = 2.0
    # same formula, different gamma value
    table2 = np.array([((i / 255.0) ** gamma2) * 255 for i in range(256)]).astype("uint8")
    res_b = cv2.LUT(img, table2)

    # --- part c: contrast stretching ---
    # creating a function for the piecewise linear transform
    def contrast_stretch(image):
        # convert to float 0-1 range first
        norm_img = image.astype("float32") / 255.0
        output = np.zeros_like(norm_img)
        
        # thresholds given in the assignment
        r1 = 0.2
        r2 = 0.8
        
        # 1. pixels between r1 and r2 -> stretch them
        mask_mid = (norm_img >= r1) & (norm_img <= r2)
        output[mask_mid] = (norm_img[mask_mid] - r1) / (r2 - r1)
        
        # 2. pixels above r2 -> make them white (1.0)
        mask_high = norm_img > r2
        output[mask_high] = 1.0
        
        # 3. pixels below r1 stay 0 (black), handled by zeros_like init
        
        # convert back to 0-255
        return (output * 255).astype("uint8")

    res_c = contrast_stretch(img)

    # show results
    plt.figure(figsize=(10, 8))

    # og
    plt.subplot(2, 2, 1)
    plt.imshow(img, cmap='gray')
    plt.title('original')
    plt.axis('off')

    # gamma 0.5
    plt.subplot(2, 2, 2)
    plt.imshow(res_a, cmap='gray')
    plt.title('gamma = 0.5')
    plt.axis('off')

    # gamma 2.0
    plt.subplot(2, 2, 3)
    plt.imshow(res_b, cmap='gray')
    plt.title('gamma = 2.0')
    plt.axis('off')
    
    # contrast stretching
    plt.subplot(2, 2, 4)
    plt.imshow(res_c, cmap='gray')
    plt.title('contrast stretching')
    plt.axis('off')

    plt.tight_layout()
    plt.show()