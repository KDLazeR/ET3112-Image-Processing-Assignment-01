import cv2
import numpy as np
import matplotlib.pyplot as plt

# load the runway image in grayscale
img = cv2.imread('images/runway.png', cv2.IMREAD_GRAYSCALE)

# safety check
if img is None:
    print("error: cant find the image file")
else:
    # --- part a: gamma correction with gamma = 0.5 ---
    gamma = 0.5
    
    # create the lookup table (LUT)
    # formula: s = r^gamma
    # we divide by 255 to normalize, do the power, then multiply back
    table = np.array([((i / 255.0) ** gamma) * 255 for i in range(256)]).astype("uint8")
    
    # apply the mapping
    res_a = cv2.LUT(img, table)

    # show results
    plt.figure(figsize=(10, 5))

    #og
    plt.subplot(1, 2, 1)
    plt.imshow(img, cmap='gray')
    plt.title('Original')
    plt.axis('off')

    #gamma 0.5 
    plt.subplot(1, 2, 2)
    plt.imshow(res_a, cmap='gray')
    plt.title('Gamma = 0.5')
    plt.axis('off')

    plt.tight_layout()
    plt.show()