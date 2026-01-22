import cv2
import numpy as np
import matplotlib.pyplot as plt

# load the runway image in grayscale
img = cv2.imread('images/runway.png', cv2.IMREAD_GRAYSCALE)
#safety Check
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

    # show results
    plt.figure(figsize=(12, 5))

    # og
    plt.subplot(1, 3, 1)
    plt.imshow(img, cmap='gray')
    plt.title('original')
    plt.axis('off')

    #  gamma 0.5
    plt.subplot(1, 3, 2)
    plt.imshow(res_a, cmap='gray')
    plt.title('gamma = 0.5')
    plt.axis('off')

    #  gamma 2.0
    plt.subplot(1, 3, 3)
    plt.imshow(res_b, cmap='gray')
    plt.title('gamma = 2.0')
    plt.axis('off')

    plt.tight_layout()
    plt.show()