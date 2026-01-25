import cv2
import numpy as np
import matplotlib.pyplot as plt


img = cv2.imread('images/highlights_and_shadows.jpg')

if img is None:
    print("Error: Image not found.")
else:
    # Convert to RGB for matplotlib display
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Define a Sharpening Kernel
    # This kernel highlights edges by subtracting neighbors from the center.
    # The sum of elements is 1 (5 - 4 = 1), so brightness stays the same.
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])

    # 3. Apply the Kernel using filter2D
    img_sharpened = cv2.filter2D(img_rgb, -1, kernel)

    # --- Visualization ---
    plt.figure(figsize=(15, 8))

    # Original
    plt.subplot(1, 2, 1)
    plt.imshow(img_rgb)
    plt.title('Original Image')
    plt.axis('off')

    # Sharpened
    plt.subplot(1, 2, 2)
    plt.imshow(img_sharpened)
    plt.title('Sharpened Image')
    plt.axis('off')

    plt.tight_layout()
    plt.show()