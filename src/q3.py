import cv2
import numpy as np
import matplotlib.pyplot as plt

# load the runway image
img = cv2.imread('images/runway.png', cv2.IMREAD_GRAYSCALE)
#safety check
if img is None:
    print("error: image not found")
else:
    # --- Step 1: Define the Custom Function ---
    def my_hist_equalization(image):
        # 1. Calculate the histogram (count frequency of pixels)
        # flatten() turns the 2D image into a long 1D list
        hist, bins = np.histogram(image.flatten(), 256, [0, 256])
        
        # 2. Calculate Cumulative Distribution Function (CDF)
        # cumsum() adds them up: [1, 2, 3] -> [1, 3, 6]
        cdf = hist.cumsum()
        
        # 3. Normalize the CDF
        # We want the values to be between 0 and 255
        # Formula: (cdf - min) * 255 / (max - min)
        # Mask zeros first so we don't calculate for empty bins
        cdf_m = np.ma.masked_equal(cdf, 0)
        cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min())
        
        # Fill masked places back with 0
        cdf_final = np.ma.filled(cdf_m, 0).astype('uint8')
        
        # 4. Map the original pixels to new values using the CDF as a lookup table
        # cdf_final[pixel_value] gives the new equalized value
        img_equalized = cdf_final[image]
        
        return img_equalized

    # --- Step 2: Apply the function ---
    res_eq = my_hist_equalization(img)

    # --- Step 3: Visualization ---
    plt.figure(figsize=(12, 8))

    # Original Image
    plt.subplot(2, 2, 1)
    plt.imshow(img, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')

    # Equalized Image
    plt.subplot(2, 2, 2)
    plt.imshow(res_eq, cmap='gray')
    plt.title('Histogram Equalized (Custom Function)')
    plt.axis('off')

    # Original Histogram
    plt.subplot(2, 2, 3)
    plt.hist(img.ravel(), 256, [0, 256], color='black')
    plt.title('Original Histogram')
    plt.xlabel('Intensity')
    plt.ylabel('Count')

    # Equalized Histogram 
    plt.subplot(2, 2, 4)
    plt.hist(res_eq.ravel(), 256, [0, 256], color='black')
    plt.title('Equalized Histogram')
    plt.xlabel('Intensity')

    plt.tight_layout()
    plt.show()