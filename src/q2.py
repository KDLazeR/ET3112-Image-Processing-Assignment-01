import cv2
import numpy as np
import matplotlib.pyplot as plt

# load figure 2 
img_bgr = cv2.imread('images/highlights_and_shadows.jpg')
# safety Check
if img_bgr is None:
    print("error: could not find image")
else:
    # --- part a: convert to Lab and apply gamma to L ---
    
    # 1. convert BGR to Lab
    img_lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    
    # 2. split channels
    l, a, b = cv2.split(img_lab)
    
    # 3. apply gamma = 2.0 to L channel
    gamma = 2.0
    
    # normalize, power law, scale back
    l_float = l.astype("float32") / 255.0
    l_corrected = (l_float ** gamma) * 255.0
    l_corrected = l_corrected.astype("uint8")
    
    # 4. merge and convert back for display
    lab_corrected = cv2.merge((l_corrected, a, b))
    img_result = cv2.cvtColor(lab_corrected, cv2.COLOR_LAB2BGR)

    # --- part b: histograms ---
    
    plt.figure(figsize=(12, 8))
    
    # 1. original image
    plt.subplot(2, 2, 1)
    plt.imshow(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    plt.title('original')
    plt.axis('off')
    
    # 2. corrected image
    plt.subplot(2, 2, 2)
    plt.imshow(cv2.cvtColor(img_result, cv2.COLOR_BGR2RGB))
    plt.title(f'corrected (gamma={gamma})')
    plt.axis('off')
    
    # 3. histogram of original L
    plt.subplot(2, 2, 3)
    # ravel() flattens the 2D array to 1D
    plt.hist(l.ravel(), 256, [0, 256], color='gray')
    plt.title('histogram of L (original)')
    plt.xlabel('pixel intensity')
    plt.ylabel('count')
    
    # 4. histogram of corrected L
    plt.subplot(2, 2, 4)
    plt.hist(l_corrected.ravel(), 256, [0, 256], color='gray')
    plt.title(f'histogram of L (gamma={gamma})')
    plt.xlabel('pixel intensity')
    
    plt.tight_layout()
    plt.show()