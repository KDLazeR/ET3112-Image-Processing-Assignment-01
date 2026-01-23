import cv2
import numpy as np
import matplotlib.pyplot as plt

# load figure 2 
img_bgr = cv2.imread('images/highlights_and_shadows.jpg')

# check if image loaded correctly
if img_bgr is None:
    print("error: could not find image")
else:
    # --- part a: convert to Lab and apply gamma to L ---
    
    # step 1: convert BGR to Lab
    # in Lab: L = Lightness, a = Green-Red, b = Blue-Yellow
    img_lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    
    # step 2: split into channels
    l, a, b = cv2.split(img_lab)
    
    # step 3: apply gamma to L channel
    # using gamma = 2.0 to darken the bright rocks/dress so we can see details
    gamma = 2.0
    
    # normalize to 0-1, apply power, scale back
    l_float = l.astype("float32") / 255.0
    l_corrected = (l_float ** gamma) * 255.0
    l_corrected = l_corrected.astype("uint8")
    
    # step 4: merge channels back
    lab_corrected = cv2.merge((l_corrected, a, b))
    
    # step 5: convert back to BGR for display
    img_result = cv2.cvtColor(lab_corrected, cv2.COLOR_LAB2BGR)

    # --- visualization ---
    plt.figure(figsize=(10, 5))
    
    # original (convert bgr to rgb for matplotlib)
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    plt.title('original')
    plt.axis('off')
    
    # corrected
    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(img_result, cv2.COLOR_BGR2RGB))
    plt.title(f'corrected (gamma={gamma})')
    plt.axis('off')
    
    plt.show()