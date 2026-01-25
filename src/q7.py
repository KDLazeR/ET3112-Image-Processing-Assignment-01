import cv2
import numpy as np
import matplotlib.pyplot as plt

def calculate_ssd(img1, img2):
   
    # Computes the Normalized Sum of Squared Difference (SSD) between two images.
    # Lower is better (0 means identical).
   
    if img1.shape != img2.shape:
        print(f"Warning: Shape mismatch! {img1.shape} vs {img2.shape}")
        # Resize img2 to match img1 strictly for SSD calculation if off by 1 pixel
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

    # Convert to float to avoid overflow during subtraction
    diff = img1.astype(float) - img2.astype(float)
    
    # Square the differences
    sq_diff = diff ** 2
    
    # Sum and Normalize (Divide by total number of pixels)
    ssd = np.sum(sq_diff) / (img1.shape[0] * img1.shape[1] * img1.shape[2])
    
    return ssd

def zoom_image(image, scale, method='bilinear'):
    
    # Zooms an image by a factor 's' using the specified method.
    # handle (a) nearest-neighbor, and (b) bilinear interpolation.
    
    # Calculate new dimensions
    new_width = int(image.shape[1] * scale)
    new_height = int(image.shape[0] * scale)
    dim = (new_width, new_height)
    
    if method == 'nearest':
        # (a) Nearest-Neighbor Interpolation
        return cv2.resize(image, dim, interpolation=cv2.INTER_NEAREST)
    
    elif method == 'bilinear':
        # (b) Bilinear Interpolation
        return cv2.resize(image, dim, interpolation=cv2.INTER_LINEAR)
    
    else:
        print("Error: Unknown method")
        return image

# --- Main Program ---

# 1. Load the images
# 'im01.png' is the Large Original (Ground Truth)
# 'im01small.png' is the Small version we need to zoom up
img_large = cv2.imread('images/im01.png')
img_small = cv2.imread('images/im01small.png')

if img_large is None or img_small is None:
    print("Error: Images not found. Check filenames.")
else:
    # 2. Calculate the Zoom Factor s
    # We want to scale the small one UP to match the large one.
    # factor = Large Width / Small Width
    s = img_large.shape[1] / img_small.shape[1]
    print(f"Calculated Zoom Factor s: {s:.4f}")

    # 3. Apply Zooming using our function
    # (a) Nearest Neighbor
    zoom_nn = zoom_image(img_small, s, method='nearest')
    
    # (b) Bilinear Interpolation
    zoom_bl = zoom_image(img_small, s, method='bilinear')

    # 4. Calculate SSD (Error)
    # We compare our "Zoomed Up" result against the "True Large" image
    ssd_nn = calculate_ssd(img_large, zoom_nn)
    ssd_bl = calculate_ssd(img_large, zoom_bl)

    print(f"SSD Error (Nearest Neighbor): {ssd_nn:.2f}")
    print(f"SSD Error (Bilinear):         {ssd_bl:.2f}")
    print("Lower SSD means better quality.")

    # 5. Visualization
    # zooming in on a specific detail to show the difference visually
    # Cropping a region for display 
    y1, y2, x1, x2 = 200, 400, 800, 1000 
    
    crop_orig = img_large[y1:y2, x1:x2]
    crop_nn = zoom_nn[y1:y2, x1:x2]
    crop_bl = zoom_bl[y1:y2, x1:x2]

    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.imshow(cv2.cvtColor(crop_orig, cv2.COLOR_BGR2RGB))
    plt.title('Original High-Res (Ground Truth)')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(cv2.cvtColor(crop_nn, cv2.COLOR_BGR2RGB))
    plt.title(f'Nearest Neighbor\nSSD: {ssd_nn:.2f} (Blocky)')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(cv2.cvtColor(crop_bl, cv2.COLOR_BGR2RGB))
    plt.title(f'Bilinear Interpolation\nSSD: {ssd_bl:.2f} (Smoother)')
    plt.axis('off')

    plt.tight_layout()
    plt.show()