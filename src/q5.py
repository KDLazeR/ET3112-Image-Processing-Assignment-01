import numpy as np

def get_gaussian_kernel(k_size, sigma):
  
    #Generates a 2D Gaussian kernel using the formula:
    #G(x, y) = (1 / (2 * pi * sigma^2)) * exp( -(x^2 + y^2) / (2 * sigma^2) )
 
    # 1. Create a grid of (x, y) coordinates
    # For a 5x5 kernel, the center is at index 2. 
    # Coordinates range from -2 to +2.
    center = k_size // 2
    x, y = np.mgrid[-center:center+1, -center:center+1]
    
    # 2. Apply the Gaussian formula
    normalizer = 1 / (2.0 * np.pi * sigma**2)
    kernel = normalizer * np.exp(-(x**2 + y**2) / (2.0 * sigma**2))
    
    # 3. Normalize so the sum of all elements is 1
    # (This ensures the image doesn't get brighter or darker overall)
    kernel = kernel / kernel.sum()
    
    return kernel

# --- Part (a): Compute 5x5 Kernel for Sigma = 2 ---
sigma = 2
kernel_5x5 = get_gaussian_kernel(5, sigma)

print("--- Part (a): Normalized 5x5 Gaussian Kernel (Sigma=2) ---")
# Printing with 4 decimal places
np.set_printoptions(precision=4, suppress=True)
print(kernel_5x5)