import numpy as np

def get_gaussian_kernel(k_size, sigma):
    # Generates the base Gaussian kernel G(x,y)
    center = k_size // 2
    x, y = np.mgrid[-center:center+1, -center:center+1]
    
    normalizer = 1 / (2.0 * np.pi * sigma**2)
    kernel = normalizer * np.exp(-(x**2 + y**2) / (2.0 * sigma**2))
    
    # Standard Gaussian sums to 1
    kernel = kernel / kernel.sum()
    return kernel, x, y

# --- Parameters ---
sigma = 2
k_size = 5

# 1. Get Base Gaussian and Grid Coordinates
G, x, y = get_gaussian_kernel(k_size, sigma)

# 2. Compute Derivative Kernels using formulas from 6(a)
# Gx = -(x / sigma^2) * G
Gx = -(x / sigma**2) * G

# Gy = -(y / sigma^2) * G
Gy = -(y / sigma**2) * G

# --- Print Results ---
np.set_printoptions(precision=4, suppress=True)

print("--- Part (b): Derivative of Gaussian (Gx) ---")
print("Detects Vertical Edges")
print(Gx)

print("\n--- Part (b): Derivative of Gaussian (Gy) ---")
print("Detects Horizontal Edges")
print(Gy)