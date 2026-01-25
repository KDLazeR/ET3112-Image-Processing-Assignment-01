import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def get_gaussian_kernel(k_size, sigma):
    center = k_size // 2
    x, y = np.mgrid[-center:center+1, -center:center+1]
    
    normalizer = 1 / (2.0 * np.pi * sigma**2)
    kernel = normalizer * np.exp(-(x**2 + y**2) / (2.0 * sigma**2))
    kernel = kernel / kernel.sum()
    return kernel, x, y

# --- Parameters ---
k_size = 51
sigma = 5  # Using a larger sigma for a smoother looking plot

# 1. Compute Base Gaussian
G, x, y = get_gaussian_kernel(k_size, sigma)

# 2. Compute X-Derivative (Gx)
# Formula: -(x / sigma^2) * G
Gx = -(x / sigma**2) * G

# --- 3D Visualization ---
print("Generating 3D plot for Derivative of Gaussian (Gx)...")

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# Create plot grid
center = k_size // 2
X, Y = np.meshgrid(np.arange(-center, center+1), np.arange(-center, center+1))

# Plot the surface
# cmap='coolwarm' is perfect here: Red=Positive (Hill), Blue=Negative (Valley)
surf = ax.plot_surface(X, Y, Gx, cmap='coolwarm', edgecolor='none')

ax.set_title(f'3D Derivative of Gaussian (Gx) Kernel\nSize={k_size}, $\sigma$={sigma}')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Weight')

# Add colorbar
fig.colorbar(surf, shrink=0.5, aspect=10)

plt.show()