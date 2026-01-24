import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Necessary for 3D plotting

def get_gaussian_kernel(k_size, sigma):
    
    #Generates a 2D Gaussian kernel using the formula:
    #G(x, y) = (1 / (2 * pi * sigma^2)) * exp( -(x^2 + y^2) / (2 * sigma^2) )
    
    # 1. Create a grid of (x, y) coordinates
    # For a 5x5 kernel, the center is at index 2. Coordinates range from -2 to +2.
    center = k_size // 2
    x, y = np.mgrid[-center:center+1, -center:center+1]
    
    # 2. Apply the Gaussian formula
    normalizer = 1 / (2.0 * np.pi * sigma**2)
    kernel = normalizer * np.exp(-(x**2 + y**2) / (2.0 * sigma**2))
    
    # 3. Normalize so the sum of all elements is 1
    kernel = kernel / kernel.sum()
    
    return kernel

# --- Part (a): Compute 5x5 Kernel for Sigma = 2 ---
sigma = 2
kernel_5x5 = get_gaussian_kernel(5, sigma)

print("--- Part (a): Normalized 5x5 Gaussian Kernel (Sigma=2) ---")
np.set_printoptions(precision=4, suppress=True)
print(kernel_5x5)

# --- Part (b): Visualize 51x51 Kernel as 3D Surface Plot ---
print("\n--- Generating Part (b) 3D Plot... check popup window ---")

# 1. Generate the large kernel
k_size_large = 51
kernel_large = get_gaussian_kernel(k_size_large, sigma)

# 2. Setup 3D Plotting
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# 3. Create X, Y grid for the plot axes (from -25 to +25)
center = k_size_large // 2
X, Y = np.meshgrid(np.arange(-center, center+1), np.arange(-center, center+1))

# 4. Plot the surface
# cmap='viridis' adds color (Yellow=High, Purple=Low)
surf = ax.plot_surface(X, Y, kernel_large, cmap='viridis', edgecolor='none')

# 5. Add labels and title
ax.set_title(f'3D Gaussian Kernel ({k_size_large}x{k_size_large}, $\sigma$={sigma})')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Kernel Value (Height)')

# Add a color bar to show height scale
fig.colorbar(surf, shrink=0.5, aspect=10)

plt.show()