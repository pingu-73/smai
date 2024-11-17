import os
import sys
import numpy as np
import matplotlib.pyplot as plt
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from models.kde.kde import KDE
from models.gmm.gmm import GMM
# from sklearn.mixture import GaussianMixture as GMM

def generate_circle_data(radius, num_samples, noise_level):
    """
    Generates points within a circle of a given radius, with added noise.
    - radius: Radius of the circle.
    - num_samples: Number of points to generate.
    - noise_level: Standard deviation of Gaussian noise added to the points.
    """
    angles = 2 * np.pi * np.random.rand(num_samples)
    radii = radius * np.sqrt(np.random.rand(num_samples))
    x = radii * np.cos(angles) + np.random.normal(0, noise_level, num_samples)
    y = radii * np.sin(angles) + np.random.normal(0, noise_level, num_samples)
    return np.vstack((x, y)).T

# Parameters
outer_circle_radius = 5.0
inner_circle_radius = 2.0
outer_circle_samples = 3000
inner_circle_samples = 500
noise_level_outer = 0.5
noise_level_inner = 0.2

# Generate data
outer_circle_data = generate_circle_data(outer_circle_radius, outer_circle_samples, noise_level_outer)
inner_circle_data = generate_circle_data(inner_circle_radius, inner_circle_samples, noise_level_inner)

# Combine data for visualization
data = np.vstack((outer_circle_data, inner_circle_data))

# Plot the data
plt.figure(figsize=(8, 8))
plt.scatter(outer_circle_data[:, 0], outer_circle_data[:, 1], color='blue', s=5, label='Outer Circle')
plt.scatter(inner_circle_data[:, 0], inner_circle_data[:, 1], color='red', s=5, label='Inner Circle')
plt.axis('equal')
plt.title("Synthetic Data: Diffused Outer Circle and Dense Inner Circle")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.show()





# Assuming KDE class is defined as before and dataset is already generated

# Fit KDE
kde = KDE(bandwidth=0.5, kernel='gaussian')
kde.fit(data)

# Fit GMM with 2 components
gmm_2 = GMM(n_components=2, covariance_type='full', random_state=42)
gmm_2.fit(data)

# Fit GMM with increasing components (e.g., 5 components)
gmm_5 = GMM(n_components=5, covariance_type='full', random_state=42)
gmm_5.fit(data)

# Plot KDE visualization
plt.figure(figsize=(15, 5))
# Plot GMM with 2 components
plt.subplot(1, 3, 1)
plt.scatter(data[:, 0], data[:, 1], s=5, color='grey')
x, y = np.meshgrid(np.linspace(-6, 6, 100), np.linspace(-6, 6, 100))
z = np.exp(gmm_2.score_samples(np.array([x.ravel(), y.ravel()]).T))
z = z.reshape(x.shape)
plt.contourf(x, y, z, cmap='viridis')
plt.colorbar()
plt.scatter(data[:, 0], data[:, 1], color='grey', s=5)
plt.xlabel('X')
plt.ylabel('Y')
plt.title("GMM Fit (2 Components)")

# Plot GMM with 5 components
plt.subplot(1, 3, 2)
plt.scatter(data[:, 0], data[:, 1], s=5, color='grey')
z = np.exp(gmm_5.score_samples(np.array([x.ravel(), y.ravel()]).T))
z = z.reshape(x.shape)
plt.contourf(x, y, z, cmap='viridis')
plt.colorbar()
plt.scatter(data[:, 0], data[:, 1], color='grey', s=5)
plt.xlabel('X')
plt.ylabel('Y')
plt.title("GMM Fit (5 Components)")

plt.subplot(1, 3, 3)
kde.visualize()
plt.title("KDE Fit")

plt.show()
