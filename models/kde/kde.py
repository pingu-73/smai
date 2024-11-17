import numpy as np
import matplotlib.pyplot as plt

class KDE:
    def __init__(self, bandwidth=1.0, kernel='gaussian'):
        """
        Initialize KDE with specified kernel and bandwidth.
        - bandwidth: controls the smoothness of the KDE.
        - kernel: type of kernel, choose from 'box', 'gaussian', or 'triangular'.
        """
        self.bandwidth = bandwidth
        self.kernel = kernel
        self.data = None

    def _kernel_function(self, distance):
        if self.kernel == 'box':
            return 0.5 if abs(distance) <= 1 else 0
        elif self.kernel == 'gaussian':
            return np.exp(-0.5 * distance ** 2) / np.sqrt(2 * np.pi)
        elif self.kernel == 'triangular':
            return max(0, 1 - abs(distance))
        else:
            raise ValueError("Invalid kernel type. Choose 'box', 'gaussian', or 'triangular'.")

    def fit(self, data):
        """Fit the KDE to the provided data."""
        self.data = np.asarray(data)

    def predict(self, x):
        """
        Estimate the density at a given point x.
        - x: A single data point (1D array of shape (n_dimensions,))
        """
        densities = []
        for data_point in self.data:
            distance = np.linalg.norm((x - data_point) / self.bandwidth)
            densities.append(self._kernel_function(distance))
        return np.sum(densities) / (len(self.data) * self.bandwidth)

    def visualize(self):
        """
        Plot KDE density estimate if data is 2D.
        """
        if self.data.shape[1] != 2:
            raise ValueError("Visualization only supported for 2D data.")
        
        # Define grid for plotting
        x_min, x_max = self.data[:, 0].min() - 1, self.data[:, 0].max() + 1
        y_min, y_max = self.data[:, 1].min() - 1, self.data[:, 1].max() + 1
        x_grid, y_grid = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
        
        z = np.array([self.predict(np.array([x, y])) for x, y in zip(x_grid.ravel(), y_grid.ravel())])
        z = z.reshape(x_grid.shape)

        plt.contourf(x_grid, y_grid, z, cmap='viridis')
        plt.colorbar()
        plt.scatter(self.data[:, 0], self.data[:, 1], c='red', s=5)
        plt.title(f"KDE with {self.kernel} kernel")
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.show()
