import numpy as np

class PcaAutoencoder:
    def __init__(self, n_components):
        self.n_components = n_components
        self.mean = None
        self.eigenvectors = None
        
    def fit(self, X):
        self.mean = np.mean(X, axis=0)
        X_centered = X - self.mean
        
        cov_matrix = np.cov(X_centered, rowvar=False)
        
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        self.eigenvectors = eigenvectors[:, :self.n_components]
    
    def encode(self, X):
        X_centered = X - self.mean
        return np.dot(X_centered, self.eigenvectors)
    
    def forward(self, X):
        encoded = self.encode(X)
        reconstructed = np.dot(encoded, self.eigenvectors.T) + self.mean
        return reconstructed
