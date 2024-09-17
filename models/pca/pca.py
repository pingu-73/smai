import numpy as np

class PCA():
    def __init__(self, num_components:int):
        self.num_components = num_components
        self.principal_components = None
        self.transformed_data = None
        
    def fit(self, data):
        data = data - np.mean(data, axis=0)
        cov_mat = np.cov(data, rowvar=False)
        eigen_values, eigen_vectors = np.linalg.eig(cov_mat)

        eigen_values = eigen_values.real
        eigen_vectors = eigen_vectors.real
        sorted_indices = np.argsort(eigen_values)[::-1]
        sorted_eigen_vectors = eigen_vectors[:, sorted_indices]
        self.principal_components = sorted_eigen_vectors[:, :self.num_components]

    def transform(self, data):
        data = data - np.mean(data, axis=0)
        self.transformed_data = data @ self.principal_components
        return self.transformed_data

    def checkPCA(self) -> bool:
        if self.transform_data == None:
            return False
        elif self.transform_data.shape[1] != self.num_components:
            return False
        return True