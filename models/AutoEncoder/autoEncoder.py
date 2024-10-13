import numpy as np
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from models.mlp.mlp_regression import MLPRegression

class AutoEncoder:
    def _init_(self, input_size, latent_size=6, learning_rate=0.001, max_iter=1000, batch_size=32, tol=1e-4):
        self.input_size = input_size
        self.latent_size = latent_size
        
        self.model = MLPRegression(hidden_layer_sizes=(latent_size,), activation='relu', 
                                   learning_rate=learning_rate, max_iter=max_iter, 
                                   batch_size=batch_size, tol=tol)

    def fit(self, X):
        print("Training the AutoEncoder...")
        self.model.fit(X, X)

    def get_latent(self, X):
        self.model._initialize_weights(X.shape[1])
        self.model._forward(X)
        
        latent_representation = self.model.activations[1]
        return latent_representation

    def reconstruct(self, X):
        reconstructed_X = self.model.predict(X)
        return reconstructed_X
