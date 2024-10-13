import numpy as np
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from models.mlp.mlp_regression import MLPRegression

class AutoEncoder:
    def __init__(self, input_dim, latent_dim, hidden_layer_sizes=(100,), activation='relu',
                 learning_rate=0.001, max_iter=1000, tol=1e-4, batch_size=32):
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        
        # Encoder: input_dim -> hidden layer(s) -> latent_dim
        self.encoder = MLPRegression(hidden_layer_sizes=hidden_layer_sizes + (latent_dim,),
                                      activation=activation, learning_rate=learning_rate,
                                      max_iter=max_iter, tol=tol, batch_size=batch_size)
        
        # Decoder: latent_dim -> hidden layer(s) -> input_dim
        self.decoder = MLPRegression(hidden_layer_sizes=(latent_dim,) + hidden_layer_sizes[::-1],
                                      activation=activation, learning_rate=learning_rate,
                                      max_iter=max_iter, tol=tol, batch_size=batch_size)

    def fit(self, X):
        latent_representation = self.encoder.fit(X, X).predict(X)
        self.decoder.fit(latent_representation, X)

    def get_latent(self, X):
        return self.encoder.predict(X)

    def reconstruct(self, X):
        latent_representation = self.get_latent(X)
        return self.decoder.predict(latent_representation)
