import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import imageio
import os
import glob
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from performance_measures.performance_lin_reg import PerformanceMeasures

class LinearRegression(PerformanceMeasures):
    def __init__(self, points, salt, k=1, learning_rate=0.1, epochs=100, lambda_reg = 0,L1=0.1,L2=0.1):
        self.points = points
        self.k = k
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.beta_vector = np.zeros(self.k + 1, dtype=np.float32)
        self.lambda_reg = lambda_reg   # Lambda for Regression and Regularization 
        
        self.L1 = L1
        self.L2 = L2
        self.salt = salt
        self.mse_history = []  
        self.variance_history = []  
        self.stddev_history = []  

        self.y_vals = points[:, 1]
        self.variance_data = np.var(self.y_vals)
        self.stddev_data = np.std(self.y_vals)
        
    def predict(self, data):
        y_predicted = [sum(self.beta_vector[j] * (data[i, 0]**j) for j in range(len(self.beta_vector))) for i in range(len(data))]
        mse = self.mean_sq_error(data, self.beta_vector)
        variance = self.variance(y_predicted)
        stddev = self.stddev(y_predicted)
        return mse, variance, stddev
    
    def fit(self):
        frames = []
        for epoch in range(self.epochs):
            self.gradient_descent()
            mse, variance, stddev = self.predict(self.points)
            
            self.mse_history.append(mse)
            self.variance_history.append(variance)
            self.stddev_history.append(stddev)
            
            filename = f'./assignments/1/figures/{self.salt}_frame_k{self.k}_epoch{epoch + 1}.png'
            self.plot_fit(self.points, epoch, filename)
            frames.append(imageio.imread(filename))
        
        gif_filename = f'./assignments/1/figures/{self.salt}_convergence_k{self.k}.gif'
        imageio.mimsave(gif_filename, frames, duration=0.1)
        
        for file in glob.glob(f'./assignments/1/figures/{self.salt}_frame_k{self.k}_epoch*.png'):
            os.remove(file)
    
    def gradient_descent(self):
        beta_gradient = np.zeros(len(self.beta_vector))
        no_of_points = len(self.points)
        
        for i in range(no_of_points):
            x = self.points[i, 0]
            y = self.points[i, 1]
            y_pred = sum(self.beta_vector[j] * (x**j) for j in range(len(self.beta_vector)))
            
            for j in range(len(self.beta_vector)):
                    beta_gradient[j] += (-2 / no_of_points) * (y - y_pred) * (x**j)
            
        if self.lambda_reg == 1:
            for j in range(len(self.beta_vector)):
                sign = np.sign(self.beta_vector[j])  
                beta_gradient[j] += self.L1 * sign  

        elif (self.lambda_reg==2):
            beta_gradient += 2*self.L2*self.beta_vector

        self.beta_vector -= self.learning_rate * beta_gradient

    def plot_fit(self, data, epoch, filename):
        x_vals = np.linspace(min(data[:, 0]), max(data[:, 0]), 100)
        y_vals = sum(self.beta_vector[j] * (x_vals**j) for j in range(len(self.beta_vector)))
        
        fig, axs = plt.subplots(2, 2, figsize=(10, 8))
        fig.suptitle(f"Epoch {epoch + 1}")
        axs[0, 0].scatter(data[:, 0], data[:, 1], color='blue', label='Data Points')
        axs[0, 0].plot(x_vals, y_vals, color='red', label='Fitted Curve')
        axs[0, 0].legend()
        axs[0, 0].set_title("Line Fitting")
        
        axs[0, 1].plot(range(1, epoch + 2), self.mse_history, 'b-')
        axs[0, 1].set_title("MSE")
        axs[0, 1].set_xlabel('Epoch')
        axs[0, 1].set_ylabel('MSE')
        
        axs[1, 0].plot(range(1, epoch + 2), self.variance_history, 'g-', label='Predicted Variance')
        axs[1, 0].axhline(self.variance_data, color='orange', linestyle='--', label='Observed Variance')
        axs[1, 0].set_title("Variance")
        axs[1, 0].set_xlabel('Epoch')
        axs[1, 0].set_ylabel('Variance')
        axs[1, 0].legend()
        
        axs[1, 1].plot(range(1, epoch + 2), self.stddev_history, 'r-', label='Predicted Std Dev')
        axs[1, 1].axhline(self.stddev_data, color='orange', linestyle='--', label='Observed Std Dev')
        axs[1, 1].set_title("Standard Deviation")
        axs[1, 1].set_xlabel('Epoch')
        axs[1, 1].set_ylabel('Standard Deviation')
        axs[1, 1].legend()
        
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()