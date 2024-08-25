import numpy as np
class PerformanceMeasures:
    def mean_sq_error(self, data, beta_vector):
        error = 0
        no_of_points = len(data)
        for i in range(no_of_points):
            x = data[i, 0]
            y = data[i, 1]
            y_pred = sum(beta_vector[j] * (x**j) for j in range(len(beta_vector)))
            error += (1 / no_of_points) * ((y - y_pred) ** 2)
        return error
    
    def variance(self, y_predicted):
        return np.var(y_predicted)
    
    def stddev(self, y_predicted):
        return np.std(y_predicted)