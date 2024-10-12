import numpy as np

class ActivationFunction:
    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def sigmoid_derivative(x):
        return x * (1 - x)

    @staticmethod
    def tanh(x):
        return np.tanh(x)

    @staticmethod
    def tanh_derivative(x):
        return 1 - np.tanh(x) ** 2

    @staticmethod
    def relu(x):
        return np.maximum(0, x)

    @staticmethod
    def relu_derivative(x):
        return np.where(x > 0, 1, 0)

class MLPRegressor:
    def __init__(self, input_size, hidden_layers, output_size, lr=0.01, activation='relu'):
        self.layers = [input_size] + hidden_layers + [output_size]
        self.lr = lr
        self.set_activation(activation)
        self.weights = [np.random.randn(self.layers[i], self.layers[i+1]) for i in range(len(self.layers)-1)]
        self.biases = [np.zeros((1, self.layers[i+1])) for i in range(len(self.layers)-1)]

    def set_activation(self, activation):
        if activation == 'sigmoid':
            self.activation = ActivationFunction.sigmoid
            self.activation_derivative = ActivationFunction.sigmoid_derivative
        elif activation == 'tanh':
            self.activation = ActivationFunction.tanh
            self.activation_derivative = ActivationFunction.tanh_derivative
        elif activation == 'relu':
            self.activation = ActivationFunction.relu
            self.activation_derivative = ActivationFunction.relu_derivative
        else:
            raise ValueError("Unsupported activation function")

    def forward(self, X):
        self.z = []
        self.a = [X]
        for i in range(len(self.weights)):
            self.z.append(np.dot(self.a[-1], self.weights[i]) + self.biases[i])
            if i == len(self.weights) - 1:
                self.a.append(self.z[-1])  # No activation in the output layer
            else:
                self.a.append(self.activation(self.z[-1]))
        return self.a[-1]

    def backward(self, X, y):
        m = X.shape[0]
        y = y.reshape(-1, 1)
        # self.dz = [self.a[-1] - y.reshape(self.a[-1].shape)] 
        self.dz = [self.a[-1] - y]
        for i in reversed(range(len(self.weights))):
            if i != 0:  # Not for the input layer
                self.dz.append(np.dot(self.dz[-1], self.weights[i].T) * self.activation_derivative(self.a[i]))
            else:
                self.dz.append(np.dot(self.dz[-1], self.weights[i].T) * self.activation_derivative(self.a[i]))

        self.dz.reverse()
        
        self.dw = [np.dot(self.a[-2].T, self.dz[-1]) / m]
        self.db = [np.sum(self.dz[-1], axis=0, keepdims=True) / m]
        for i in range(len(self.weights)-2, -1, -1):
            self.dz.append(np.dot(self.dz[-1], self.weights[i+1].T) * self.activation_derivative(self.a[i+1]))
            self.dw.append(np.dot(self.a[i].T, self.dz[-1]) / m)
            self.db.append(np.sum(self.dz[-1], axis=0, keepdims=True) / m)
        self.dw.reverse()
        self.db.reverse()

    def update_parameters(self):
        for i in range(len(self.weights)):
            self.weights[i] -= self.lr * self.dw[i]
            self.biases[i] -= self.lr * self.db[i]

    def fit(self, X, y, epochs, method='sgd', batch_size=None, early_stopping=False, patience=5, X_val=None, y_val=None):
        y= y.ravel()
        best_loss = float('inf')
        patience_counter = 0
        for epoch in range(epochs):
            if method == 'sgd':
                for i in range(X.shape[0]):
                    self.train(X[i:i+1], y[i:i+1])
            elif method == 'batch':
                self.train(X, y)
            elif method == 'mini_batch':
                if batch_size is None:
                    raise ValueError("batch_size must be specified for mini_batch method")
                indices = np.arange(X.shape[0])
                np.random.shuffle(indices)
                for start_idx in range(0, X.shape[0], batch_size):
                    end_idx = min(start_idx + batch_size, X.shape[0])
                    batch_indices = indices[start_idx:end_idx]
                    self.train(X[batch_indices], y[batch_indices])
            else:
                raise ValueError("Unsupported training method")
            
            # Early stopping
            if early_stopping and X_val is not None and y_val is not None:
                val_loss = self.loss(X_val, y_val)
                if val_loss < best_loss:
                    best_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch}")
                    break


    def train(self, X, y):
        self.output = self.forward(X)
        self.backward(X, y)
        self.update_parameters()

    def predict(self, X):
        return self.forward(X)

    def loss(self, X, y):
        predictions = self.predict(X)
        return np.mean((predictions - y) ** 2)

    def gradient_check(self, X, y, epsilon=1e-7, tolerance=1e-5):
        self.forward(X)
        self.backward(X, y)
        
        analytical_grads = {
            'weights': self.weights,
            'biases': self.biases
        }
        
        numerical_grads = self.compute_numerical_gradient(X, y, epsilon)
        
        for param_name in analytical_grads:
            analytical_grad = analytical_grads[param_name]
            numerical_grad = numerical_grads[param_name]
            relative_error = np.linalg.norm(analytical_grad - numerical_grad) / (np.linalg.norm(analytical_grad) + np.linalg.norm(numerical_grad))
            if relative_error > tolerance:
                print(f"Gradient check failed for {param_name}. Relative error: {relative_error}")
            else:
                print(f"Gradient check passed for {param_name}. Relative error: {relative_error}")

    def compute_numerical_gradient(self, X, y, epsilon=1e-7):
        numerical_grads = {}
        for param_name in ['weights', 'biases']:
            param = getattr(self, param_name)
            grad = np.zeros_like(param)
            it = np.nditer(param, flags=['multi_index'], op_flags=['readwrite'])
            while not it.finished:
                idx = it.multi_index
                original_value = param[idx]
                
                param[idx] = original_value + epsilon
                loss_plus = self.loss(X, y)
                
                param[idx] = original_value - epsilon
                loss_minus = self.loss(X, y)
                
                grad[idx] = (loss_plus - loss_minus) / (2 * epsilon)
                param[idx] = original_value
                it.iternext()
            numerical_grads[param_name] = grad
        return numerical_grads