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

    @staticmethod
    def linear(x):
        return x

    @staticmethod
    def linear_derivative(x):
        return np.ones_like(x)

class MLP:
    def __init__(self, input_size, hidden_size, output_size, lr=0.01, activation='sigmoid'):
        self.W1 = np.random.randn(input_size, hidden_size)
        self.W2 = np.random.randn(hidden_size, output_size)
        self.b1 = np.zeros((1, hidden_size))
        self.b2 = np.zeros((1, output_size))
        self.lr = lr
        self.set_activation(activation)

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
        elif activation == 'linear':
            self.activation = ActivationFunction.linear
            self.activation_derivative = ActivationFunction.linear_derivative
        else:
            raise ValueError("Unsupported activation function")

    def forward(self, X):
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.activation(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.activation(self.z2)
        return self.a2

    def backward(self, X, y):
        self.error = y - self.a2
        self.a2_delta = self.error * self.activation_derivative(self.a2)
        self.a1_error = self.a2_delta.dot(self.W2.T)
        self.a1_delta = self.a1_error * self.activation_derivative(self.a1)
        self.W1 += X.T.dot(self.a1_delta) * self.lr
        self.W2 += self.a1.T.dot(self.a2_delta) * self.lr
        self.b1 += np.sum(self.a1_delta, axis=0, keepdims=True) * self.lr
        self.b2 += np.sum(self.a2_delta, axis=0, keepdims=True) * self.lr

    def compute_numerical_gradient(self, X, y, epsilon=1e-7):
        numerical_grads = {}
        for param_name in ['W1', 'W2', 'b1', 'b2']:
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

    def loss(self, X, y):
        output = self.forward(X)
        return np.mean((y - output) ** 2)

    def gradient_check(self, X, y, epsilon=1e-7, tolerance=1e-5):
        self.forward(X)
        self.backward(X, y)
        
        analytical_grads = {
            'W1': self.W1,
            'W2': self.W2,
            'b1': self.b1,
            'b2': self.b2
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

    def train(self, X, y):
        self.output = self.forward(X)
        self.backward(X, y)

    def predict(self, X):
        return self.forward(X)

    def fit(self, X, y, epochs, method='sgd', batch_size=None):
        if method == 'sgd':
            self.sgd(X, y, epochs)
        elif method == 'batch':
            self.batch_gradient_descent(X, y, epochs)
        elif method == 'mini_batch':
            if batch_size is None:
                raise ValueError("batch_size must be specified for mini_batch method")
            self.mini_batch_gradient_descent(X, y, epochs, batch_size)
        else:
            raise ValueError("Unsupported training method")

    def sgd(self, X, y, epochs):
        for _ in range(epochs):
            for i in range(X.shape[0]):
                self.train(X[i:i+1], y[i:i+1])

    def batch_gradient_descent(self, X, y, epochs):
        for _ in range(epochs):
            self.train(X, y)

    def mini_batch_gradient_descent(self, X, y, epochs, batch_size):
        for _ in range(epochs):
            indices = np.arange(X.shape[0])
            np.random.shuffle(indices)
            for start_idx in range(0, X.shape[0], batch_size):
                end_idx = min(start_idx + batch_size, X.shape[0])
                batch_indices = indices[start_idx:end_idx]
                self.train(X[batch_indices], y[batch_indices])

    def save(self, path):
        np.savez(path, W1=self.W1, W2=self.W2, b1=self.b1, b2=self.b2)

    def load(self, path):
        data = np.load(path)
        self.W1 = data['W1']
        self.W2 = data['W2']
        self.b1 = data['b1']
        self.b2 = data['b2']