import numpy as np

class ActivationFunction:
    @staticmethod
    def sigmoid(x):
        if isinstance(x, (int, float)):
            return 1 / (1 + np.exp(-x))
        else:
            return 1 / (1 + np.exp(-np.asarray(x, dtype=float)))

    @staticmethod
    def sigmoid_derivative(x):
        x = np.asarray(x, dtype=float)
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

class MLP_MULTI:
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
        X = np.asarray(X, dtype=float)
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.activation(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = ActivationFunction.sigmoid(self.z2)  # Use sigmoid for multi-label output
        return self.a2

    def backward(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        self.error = y - self.a2
        self.a2_delta = self.error * ActivationFunction.sigmoid_derivative(self.a2)
        self.a1_error = self.a2_delta.dot(self.W2.T)
        self.a1_delta = self.a1_error * self.activation_derivative(self.a1)
        self.W1 += X.T.dot(self.a1_delta) * self.lr
        self.W2 += self.a1.T.dot(self.a2_delta) * self.lr
        self.b1 += np.sum(self.a1_delta, axis=0, keepdims=True) * self.lr
        self.b2 += np.sum(self.a2_delta, axis=0, keepdims=True) * self.lr

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