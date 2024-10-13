import numpy as np
class MLPRegression:
    def __init__(self, hidden_layer_sizes=(100,), activation='relu', learning_rate=0.001,
                 max_iter=1000, tol=1e-4, early_stopping=False, validation_split=0.1,
                 patience=10, alpha=0.0001, batch_size=32):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.tol = tol
        self.early_stopping = early_stopping
        self.validation_split = validation_split
        self.patience = patience
        self.alpha = alpha  # L2 regularization
        self.batch_size = batch_size
        self.weights = []
        self.biases = []

    def _initialize_weights(self, n_inputs):
        self.weights = []
        self.biases = []
        layer_sizes = [n_inputs] + list(self.hidden_layer_sizes) + [1]
        for i in range(len(layer_sizes) - 1):
            limit = np.sqrt(6 / (layer_sizes[i] + layer_sizes[i + 1]))
            weight_matrix = np.random.uniform(-limit, limit, (layer_sizes[i], layer_sizes[i + 1]))
            bias_vector = np.zeros((1, layer_sizes[i + 1]))
            self.weights.append(weight_matrix)
            self.biases.append(bias_vector)

    def _activation(self, z, derivative=False):
        if self.activation == 'relu':
            if not derivative:
                return np.maximum(0, z)
            else:
                return (z > 0).astype(float)
        elif self.activation == 'sigmoid':
            if not derivative:
                return 1 / (1 + np.exp(-np.clip(z, -500, 500)))
            else:
                s = self._activation(z)
                return s * (1 - s)
        elif self.activation == 'tanh':
            if not derivative:
                return np.tanh(z)
            else:
                return 1 - np.tanh(z) ** 2 + 1e-8  # Add small epsilon
        else:
            raise ValueError("Unsupported activation function.")

    def _forward(self, X):
        a = X
        self.activations = [X]
        self.z_values = []

        for weight, bias in zip(self.weights, self.biases):
            z = a @ weight + bias
            self.z_values.append(z)
            a = self._activation(z)
            self.activations.append(a)

        return a

    def _backward(self, X_batch, y_batch):
        deltas = []
        self.layers = self.activations

        output = self.layers[-1]
        delta = (output - y_batch.reshape(-1, 1)) * self._activation(self.z_values[-1], derivative=True)
        deltas.append(delta)

        for i in reversed(range(len(self.weights) - 1)):
            delta = deltas[-1] @ self.weights[i + 1].T * self._activation(self.z_values[i], derivative=True)
            deltas.append(delta)

        deltas.reverse()

        for i in range(len(self.weights)):
            gradient_w = self.layers[i].T @ deltas[i] / X_batch.shape[0] + self.alpha * self.weights[i]  # L2 regularization
            gradient_b = np.sum(deltas[i], axis=0, keepdims=True) / X_batch.shape[0]
            
            self.weights[i] -= self.learning_rate * gradient_w

    def _compute_loss(self, y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)

        if self.early_stopping:
            split_idx = int(X.shape[0] * (1 - self.validation_split))
            X_train, X_val = X[:split_idx], X[split_idx:]
            y_train, y_val = y[:split_idx], y[split_idx:]
        else:
            X_train, y_train = X, y

        self._initialize_weights(X_train.shape[1])
        best_loss = float('inf')
        patience_counter = 0

        for epoch in range(self.max_iter):
            indices = np.arange(X_train.shape[0])
            np.random.shuffle(indices)
            X_train_shuffled = X_train[indices]
            y_train_shuffled = y_train[indices]

            for start in range(0, X_train.shape[0], self.batch_size):
                end = start + self.batch_size
                X_batch = X_train_shuffled[start:end]
                y_batch = y_train_shuffled[start:end]

                y_pred = self._forward(X_batch)
                self._backward(X_batch, y_batch)

            if self.early_stopping:
                val_loss = self._compute_loss(y_val, self._forward(X_val))
                train_loss = self._compute_loss(y_train, self._forward(X_train))
                print(f'Epoch {epoch}, Training loss: {train_loss:.4f}, Validation loss: {val_loss:.4f}')

                if val_loss < best_loss:
                    best_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter >= self.patience:
                    print("Early stopping.")
                    break
            else:
                train_loss = self._compute_loss(y_train, self._forward(X_train))
                print(f'Epoch {epoch}, Loss: {train_loss:.4f}')

            if np.isnan(train_loss):
                print("NaN loss encountered. Stopping training.")
                break

        return self

    def predict(self, X):
        return self._forward(X)
    


class MLP__N:
    def __init__(self, input_size, loss_function, learning_rate=0.01, epochs=1000):
        self.weights = np.random.randn(input_size)
        self.bias = np.random.randn()
        self.loss_function = loss_function
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.losses = []

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def predict(self, X):
        z = np.dot(X, self.weights) + self.bias
        return self.sigmoid(z)

    def compute_loss(self, y_true, y_pred):
        if self.loss_function == 'BCE':
            return -np.mean(y_true * np.log(y_pred + 1e-12) + (1 - y_true) * np.log(1 - y_pred + 1e-12))
        elif self.loss_function == 'MSE':
            return np.mean((y_true - y_pred) ** 2)

    def train(self, X, y):
        for epoch in range(self.epochs):

            y_pred = self.predict(X)

            loss = self.compute_loss(y, y_pred)
            self.losses.append(loss)

            if self.loss_function == 'BCE':
                gradient = np.dot(X.T, (y_pred - y)) / y.size
            elif self.loss_function == 'MSE':
                gradient = -2 * np.dot(X.T, (y - y_pred)) / y.size

            self.weights -= self.learning_rate * gradient
            self.bias -= self.learning_rate * np.mean(y_pred - y)