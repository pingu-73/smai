import numpy as np
from collections import Counter

class distanceMethod:
    valid_methods=['euclidean','manhattan','cosine']
    def __init__(self, method='euclidean'):
        if method not in self.valid_methods:
            raise ValueError("Not a valid distance method")
        self.method = method

    def calculate_distance(self, x_train, x_test):
        if self.method == 'euclidean':
            distance = np.sqrt(np.sum((x_train - x_test) ** 2, axis=1))
            return distance
        elif self.method == 'manhattan':
            distance = np.sum(np.abs(x_train - x_test), axis=1)
            return distance
        else:
            x_train_norm = x_train / np.linalg.norm(x_train, axis=1, keepdims=True)
            x_test_norm = x_test / np.linalg.norm(x_test)
            distance = 1 - np.dot(x_train_norm, x_test_norm)
            return distance

class KNN:
    x_train = None
    y_train = None

    def __init__(self, k:int=1, distance_method:str='euclidean'):
        self.k = k
        self.distance_method = distance_method

    def fit(self, x_train, y_train) -> None:
        self.x_train = x_train
        self.y_train = y_train

    def predict(self, x_test) -> str:
        dist_method = distanceMethod(method=self.distance_method)

        distance = dist_method.calculate_distance(self.x_train, x_test)
        indices = np.argsort(distance)
        labels = self.y_train[indices[:self.k]]

        # labels_list = labels.tolist()
        counts = Counter(labels)
        most_common_label = counts.most_common(1)[0][0]

        return most_common_label




class Model_Evaluation:
    def __init__(self, y_true, y_pred, classes_list):
        self.y_true = np.array(y_true)
        self.y_pred = np.array(y_pred)
        self.classes_list = classes_list
        self.num_classes = len(classes_list)

    def accuracy_score(self):
        correct_predictions = np.sum(self.y_true == self.y_pred)
        total_predictions = len(self.y_true)
        return correct_predictions / total_predictions

    def precision_score(self, method='macro'):
        if method == 'macro':
            precisions = []
            for cls in self.classes_list:
                true_positive = np.sum((self.y_true == cls) & (self.y_pred == cls))
                false_positive = np.sum((self.y_true != cls) & (self.y_pred == cls))
                precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0
                precisions.append(precision)
            return np.mean(precisions)
        elif method == 'micro':
            true_positive = np.sum(self.y_true == self.y_pred)
            false_positive = np.sum((self.y_true != self.y_pred) & (self.y_pred != self.y_true))
            precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0
            return precision

    def recall_score(self, method='macro'):
        if method == 'macro':
            recalls = []
            for cls in self.classes_list:
                true_positive = np.sum((self.y_true == cls) & (self.y_pred == cls))
                false_negative = np.sum((self.y_true == cls) & (self.y_pred != cls))
                recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0
                recalls.append(recall)
            return np.mean(recalls)
        elif method == 'micro':
            true_positive = np.sum(self.y_true == self.y_pred)
            false_negative = np.sum((self.y_true != self.y_pred) & (self.y_pred != self.y_true))
            recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0
            return recall

    def f1_score(self, method='macro'):
        if method == 'macro':
            precisions = []
            recalls = []
            for cls in self.classes_list:
                true_positive = np.sum((self.y_true == cls) & (self.y_pred == cls))
                false_positive = np.sum((self.y_true != cls) & (self.y_pred == cls))
                false_negative = np.sum((self.y_true == cls) & (self.y_pred != cls))
                precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0
                recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0
                precisions.append(precision)
                recalls.append(recall)
            f1_scores = [2 * (p * r) / (p + r) if (p + r) > 0 else 0 for p, r in zip(precisions, recalls)]
            return np.mean(f1_scores)
        elif method == 'micro':
            true_positive = np.sum(self.y_true == self.y_pred)
            false_positive = np.sum((self.y_true != self.y_pred) & (self.y_pred != self.y_true))
            false_negative = np.sum((self.y_true == self.y_pred) & (self.y_pred != self.y_true))
            precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0
            recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0
            return 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0