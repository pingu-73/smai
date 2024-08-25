# from math import sqrt

# class Distance:
#     @staticmethod
#     def euclidean(r1, r2):
#         dis = 0
#         for i in range(len(r1)):
#             dis += (r1[i] - r2[i]) ** 2
#         return sqrt(dis)

# class Neighbour:
#     @staticmethod
#     def get_neighbour(train, test_r, n):
#         dis_list = list()
#         for i in train:
#             dis = Distance.euclidean(test_r, i)
#             dis_list.append((i, dis))
#         dis_list.sort(key=lambda tup: tup[1])
        
#         neighbors = list()
#         for i in range(n):
#             neighbors.append(dis_list[i][0])
#         return neighbors

# class Predict:
#     @staticmethod
#     def predict_classification(train, test_row, num_neighbors):
#         neighbors = Neighbour.get_neighbour(train, test_row, num_neighbors)
#         output_values = [row[-1] for row in neighbors]
#         prediction = max(set(output_values), key=output_values.count)
#         return prediction

# class Knn:
#     @staticmethod
#     def knn(train, test, num_neighbors):
#         predictions = list()
#         for row in test:
#             output = Predict.predict_classification(train, row, num_neighbors)
#             predictions.append(output)
#         return predictions

import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

import numpy as np
from collections import Counter

class K_Nearest_Neighbour:
    def __init__(self, k=3, distance_formula='euclidean'):
        self.k = k
        self.distance_formula = distance_formula
    
    def euclidean(self, x1, x2):
        return np.sqrt(np.sum((x1 - x2) ** 2, axis=1))
    
    def manhattan(self, x1, x2):
        return np.sum(np.abs(x1 - x2), axis=1)
    
    def fit(self, points):
        self.points = points
    
    def predict(self, new_point):
        if self.distance_formula == 'euclidean':
            distance_function = self.euclidean
        elif self.distance_formula == 'manhattan':
            distance_function = self.manhattan
        
        distances = []
        for category in self.points:
            points_in_category = np.array(self.points[category])
            calculated_distances = distance_function(points_in_category, new_point.reshape(1, -1))
            distances.extend(zip(calculated_distances, [category] * len(calculated_distances)))

        categories = [category for _, category in sorted(distances)[:self.k]]
        return Counter(categories).most_common(1)[0][0]


import numpy as np

class Evaluation:
    def accuracy(self, y_true, y_pred):
        return np.mean(y_true == y_pred)
    
    def precision(self, y_true, y_pred, average='macro'):
        classes = np.unique(y_true)
        if average == 'micro':
            tp = np.sum((y_pred == y_true) & (y_true != 0))
            fp = np.sum((y_pred != y_true) & (y_pred != 0))
            return tp / (tp + fp) if (tp + fp) > 0 else 0
        else:
            precisions = []
            for cls in classes:
                tp = np.sum((y_pred == cls) & (y_true == cls))
                fp = np.sum((y_pred == cls) & (y_true != cls))
                precisions.append(tp / (tp + fp) if (tp + fp) > 0 else 0)
            return np.mean(precisions) if average == 'macro' else np.sum(precisions) / len(classes)

    def recall(self, y_true, y_pred, average='macro'):
        classes = np.unique(y_true)
        if average == 'micro':
            tp = np.sum((y_pred == y_true) & (y_true != 0))
            fn = np.sum((y_pred != y_true) & (y_true != 0))
            return tp / (tp + fn) if (tp + fn) > 0 else 0
        else:
            recalls = []
            for cls in classes:
                tp = np.sum((y_pred == cls) & (y_true == cls))
                fn = np.sum((y_pred != cls) & (y_true == cls))
                recalls.append(tp / (tp + fn) if (tp + fn) > 0 else 0)
            return np.mean(recalls) if average == 'macro' else np.sum(recalls) / len(classes)

    def f1_score(self, y_true, y_pred, average='macro'):
        precision = self.precision(y_true, y_pred, average)
        recall = self.recall(y_true, y_pred, average)
        return 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0