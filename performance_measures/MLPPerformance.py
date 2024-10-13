import numpy as np
from sklearn.metrics import confusion_matrix

class PerformanceMatrix:
    def __init__(self, y_true, y_pred, average='weighted'):
        self.y_true = y_true
        self.y_pred = y_pred
        self.average = average
        # self.classes = np.unique(y_true) 
        # self.classes = np.unique(np.concatenate((y_true, y_pred)))
        self.classes = np.unique(y_true)

    def softmax(self, logits):
        """Apply softmax to logits."""
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))  # for numerical stability
        return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

    def acc(self):
        correct = np.sum(self.y_true == self.y_pred)
        total = len(self.y_true)
        return correct / total
    
    def accuracy_score(self):
        correct = np.sum(self.y_true == self.y_pred)
        total = len(self.y_true)
        return correct / total

    def precision_score(self):
        unique_classes = np.unique(self.y_true)
        precisions = []

        for cls in unique_classes:
            true_positive = np.sum((self.y_true == cls) & (self.y_pred == cls))
            false_positive = np.sum((self.y_true != cls) & (self.y_pred == cls))
            
            if true_positive + false_positive == 0:
                precision = 0  # Handle division by zero if no positive predictions
            else:
                precision = true_positive / (true_positive + false_positive)
            
            precisions.append(precision)

        if self.average == 'weighted':
            class_counts = [np.sum(self.y_true == cls) for cls in unique_classes]
            weighted_precision = np.average(precisions, weights=class_counts)
            return weighted_precision
        else:  # For 'macro' average
            return np.mean(precisions)

    def recall_score(self):
        unique_classes = np.unique(self.y_true)
        recalls = []

        for cls in unique_classes:
            true_positive = np.sum((self.y_true == cls) & (self.y_pred == cls))
            false_negative = np.sum((self.y_true == cls) & (self.y_pred != cls))

            if true_positive + false_negative == 0:
                recall = 0  # Handle division by zero if no relevant instances
            else:
                recall = true_positive / (true_positive + false_negative)

            recalls.append(recall)

        if self.average == 'weighted':
            class_counts = [np.sum(self.y_true == cls) for cls in unique_classes]
            weighted_recall = np.average(recalls, weights=class_counts)
            return weighted_recall
        else:  # For 'macro' average
            return np.mean(recalls)

    def f1_score(self):
        unique_classes = np.unique(self.y_true)
        f1_scores = []

        for cls in unique_classes:
            true_positive = np.sum((self.y_true == cls) & (self.y_pred == cls))
            false_positive = np.sum((self.y_true != cls) & (self.y_pred == cls))
            false_negative = np.sum((self.y_true == cls) & (self.y_pred != cls))

            if true_positive == 0:
                precision = 0
                recall = 0
            else:
                precision = true_positive / (true_positive + false_positive)
                recall = true_positive / (true_positive + false_negative)

            if precision + recall == 0:
                f1 = 0
            else:
                f1 = 2 * (precision * recall) / (precision + recall)

            f1_scores.append(f1)

        if self.average == 'weighted':
            class_counts = [np.sum(self.y_true == cls) for cls in unique_classes]
            weighted_f1 = np.average(f1_scores, weights=class_counts)
            return weighted_f1
        else:  # For 'macro' average
            return np.mean(f1_scores)
    
    def confusion_matrix(self):
        """Compute the confusion matrix using sklearn's method."""
        return confusion_matrix(self.y_true, self.y_pred)

    # def confusion_matrix(self):
    #     """Compute confusion matrix from scratch."""
    #     # Initialize the confusion matrix
    #     matrix = np.zeros((len(self.classes), len(self.classes)), dtype=int)

    #     # Loop over true and predicted labels to populate the matrix
    #     for true_label, pred_label in zip(self.y_true, self.y_pred):
    #         true_idx = np.where(self.classes == true_label)[0][0]  # Find index of true label
    #         pred_idx = np.where(self.classes == pred_label)[0][0]  # Find index of predicted label
    #         matrix[true_idx, pred_idx] += 1

    #     return matrix
    # def confusion_matrix(self):
    #     """Compute confusion matrix from scratch."""
    #     # Get unique classes from both true and predicted labels
    #     all_classes = np.unique(np.concatenate((self.y_true, self.y_pred)))
        
    #     # Initialize the confusion matrix
    #     matrix = np.zeros((len(all_classes), len(all_classes)), dtype=int)

    #     # Create a mapping from class labels to matrix indices
    #     class_to_index = {cls: idx for idx, cls in enumerate(all_classes)}

    #     # Loop over true and predicted labels to populate the matrix
    #     for true_label, pred_label in zip(self.y_true, self.y_pred):
    #         true_idx = class_to_index[true_label]
    #         pred_idx = class_to_index[pred_label]
    #         matrix[true_idx, pred_idx] += 1

    #     return matrix, all_classes
    
    # def classification_report(self):
    #     """Compute classification report from scratch."""
    #     report = {}
    #     for cls in self.classes:
    #         cls_str = str(cls)  # Convert class label to string
    #         true_positive = np.sum((self.y_true == cls) & (self.y_pred == cls))
    #         false_positive = np.sum((self.y_true != cls) & (self.y_pred == cls))
    #         false_negative = np.sum((self.y_true == cls) & (self.y_pred != cls))
    #         support = np.sum(self.y_true == cls)

    #         precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0
    #         recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0
    #         f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    #         report[cls_str] = {  # Use cls_str as the key
    #             "precision": precision,
    #             "recall": recall,
    #             "f1-score": f1,
    #             "support": support
    #         }
        
    #     if self.average == 'macro':
    #         avg_precision = np.mean([report[cls]["precision"] for cls in self.classes])
    #         avg_recall = np.mean([report[cls]["recall"] for cls in self.classes])
    #         avg_f1 = np.mean([report[cls]["f1-score"] for cls in self.classes])
    #         avg_support = np.mean([report[cls]["support"] for cls in self.classes])

    #         report["avg"] = {
    #             "precision": avg_precision,
    #             "recall": avg_recall,
    #             "f1-score": avg_f1,
    #             "support": avg_support
    #         }
    #     return report
    def classification_report(self):
        """Compute classification report from scratch."""
        report = {}
        header = "{:<12} {:<10} {:<10} {:<10} {:<10}".format('Class', 'Precision', 'Recall', 'F1-Score', 'Support')
        report_str = header + "\n"
        
        total_true_positive = 0
        total_support = 0
        weighted_precision_sum = 0
        weighted_recall_sum = 0
        weighted_f1_sum = 0

        for cls in self.classes:
            cls_str = str(cls)  # Convert class label to string
            true_positive = np.sum((self.y_true == cls) & (self.y_pred == cls))
            false_positive = np.sum((self.y_true != cls) & (self.y_pred == cls))
            false_negative = np.sum((self.y_true == cls) & (self.y_pred != cls))
            support = np.sum(self.y_true == cls)

            precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0
            recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

            report[cls_str] = {  # Use cls_str as the key
                "precision": precision,
                "recall": recall,
                "f1-score": f1,
                "support": support
            }

            total_true_positive += true_positive
            total_support += support
            weighted_precision_sum += precision * support
            weighted_recall_sum += recall * support
            weighted_f1_sum += f1 * support

            report_str += "{:<12} {:<10.2f} {:<10.2f} {:<10.2f} {:<10}\n".format(
                cls, precision, recall, f1, support)

        # Calculate overall accuracy
        accuracy = total_true_positive / total_support

        # Calculate macro averages
        avg_precision = np.mean([report[str(cls)]["precision"] for cls in self.classes])
        avg_recall = np.mean([report[str(cls)]["recall"] for cls in self.classes])
        avg_f1 = np.mean([report[str(cls)]["f1-score"] for cls in self.classes])

        # Calculate weighted averages
        weighted_precision = weighted_precision_sum / total_support
        weighted_recall = weighted_recall_sum / total_support
        weighted_f1 = weighted_f1_sum / total_support

        # Add the overall accuracy and averages to the report
        report_str += "\n{:<12} {:<10} {:<10} {:<10} {:<10}".format(
            'accuracy', '', '', f"{accuracy:.2f}", total_support)
        report_str += "\n{:<12} {:<10.2f} {:<10.2f} {:<10.2f} {:<10}".format(
            'macro avg', avg_precision, avg_recall, avg_f1, total_support)
        report_str += "\n{:<12} {:<10.2f} {:<10.2f} {:<10.2f} {:<10}".format(
            'weighted avg', weighted_precision, weighted_recall, weighted_f1, total_support)

        return report_str








class Matrix:
    def __init__(self, y_true, y_pred, average='weighted'):
        self.y_true = np.argmax(y_true, axis=1) if y_true.ndim > 1 else y_true  # one-hot encoding
        self.y_pred = np.argmax(y_pred, axis=1) if y_pred.ndim > 1 else y_pred  # predicted probabilities if necessary
        self.average = average
        self.classes = np.unique(self.y_true)

    def softmax(self, logits):
        """Apply softmax to logits."""
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))  # for numerical stability
        return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

    def acc(self):
        correct = np.sum(self.y_true == self.y_pred)
        total = len(self.y_true)
        return correct / total

    def accuracy_score(self):
        correct = np.sum(self.y_true == self.y_pred)
        total = len(self.y_true)
        return correct / total

    def hamming_loss(self):
        """Hamming loss is the fraction of labels that are incorrectly predicted."""
        incorrect = np.sum(self.y_true != self.y_pred)
        return incorrect / len(self.y_true)

    def precision_score(self):
        unique_classes = np.unique(self.y_true)
        precisions = []

        for cls in unique_classes:
            true_positive = np.sum((self.y_true == cls) & (self.y_pred == cls))
            false_positive = np.sum((self.y_true != cls) & (self.y_pred == cls))
            
            if true_positive + false_positive == 0:
                precision = 0  # division by zero if no positive predictions
            else:
                precision = true_positive / (true_positive + false_positive)
            
            precisions.append(precision)

        if self.average == 'weighted':
            class_counts = [np.sum(self.y_true == cls) for cls in unique_classes]
            weighted_precision = np.average(precisions, weights=class_counts)
            return weighted_precision
        else:  # 'macro' average
            return np.mean(precisions)
        
    def recall_score(self):
        unique_classes = np.unique(self.y_true)
        recalls = []

        for cls in unique_classes:
            true_positive = np.sum((self.y_true == cls) & (self.y_pred == cls))
            false_negative = np.sum((self.y_true == cls) & (self.y_pred != cls))

            if true_positive + false_negative == 0:
                recall = 0  # division by zero if no relevant instances
            else:
                recall = true_positive / (true_positive + false_negative)

            recalls.append(recall)

        if self.average == 'weighted':
            class_counts = [np.sum(self.y_true == cls) for cls in unique_classes]
            weighted_recall = np.average(recalls, weights=class_counts)
            return weighted_recall
        else:  # 'macro' average
            return np.mean(recalls)

    def f1_score(self):
        unique_classes = np.unique(self.y_true)
        f1_scores = []

        for cls in unique_classes:
            true_positive = np.sum((self.y_true == cls) & (self.y_pred == cls))
            false_positive = np.sum((self.y_true != cls) & (self.y_pred == cls))
            false_negative = np.sum((self.y_true == cls) & (self.y_pred != cls))

            if true_positive == 0:
                precision = 0
                recall = 0
            else:
                precision = true_positive / (true_positive + false_positive)
                recall = true_positive / (true_positive + false_negative)

            if precision + recall == 0:
                f1 = 0
            else:
                f1 = 2 * (precision * recall) / (precision + recall)

            f1_scores.append(f1)

        if self.average == 'weighted':
            class_counts = [np.sum(self.y_true == cls) for cls in unique_classes]
            weighted_f1 = np.average(f1_scores, weights=class_counts)
            return weighted_f1
        else:  #'macro' average
            return np.mean(f1_scores)
