import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import warnings
import sys, os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, f1_score, classification_report
# from sklearn.neural_network import MLPClassifier  # Example model
warnings.filterwarnings('ignore')

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from helper import onehot_encoding, Analysis, Data_preparation, train_and_evaluate, Quest_two_three, Quest_two_three_part_2
from helper import evaluate_model, evaluate_best_model, data_prep
from models.mlp.mlp_multi import MLP_MULTI

file_path = "./data/external/WineQT.csv"
df = pd.read_csv(file_path)
df = Analysis(df)
X, y, X_train, X_test, y_train, y_test = Data_preparation(df)
input_size = X.shape[1]
output_size = y_train.shape[1]
train_and_evaluate(X_train, y_train, X_test, y_test, input_size, output_size)
# Quest_two_three(X_train, y_train, X_test, y_test, input_size, output_size)
# Quest_two_three_part_2(X_train, y_train, X_test, y_test, input_size, output_size)
best_model_path = "./data/interim/best_mlp_model.pkl"
evaluate_best_model(X_test, y_test,file_path=best_model_path)



### Multi-label MLP
df = pd.read_csv("./data/external/advertisement.csv")
X_train, X_test, y_train, y_test = data_prep(df)
X_train = np.asarray(X_train, dtype=float)
y_train = np.asarray(y_train, dtype=float)
input_size = X_train.shape[1]
hidden_size = 64  # You can adjust this
output_size = y_train.shape[1]
model = MLP_MULTI(input_size, hidden_size, output_size, lr=0.01, activation='relu')
model.fit(X_train, y_train, epochs=100, method='mini_batch', batch_size=32)
# Evaluate the model
accuracy, precision, recall, f1, hamming = evaluate_model(model, X_test, y_test)
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-score: {f1:.4f}")
print(f"Hamming loss: {hamming:.4f}")