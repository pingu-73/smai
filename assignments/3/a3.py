import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import warnings
import sys, os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import accuracy_score, f1_score, classification_report
# from sklearn.neural_network import MLPClassifier  # Example model
warnings.filterwarnings('ignore')

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from helper import onehot_encoding, Analysis, Data_preparation, train_and_evaluate, Quest_two_three, Quest_two_three_part_2
from helper import  evaluate_best_model, data_prep
from models.mlp.mlp_multi import MLP_MULTI
from models.mlp.mlp_regression import MLPRegressor
from performance_measures.MLPPerformance import Matrix

### """ Question 2: MLP Classifier (2.1)"""

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



# ### Multi-label MLP ------- 2.6
df = pd.read_csv("./data/external/advertisement.csv")
X_train, X_test, y_train, y_test = data_prep(df)

X_train = np.asarray(X_train, dtype=float)
y_train = np.asarray(y_train, dtype=float)
X_test = np.asarray(X_test, dtype=float)
y_test = np.asarray(y_test, dtype=float)

input_size = X_train.shape[1]
hidden_size = 64 
output_size = y_train.shape[1]

model = MLP_MULTI(input_size, hidden_size, output_size, lr=0.01, activation='relu')
model.fit(X_train, y_train, epochs=100, method='mini_batch', batch_size=32)

y_pred = model.predict(X_test) 

accuracy = model.accuracy(X_test, y_test)
perf = Matrix(y_test, y_pred) 
hamming = perf.hamming_loss()
precision = perf.precision_score()
recall = perf.recall_score()
f1 = perf.f1_score()

print(f"Accuracy: {accuracy:.2f}")
print(f"Hamming Loss: {hamming:.2f}")
print(f"Precision: {precision:.2f}") 
print(f"Recall: {recall:.2f}")
print(f"F-1 score: {f1:.2f}") 


### ------- Question: 3
df = pd.read_csv("./data/external/HousingData.csv")

print(df.describe())
print(df.isnull().sum())
df = df.dropna()
# print(df.isnull().sum())

plt.figure(figsize=(10, 6))
sns.histplot(df['MEDV'], bins=30, kde=True)
plt.title('Distribution of MEDV')
plt.xlabel('MEDV')
plt.ylabel('Frequency')
plt.show()

# Partition the dataset
X = df.drop('MEDV', axis=1)
y = df['MEDV']
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Normalize and standardize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)


input_size = X_train.shape[1]  # Number of features
hidden_layers = [10]  # You can adjust this
output_size = 1  # For regression, output size is 1
learning_rate = 0.01
# activation_function = 'relu'  # Choose activation function

mlp = MLPRegressor(input_size, hidden_layers, output_size, lr=learning_rate)#, activation=activation_function)
mlp.fit(X_train, y_train, epochs=1000, method='batch')
predictions = mlp.predict(X_val)
mse = np.mean((predictions - y_val.to_numpy().reshape(-1, 1)) ** 2)
rmse = np.sqrt(mse)
ss_res = np.sum((y_val.values - predictions) ** 2)
ss_tot = np.sum((y_val.values - np.mean(y_val.values)) ** 2)
r_squared = 1 - (ss_res / ss_tot)

print(f'Mean Squared Error(Validation): {mse}')
print(f"Root Mean Squared Error(Validation): {rmse}")
print(f"R-squared(Validation): {r_squared}")
