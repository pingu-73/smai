import numpy as np
import pandas as pd
import os
import sys
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from models.knn.knn import K_Nearest_Neighbour, Evaluation

DATA = "./data/external/"
df = pd.read_csv(DATA + "spotify.csv")
print(df.info())
print(df.describe())

# -----------------------------------------------------------------------------------------
# 2.3.1
# -----------------------------------------------------------------------------------------

features = ['popularity', 'duration_ms', 'danceability', 'energy', 'key', 'loudness', 'mode', 
                      'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', 'time_signature']

fig, axs = plt.subplots(4, 4, figsize=(20, 20))
axs = axs.ravel()

for i, col in enumerate(features):
    sns.histplot(df[col], kde=True, ax=axs[i])
    axs[i].set_title(f'Distribution of {col}')
    axs[i].set_xlabel(f'{col}')
    axs[i].grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(15, 10))
sns.boxplot(data=df[features])
plt.xticks(rotation=90)
plt.title('Box Plots of Numerical Features')
plt.show()

correlation_matrix = df[features].corr()
plt.figure(figsize=(15, 12))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Heatmap of Numerical Features')
plt.show()

selected_features = ['danceability', 'energy', 'loudness', 'tempo', 'popularity']
sns.pairplot(df[selected_features + ['track_genre']], hue='track_genre')
plt.show()

# -----------------------------------------------------------------------------------------
# 2.2.2
# -----------------------------------------------------------------------------------------


X = df[features]
y = df['track_genre']

def normalize(X):
    return (X - X.min()) / (X.max() - X.min())

X_normalized = normalize(X)

genre_mapping = {genre: i for i, genre in enumerate(y.unique())}
y_encoded = y.map(genre_mapping)

def train_val_test_split(X, y, val_size=0.15, test_size=0.15, random_state=42):
    np.random.seed(random_state)
    indices = np.random.permutation(len(X))
    test_size = int(len(X) * test_size)
    val_size = int(len(X) * val_size)
    test_indices = indices[:test_size]
    val_indices = indices[test_size:test_size+val_size]
    train_indices = indices[test_size+val_size:]
    return (X.iloc[train_indices], X.iloc[val_indices], X.iloc[test_indices],
            y.iloc[train_indices], y.iloc[val_indices], y.iloc[test_indices])

X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(X_normalized, y_encoded)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
save_path = 'data/interim/a1-spotify/'
os.makedirs(save_path, exist_ok=True)

np.save(save_path + 'X_train.npy', X_train)
np.save(save_path + 'X_val.npy', X_val)
np.save(save_path + 'X_test.npy', X_test)
np.save(save_path + 'y_train.npy', y_train)
np.save(save_path + 'y_val.npy', y_val)
np.save(save_path + 'y_test.npy', y_test)

print("1 dn")

X_train = np.load(save_path + 'X_train.npy')
X_val = np.load(save_path + 'X_val.npy')
X_test = np.load(save_path + 'X_test.npy')
y_train = np.load(save_path + 'y_train.npy')
y_val = np.load(save_path + 'y_val.npy')
y_test = np.load(save_path + 'y_test.npy')
# 2. Initialize and train the KNN model
knn = K_Nearest_Neighbour(k=5, distance_formula='euclidean')
train_data = {label: X_train[y_train == label] for label in np.unique(y_train)}
knn.fit(train_data)
print("2: fit dn")
# 3. Make predictions on the validation set
y_pred = np.array([knn.predict(x) for x in X_val])
print("4: predicted")
# 4. Evaluate the model's performance
evaluator = Evaluation()
accuracy = evaluator.accuracy(y_val, y_pred)
precision = evaluator.precision(y_val, y_pred)
recall = evaluator.recall(y_val, y_pred)
f1 = evaluator.f1_score(y_val, y_pred)

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")

# 5. Optional: Hyperparameter tuning
k_values = [1, 3, 5, 7, 9, 11]
distance_metrics = ['euclidean', 'manhatan']

best_f1 = 0
best_k = 0
best_metric = ''

for k in k_values:
    for metric in distance_metrics:
        knn = K_Nearest_Neighbour(k=k, distance_formula=metric)
        knn.fit(train_data)
        y_pred = np.array([knn.predict(x) for x in X_val])
        f1 = evaluator.f1_score(y_val, y_pred)
        
        if f1 > best_f1:
            best_f1 = f1
            best_k = k
            best_metric = metric

print(f"Best k: {best_k}")
print(f"Best distance metric: {best_metric}")
print(f"Best F1 Score: {best_f1:.4f}")

# 6. Final evaluation on test set
knn = K_Nearest_Neighbour(k=best_k, distance_formula=best_metric)
knn.fit(train_data)
y_pred_test = np.array([knn.predict(x) for x in X_test])

accuracy = evaluator.accuracy(y_test, y_pred_test)
precision = evaluator.precision(y_test, y_pred_test)
recall = evaluator.recall(y_test, y_pred_test)
f1 = evaluator.f1_score(y_test, y_pred_test)

print("\nFinal Test Set Performance:")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")

# --------------------------------------------------------------------------------------
# Linear Regression 3.1
# --------------------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from models.linear_regression.linear_regression import LinearRegression

def analyze_regression(file_path,max_k,lambda_r,L1,L2,size_train,size_val,size_test,salt,step_size):
    data = np.loadtxt(file_path, delimiter=',', skiprows=1)

    np.random.shuffle(data)
    train_size = int(size_train * len(data))
    val_size = int(size_test * len(data))
    train_data = data[:train_size]
    val_data = data[train_size:train_size + val_size]
    test_data = data[train_size + val_size:]

    if not os.path.exists('./assignments/1/figures'):
        os.makedirs('./assignments/1/figures')

    best_k = None
    best_mse = float('inf')
    best_beta_vector = None

    for k in range(1, max_k, step_size):
        model = LinearRegression(train_data,salt, k=k, learning_rate=0.1, epochs=100, lambda_reg=lambda_r, L1=L1, L2=L2)
        model.fit()
        mse_train, var_train, std_train = model.predict(train_data)
        mse_test, var_test, std_test = model.predict(test_data)
        print(f"\nFor k={k}: Train MSE: {mse_train}, Variance: {var_train}, Std Dev: {std_train}")
        print(f"For k={k}: Test MSE: {mse_test}, Variance: {var_test}, Std Dev: {std_test}")
        
        if mse_test < best_mse:
            best_mse = mse_test
            best_k = k
            best_beta_vector = model.beta_vector.copy()

    print(f"\nBest k: {best_k} with Test MSE: {best_mse}")
    print("Best model coefficients:", best_beta_vector)

    np.savetxt(f'./assignments/1/{salt}best_model_k{best_k}.txt', best_beta_vector)

# ---------------------------------------------------------------------
# 3.1.1
# ---------------------------------------------------------------------
file_path = './data/external/linreg.csv'
salt = "3.1"
analyze_regression(file_path,10,0,0.1,0.1,0.8,0.1,0.1,salt,2)


# ---------------------------------------------------------------------
# 3.2
# ---------------------------------------------------------------------
file_path = './data/external/regularisation.csv'
salt = "3.2"
analyze_regression(file_path,20,0,0.1,0.1,0.8,0.1,0.1,salt,4)


# ---------------------------------------------------------------------
# 3.2.1 L1 Reg
# ---------------------------------------------------------------------
file_path = './data/external/regularisation.csv'
salt = "3.2.1_L1"
analyze_regression(file_path,20,1,0.1,0.1,0.8,0.1,0.1,salt,4)

# ---------------------------------------------------------------------
# 3.2.1 L2 Reg
# ---------------------------------------------------------------------
file_path = './data/external/regularisation.csv'
salt = "3.2.1_L2"
analyze_regression(file_path,20,2,0.1,0.7,0.8, 0.1, 0.1 ,salt,4)