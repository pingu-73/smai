import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import warnings, sys, os
from sklearn.preprocessing import MinMaxScaler
warnings.filterwarnings('ignore')

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from models.mlp.mlp import MLP
from performance_measures.MLPPerformance import PerformanceMatrix

def onehot_encoding(y):
    num_classes = y.max() + 1
    encoded = np.zeros((len(y), num_classes))
    encoded[np.arange(len(y)), y.values.flatten()] = 1
    return encoded

def Analysis(df):
    print(df.head())
    print("==============================")
    print(f"df shape: {df.shape}")
    print("==============================")
    df = df.drop_duplicates()
    print(f"shape after dropping duplicates: {df.shape}")
    print("==============================")
    print(f"missing values:")
    print(df.isna().sum())
    print("==============================")
    print(f"stats of df")
    print(df.describe().T)
    print("==============================")
    print(f"unique values in quality: {df['quality'].unique()}")
    print("==============================")
    df['quality'] = df['quality'] - 2  # data normalization
    df['quality'].unique()
    print("==============================")
    df = df.drop(columns=['Id'])
    print(f"df after dropping ID's col: \n {df.head()}")
    print("==============================")

    df.hist(bins=20, figsize=(12, 10))
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 8))
    correlation_matrix = df.corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Feature Correlation Heatmap')
    plt.show()

    quality_counts = df['quality'].value_counts()
    sns.barplot(x=quality_counts.index, y=quality_counts.values, palette='coolwarm')
    plt.title('Class Distribution of Wine Quality')
    plt.xlabel('Wine Quality')
    plt.ylabel('Count')
    plt.show()

    return df

def Data_preparation(df):
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(X)
    X = pd.DataFrame(scaled_data, columns=X.columns)

    print(y.max())
    print(f"unique y: {len(np.unique(y))}")
    print("==============================")

    y_encoded = onehot_encoding(y)

    indices = np.arange(len(df))
    np.random.shuffle(indices)
    X_shuffled = X.iloc[indices]
    y_shuffled = y_encoded[indices]

    split_ratio = 0.8 
    split_index = int(len(df) * split_ratio)

    X_train, X_test = X_shuffled[:split_index], X_shuffled[split_index:]
    y_train, y_test = y_shuffled[:split_index], y_shuffled[split_index:]
    return X, y, X_train, X_test, y_train, y_test

def train_and_evaluate(X_train, y_train, X_test, y_test, input_size, output_size):
    model = MLP(input_size=input_size, hidden_size=100, output_size=output_size)
    model.fit(X_train, y_train, epochs=100)

    y_pred = model.predict(X_test)
    y_test_labels = np.argmax(y_test, axis=1)
    y_pred_labels = np.argmax(y_pred, axis=1)

    performance = PerformanceMatrix(y_test_labels, y_pred_labels)
    accuracy = performance.accuracy_score()
    f1 = performance.f1_score()
    report = performance.classification_report()
    matrix = performance.confusion_matrix()

    print(f"Accuracy: {accuracy}")
    print(f"F1 Score: {f1}")
    print("Classification Report:")
    print(report)
    print(f"Confusion Matrix:\n {matrix}")