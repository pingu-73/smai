import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import warnings, sys, os
import wandb
import joblib
from itertools import product
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
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


def Quest_two_three(X_train, y_train, X_test, y_test, input_size, output_size):
    # Define the hyperparameter space
    epochs_options = [50, 100, 150]
    lr_options = [0.001, 0.01, 0.1]
    hidden_size_options = [50, 100, 150]
    activation_options = ['sigmoid', 'relu']

    # Store the best performance
    best_accuracy = 0
    best_params = {}
    best_model = None  # Placeholder for the best model

    for epochs, lr, hidden_size, activation in product(epochs_options, lr_options, hidden_size_options, activation_options):
        # Initialize W&B for each set of hyperparameters
        wandb.init(project="Question-two-three", config={
            "epochs": epochs,
            "learning_rate": lr,
            "hidden_size": hidden_size,
            "activation": activation
        })

        model = MLP(input_size=input_size, hidden_size=hidden_size, output_size=output_size, lr=lr, activation=activation)

        # Store loss and accuracy for plotting later
        train_loss = []
        val_loss = []
        train_accuracy = []
        val_accuracy = []

        for epoch in range(epochs):
            model.fit(X_train, y_train, epochs=1)  # Train for one epoch

            # Calculate train loss and accuracy
            train_loss.append(model.loss(X_train, y_train))
            y_train_pred = model.predict(X_train)
            y_train_labels = np.argmax(y_train, axis=1)
            y_train_pred_labels = np.argmax(y_train_pred, axis=1)
            train_accuracy.append(np.mean(y_train_labels == y_train_pred_labels))

            # Calculate validation loss and accuracy
            val_loss_epoch = model.loss(X_test, y_test)
            val_loss.append(val_loss_epoch)
            y_val_pred = model.predict(X_test)
            y_val_labels = np.argmax(y_test, axis=1)
            y_val_pred_labels = np.argmax(y_val_pred, axis=1)
            val_accuracy.append(np.mean(y_val_labels == y_val_pred_labels))

            # Log metrics
            wandb.log({
                "train_loss": train_loss[-1],
                "val_loss": val_loss[-1],
                "train_accuracy": train_accuracy[-1],
                "val_accuracy": val_accuracy[-1]
            })

        # Final evaluation
        y_test_pred = model.predict(X_test)
        y_test_labels = np.argmax(y_test, axis=1)
        y_pred_labels = np.argmax(y_test_pred, axis=1)

        performance = PerformanceMatrix(y_test_labels, y_pred_labels)
        accuracy = performance.accuracy_score()

        # Update best parameters if current model is better
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_params = {
                "epochs": epochs,
                "learning_rate": lr,
                "hidden_size": hidden_size,
                "activation": activation
            }
            best_model = model  # Save the best model

        # Finish the W&B run for the current hyperparameter set
        wandb.finish()

    # Save the best model to a file
    if best_model is not None:
        joblib.dump(best_model, "best_mlp_model.pkl")

    print(f"Best Accuracy: {best_accuracy} with parameters: {best_params}")

def Quest_two_three_part_2(X_train, y_train, X_test, y_test, input_size, output_size):
    # Define the hyperparameter space
    epochs_options = [50, 100, 150]
    lr_options = [0.001, 0.01, 0.1]
    hidden_size_options = [50, 100, 150]
    activation_options = ['sigmoid', 'relu']

    # Initialize W&B project
    wandb.init(project="New_Q2.3")
    
    # Store the best performance
    best_accuracy = 0
    best_params = {}
    best_model = None  # Placeholder for the best model

    # Table to store hyperparameters and metrics
    hyperparams_table = wandb.Table(columns=["epochs", "learning_rate", "hidden_size", "activation", 
                                              "test_accuracy", "test_precision", "test_recall", "test_f1_score"])

    for epochs, lr, hidden_size, activation in product(epochs_options, lr_options, hidden_size_options, activation_options):
        # Initialize W&B for each set of hyperparameters
        wandb.config.update({
            "epochs": epochs,
            "learning_rate": lr,
            "hidden_size": hidden_size,
            "activation": activation
        }, allow_val_change=True)

        model = MLP(input_size=input_size, hidden_size=hidden_size, output_size=output_size, lr=lr, activation=activation)

        # Store loss and accuracy for plotting later
        train_loss = []
        val_loss = []
        train_accuracy = []
        val_accuracy = []

        for epoch in range(epochs):
            model.fit(X_train, y_train, epochs=1)  # Train for one epoch

            # Calculate train loss and accuracy
            train_loss.append(model.loss(X_train, y_train))
            y_train_pred = model.predict(X_train)
            y_train_labels = np.argmax(y_train, axis=1)
            y_train_pred_labels = np.argmax(y_train_pred, axis=1)
            train_accuracy.append(np.mean(y_train_labels == y_train_pred_labels))

            # Calculate validation loss and accuracy
            val_loss_epoch = model.loss(X_test, y_test)
            val_loss.append(val_loss_epoch)
            y_val_pred = model.predict(X_test)
            y_val_labels = np.argmax(y_test, axis=1)
            y_val_pred_labels = np.argmax(y_val_pred, axis=1)
            val_accuracy.append(np.mean(y_val_labels == y_val_pred_labels))

            # Log metrics
            wandb.log({
                "train_loss": train_loss[-1],
                "val_loss": val_loss[-1],
                "train_accuracy": train_accuracy[-1],
                "val_accuracy": val_accuracy[-1]
            })

        # Final evaluation
        y_test_pred = model.predict(X_test)
        y_test_labels = np.argmax(y_test, axis=1)
        y_pred_labels = np.argmax(y_test_pred, axis=1)

        # Calculate performance metrics
        accuracy = accuracy_score(y_test_labels, y_pred_labels)
        precision = precision_score(y_test_labels, y_pred_labels, average='weighted', zero_division=0)
        recall = recall_score(y_test_labels, y_pred_labels, average='weighted', zero_division=0)
        f1 = f1_score(y_test_labels, y_pred_labels, average='weighted', zero_division=0)

        # Log performance metrics to W&B
        wandb.log({
            "test_accuracy": accuracy,
            "test_precision": precision,
            "test_recall": recall,
            "test_f1_score": f1
        })

        # Add the current hyperparameters and metrics to the table
        hyperparams_table.add_data(epochs, lr, hidden_size, activation, accuracy, precision, recall, f1)

        # Update best parameters if current model is better
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_params = {
                "epochs": epochs,
                "learning_rate": lr,
                "hidden_size": hidden_size,
                "activation": activation
            }
            best_model = model  

    wandb.log({"hyperparameters_table": hyperparams_table})

    # Save the best model to a file
    if best_model is not None:
        joblib.dump(best_model, "mlp_model.pkl")  # Save with joblib

    print(f"Best Accuracy: {best_accuracy} with parameters: {best_params}")

    # Finish the W&B run
    wandb.finish()


def evaluate_best_model(X_test, y_test, file_path):
    # Load the best model
    best_model = joblib.load(file_path)

    # Make predictions
    y_pred = best_model.predict(X_test)
    y_test_labels = np.argmax(y_test, axis=1)
    y_pred_labels = np.argmax(y_pred, axis=1)

    # Calculate performance metrics
    performance = PerformanceMatrix(y_test_labels, y_pred_labels)
    accuracy = performance.accuracy_score()
    precision = performance.precision_score()
    recall = performance.recall_score()
    f1 = performance.f1_score()
    report = performance.classification_report()
    matrix = performance.confusion_matrix()

    # Print results
    print("Best Model Performance:")
    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1}")
    print("\nClassification Report:")
    print(report)
    print("\nConfusion Matrix:")
    print(matrix)