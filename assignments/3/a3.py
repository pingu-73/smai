import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import warnings, wandb
import sys, os
import seaborn as sns
from itertools import product
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
warnings.filterwarnings('ignore')

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from helper import onehot_encoding, Analysis, Data_preparation, train_and_evaluate, Quest_two_three, Quest_two_three_part_2
from helper import  evaluate_best_model, data_prep, mean_squared_error, r2_score
from models.mlp.mlp_multi import MLP_MULTI
from models.mlp.mlp_regression import MLPRegression, MLP__N
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


## ------- Question: 3
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
df = pd.read_csv("./data/external/HousingData.csv")
print(df.head())
print(df.shape)
print(df.isnull().sum())
## sns.distplot(df.MEDV)
## plt.title('Distribution of MEDV(Prices)')
## plt.show()
## sns.boxplot(df.MEDV)
## plt.title('Box Plot of MEDV(Prices)')
## plt.show()
## print(df.info())
## print("\n")
## print("Checking Coorelation of data")
##correlation = df.corr()
## print(correlation.loc['MEDV'])

## fig,axes = plt.subplots(figsize=(15,12))
## sns.heatmap(correlation,square = True,annot = True)
## plt.title('Heatmap of Correlation')
## plt.show()

## features = ['LSTAT','RM','PTRATIO']
## plt.suptitle("Scatter Plots of the Most Correlated Features with House Prices", fontsize=16)

## for i, col in enumerate(features):
##     plt.subplot(1, len(features), i + 1)
##     x = df[col]
##     y = df['MEDV']
##     plt.scatter(x, y, marker='o')
##     plt.title(f"Variation in House Prices")
##     plt.xlabel(col)
##     plt.ylabel("House Prices in $1000")
##plt.show()




imputer = SimpleImputer(strategy='median')
df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)


df_imputed['CHAS'] = df_imputed['CHAS'].astype(int)  # Convert to integer
df_imputed['RM_squared'] = df_imputed['RM'] ** 2
df_imputed['LSTAT_squared'] = df_imputed['LSTAT'] ** 2
df_imputed['PTRATIO_squared'] = df_imputed['PTRATIO'] ** 2


Y = df_imputed['MEDV'].copy()
X = df_imputed.drop(columns=['MEDV'])


Y_mean, Y_std = Y.mean(), Y.std()
Y = (Y - Y_mean) / Y_std


scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

X_train, X_test, y_train, y_test = train_test_split(X_scaled.values, Y.values.reshape(-1,1), test_size=0.2, random_state=42)

## Initialize and train the MLPRegression model
mlp = MLPRegression(
    hidden_layer_sizes=(32,),
    activation='relu',
    learning_rate=0.01,
    max_iter=1000,
    early_stopping=True,
    validation_split=0.2,
    patience=10
)

## Fit the model
mlp.fit(X_train, y_train)

# Make predictions
y_pred_train = mlp.predict(X_train)
y_pred_test = mlp.predict(X_test)

## Evaluate the model
mse_train = mean_squared_error(y_train, y_pred_train)
mse_test = mean_squared_error(y_test, y_pred_test)
r2_train = r2_score(y_train, y_pred_train)
r2_test = r2_score(y_test, y_pred_test)
rms_train = np.sqrt(mse_train)
rms_test = np.sqrt(mse_test)

print(f"Train MSE: {mse_train:.4f}")
print(f"Test MSE: {mse_test:.4f}")
print(f"Train R_squared: {r2_train:.4f}")
print(f"Test R_squared: {r2_test:.4f}")
print(f"Train RMSE: {rms_train:.4f}")
print(f"Test RMSE: {rms_test:.4f}")




########   Question 3.3
# hyperparameter_space = {
#     'hidden_layer_sizes': [(32,), (64,), (32, 16), (64, 32), (128, 64, 32)],
#     'activation': ['relu', 'tanh', 'sigmoid'],
#     'learning_rate': np.linspace(0.0001, 0.1, num=5),  # 5 values between 0.0001 and 0.1
#     'max_iter': 1000,
#     'early_stopping': True,
#     'validation_split': 0.2,
#     'patience': 20,
#     'alpha': np.linspace(0.0001, 0.01, num=5),  # 5 values between 0.0001 and 0.01
#     'batch_size': [16, 32, 64]
# }

# Initialize W&B
# wandb.init(project="mlp_regression_hyperparameter_tuning")


# df = pd.read_csv("./data/external/HousingData.csv")


# imputer = SimpleImputer(strategy='median')
# df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)


# df_imputed['CHAS'] = df_imputed['CHAS'].astype(int)
# df_imputed['RM_squared'] = df_imputed['RM'] ** 2
# df_imputed['LSTAT_squared'] = df_imputed['LSTAT'] ** 2
# df_imputed['PTRATIO_squared'] = df_imputed['PTRATIO'] ** 2


# Y = df_imputed['MEDV'].copy()
# X = df_imputed.drop(columns=['MEDV'])


# Y_mean, Y_std = Y.mean(), Y.std()
# Y = (Y - Y_mean) / Y_std
# scaler = StandardScaler()
# X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)


# X_train, X_test, y_train, y_test = train_test_split(X_scaled.values, Y.values.reshape(-1, 1), test_size=0.2, random_state=42)

# def train_and_log_model(params):
#     wandb.config.update(params, allow_val_change=True)
    
#     mlp = MLPRegression(
#         hidden_layer_sizes=params['hidden_layer_sizes'],
#         activation=params['activation'],
#         learning_rate=params['learning_rate'],
#         max_iter=params['max_iter'],
#         early_stopping=params['early_stopping'],
#         validation_split=params['validation_split'],
#         patience=params['patience'],
#         alpha=params['alpha'],
#         batch_size=params['batch_size']
#     )

#     mlp.fit(X_train, y_train)

#     y_pred_train = mlp.predict(X_train)
#     y_pred_test = mlp.predict(X_test)

#     mse_train = mean_squared_error(y_train, y_pred_train)
#     mse_test = mean_squared_error(y_test, y_pred_test)
#     r2_train = r2_score(y_train, y_pred_train)
#     r2_test = r2_score(y_test, y_pred_test)
#     rms_train = np.sqrt(mse_train)
#     rms_test = np.sqrt(mse_test)

#     wandb.log({
#         "Train MSE": mse_train,
#         "Test MSE": mse_test,
#         "Train RMSE": rms_train,
#         "Test RMSE": rms_test,
#         "Train R_squared": r2_train,
#         "Test R_squared": r2_test,
#         "Parameters": params
#     })

# for hidden_layer_sizes in hyperparameter_space['hidden_layer_sizes']:
#     for activation in hyperparameter_space['activation']:
#         for learning_rate in hyperparameter_space['learning_rate']:
#             for alpha in hyperparameter_space['alpha']:
#                 for batch_size in hyperparameter_space['batch_size']:
#                     params = {
#                         'hidden_layer_sizes': hidden_layer_sizes,
#                         'activation': activation,
#                         'learning_rate': learning_rate,
#                         'max_iter': hyperparameter_space['max_iter'],
#                         'early_stopping': hyperparameter_space['early_stopping'],
#                         'validation_split': hyperparameter_space['validation_split'],
#                         'patience': hyperparameter_space['patience'],
#                         'alpha': alpha,
#                         'batch_size': batch_size
#                     }
#                     train_and_log_model(params)

# wandb.finish()

#############


############## Question 3.4
# from sklearn.metrics import mean_absolute_error
# from sklearn.impute import SimpleImputer

# hyperparameter_space = {
#     'hidden_layer_sizes': {
#         'values': [(50,), (100,), (50, 50), (100, 50)]  # Example hidden layer sizes
#     },
#     'activation': {
#         'values': ['relu', 'tanh']  # Example activation functions
#     },
#     'learning_rate': {
#         'min': 0.0001,
#         'max': 0.1
#     },
#     'alpha': {
#         'min': 0.0001,
#         'max': 0.1
#     },
#     'batch_size': {
#         'values': [16, 32, 64]  # Example batch sizes
#     },
#     'max_iter': {
#         'value': 500  # Example maximum iterations
#     },
#     'early_stopping': {
#         'value': True  # Example for early stopping
#     },
#     'validation_split': {
#         'value': 0.2  # Example for validation split
#     },
#     'patience': {
#         'value': 10  # Example patience for early stopping
#     }
# }

# wandb.init(project="Question-3")

# df = pd.read_csv("./data/external/HousingData.csv")

# imputer = SimpleImputer(strategy='median')
# df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

# df_imputed['CHAS'] = df_imputed['CHAS'].astype(int)
# df_imputed['RM_squared'] = df_imputed['RM'] ** 2
# df_imputed['LSTAT_squared'] = df_imputed['LSTAT'] ** 2
# df_imputed['PTRATIO_squared'] = df_imputed['PTRATIO'] ** 2

# Y = df_imputed['MEDV'].copy()
# X = df_imputed.drop(columns=['MEDV'])

# Y_mean, Y_std = Y.mean(), Y.std()
# Y = (Y - Y_mean) / Y_std
# scaler = StandardScaler()
# X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

# X_train, X_test, y_train, y_test = train_test_split(X_scaled.values, Y.values.reshape(-1, 1), test_size=0.2, random_state=42)

# results = []

# def train_and_log_model(params):
#     wandb.config.update(params, allow_val_change=True)
    
#     mlp = MLPRegression(
#         hidden_layer_sizes=params['hidden_layer_sizes'],
#         activation=params['activation'],
#         learning_rate=params['learning_rate'],
#         max_iter=params['max_iter'],
#         early_stopping=params['early_stopping'],
#         validation_split=params['validation_split'],
#         patience=params['patience'],
#         alpha=params['alpha'],
#         batch_size=params['batch_size']
#     )

#     mlp.fit(X_train, y_train)

#     y_pred_train = mlp.predict(X_train)
#     y_pred_test = mlp.predict(X_test)

#     mse_train = mean_squared_error(y_train, y_pred_train)
#     mse_test = mean_squared_error(y_test, y_pred_test)
#     mae_test = mean_absolute_error(y_test, y_pred_test)
#     r2_train = r2_score(y_train, y_pred_train)
#     r2_test = r2_score(y_test, y_pred_test)
#     rms_train = np.sqrt(mse_train)
#     rms_test = np.sqrt(mse_test)

#     wandb.log({
#         "Train MSE": mse_train,
#         "Test MSE": mse_test,
#         "Train RMSE": rms_train,
#         "Test RMSE": rms_test,
#         "Train MAE": mean_absolute_error(y_train, y_pred_train),
#         "Test MAE": mae_test,
#         "Train R_squared": r2_train,
#         "Test R_squared": r2_test,
#         "Parameters": params
#     })

#     results.append({
#         "params": params,
#         "test_mse": mse_test,
#         "test_mae": mae_test
#     })

# for hidden_layer_sizes in hyperparameter_space['hidden_layer_sizes']['values']:
#     for activation in hyperparameter_space['activation']['values']:
#         learning_rate = np.random.uniform(hyperparameter_space['learning_rate']['min'], hyperparameter_space['learning_rate']['max'])
#         alpha = np.random.uniform(hyperparameter_space['alpha']['min'], hyperparameter_space['alpha']['max'])
#         for batch_size in hyperparameter_space['batch_size']['values']:
#             params = {
#                 'hidden_layer_sizes': hidden_layer_sizes,
#                 'activation': activation,
#                 'learning_rate': learning_rate,
#                 'max_iter': hyperparameter_space['max_iter']['value'],
#                 'early_stopping': hyperparameter_space['early_stopping']['value'],
#                 'validation_split': hyperparameter_space['validation_split']['value'],
#                 'patience': hyperparameter_space['patience']['value'],
#                 'alpha': alpha,
#                 'batch_size': batch_size
#             }
#             train_and_log_model(params)

# best_model = min(results, key=lambda x: x["test_mse"])

# print("Best Model Parameters:", best_model["params"])
# print("Best Model Test MSE:", best_model["test_mse"])
# print("Best Model Test MAE:", best_model["test_mae"])

# wandb.finish()
##############








#################################### Question 3.7
df = pd.read_csv('./data/external/diabetes.csv')

X = df.drop(columns=['Outcome'])
y = df['Outcome']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


model_bce = MLP__N(input_size=X_train_scaled.shape[1], loss_function='BCE', epochs=1000)
model_mse = MLP__N(input_size=X_train_scaled.shape[1], loss_function='MSE', epochs=1000)

model_bce.train(X_train_scaled, y_train)
model_mse.train(X_train_scaled, y_train)

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(model_bce.losses, label='BCE Loss', color='blue')
plt.title('Loss vs Epochs (BCE)')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(model_mse.losses, label='MSE Loss', color='red')
plt.title('Loss vs Epochs (MSE)')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()

y_pred_bce = model_bce.predict(X_test_scaled)
y_pred_mse = model_mse.predict(X_test_scaled)

y_pred_bce_binary = (y_pred_bce > 0.5).astype(int)
y_pred_mse_binary = (y_pred_mse > 0.5).astype(int)

accuracy_bce = np.mean(y_pred_bce_binary.flatten() == y_test)
accuracy_mse = np.mean(y_pred_mse_binary.flatten() == y_test)

print(f'Accuracy of BCE model: {accuracy_bce:.4f}')
print(f'Accuracy of MSE model: {accuracy_mse:.4f}')
#####################################