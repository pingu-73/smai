import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import warnings
import sys, os
warnings.filterwarnings('ignore')

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from models.mlp.mlp import MLP
from helper import onehot_encoding, Analysis, Data_preparation, train_and_evaluate



#main.py

# ### ================================================================================================
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
from helper import onehot_encoding, Analysis, Data_preparation, train_and_evaluate

file_path = "./data/external/WineQT.csv"
df = pd.read_csv(file_path)
df = Analysis(df)
X, y, X_train, X_test, y_train, y_test = Data_preparation(df)
input_size = X.shape[1]
output_size = y_train.shape[1]
train_and_evaluate(X_train, y_train, X_test, y_test, input_size, output_size)