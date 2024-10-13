<center>

# **Assignment 3 Report**

</center>

## 2 Multi Layer Perceptron Classification

### **2.1**:
```
       fixed acidity  volatile acidity  citric acid  ...      alcohol      quality           Id
count    1143.000000       1143.000000  1143.000000  ...  1143.000000  1143.000000  1143.000000
mean        8.311111          0.531339     0.268364  ...    10.442111     5.657043   804.969379
std         1.747595          0.179633     0.196686  ...     1.082196     0.805824   463.997116
min         4.600000          0.120000     0.000000  ...     8.400000     3.000000     0.000000
25%         7.100000          0.392500     0.090000  ...     9.500000     5.000000   411.000000
50%         7.900000          0.520000     0.250000  ...    10.200000     6.000000   794.000000
75%         9.100000          0.640000     0.420000  ...    11.100000     6.000000  1209.500000
max        15.900000          1.580000     1.000000  ...    14.900000     8.000000  1597.000000

[8 rows x 13 columns]
Means:
 fixed acidity             8.311111
volatile acidity          0.531339
citric acid               0.268364
residual sugar            2.532152
chlorides                 0.086933
free sulfur dioxide      15.615486
total sulfur dioxide     45.914698
density                   0.996730
pH                        3.311015
sulphates                 0.657708
alcohol                  10.442111
quality                   5.657043
Id                      804.969379
dtype: float64
Standard Deviations:
 fixed acidity             1.747595
volatile acidity          0.179633
citric acid               0.196686
residual sugar            1.355917
chlorides                 0.047267
free sulfur dioxide      10.250486
total sulfur dioxide     32.782130
density                   0.001925
pH                        0.156664
sulphates                 0.170399
alcohol                   1.082196
quality                   0.805824
Id                      463.997116
dtype: float64
Min values:
 fixed acidity           4.60000
volatile acidity        0.12000
citric acid             0.00000
residual sugar          0.90000
chlorides               0.01200
free sulfur dioxide     1.00000
total sulfur dioxide    6.00000
density                 0.99007
pH                      2.74000
sulphates               0.33000
alcohol                 8.40000
quality                 3.00000
Id                      0.00000
dtype: float64
Max values:
 fixed acidity             15.90000
volatile acidity           1.58000
citric acid                1.00000
residual sugar            15.50000
chlorides                  0.61100
free sulfur dioxide       68.00000
total sulfur dioxide     289.00000
density                    1.00369
pH                         4.01000
sulphates                  2.00000
alcohol                   14.90000
quality                    8.00000
Id                      1597.00000
dtype: float64
```
### Overview of Dataset:
The dataset used in this project is WineQT.csv, which contains the following features:
```
fixed acidity
volatile acidity
citric acid
residual sugar
chlorides
free sulfur dioxide
total sulfur dioxide
density
pH
sulphates
alcohol
quality
Id
```
The target variable is `quality`, which is converted into categorical labels `(bad, average, good)` for the purpose of classification.

### **2.2**:
<center>

<img src="../3/figures/WineQT_analysis.png" alt="histogram" width = 400>

<img src="../3/figures/WineQT_heatmap.png" alt="heatmap" width = 400>

<img src="../3/figures/WineQT_Quality.png" alt="Quality" width = 400>

</center>

## 2.2 Building MLP
```
Accuracy: 0.5327510917030568
F1 Score: 0.560145925631015
Classification Report:
              precision    recall  f1-score   support

           1       0.00      0.00      0.00         0
           2       0.00      0.00      0.00         7
           3       0.74      0.72      0.73       100
           4       0.58      0.52      0.55        86
           5       0.56      0.17      0.26        30
           6       0.00      0.00      0.00         6

    accuracy                           0.53       229
   macro avg       0.31      0.23      0.26       229
weighted avg       0.62      0.53      0.56       229

Numeric: 0.01281646502038214, Analytic: 0.019674001244321025

Gradient check passed: False
```
The Gradient check is failing during this testing it's becasue the gradient computed numeracilly do not match with gradient computed analytically during backpropagation. I suspect the reason for gradient failure in this step is that activation functions are facing an exploding gradient issue.

## 2.3 Hyperparameter Tuning
Best Model's Details:
```
Best Accuracy: 0.6244541484716157 with parameters: {'epochs': 150, 'learning_rate': 0.1, 'hidden_size': 100, 'activation': 'sigmoid'}
```
<center>

![best model](../3/figures/2.3_mlp_best_model.png)
</center>

## 2.4 Best Model Analysis
```
Best Model Performance:
Accuracy: 0.6593886462882096
Precision: 0.6062367423687839
Recall: 0.6593886462882096
F1 Score: 0.6271040448254205

Classification Report:
Class        Precision  Recall     F1-Score   Support
1            0.00       0.00       0.00       2
2            0.00       0.00       0.00       9
3            0.72       0.90       0.80       107
4            0.64       0.56       0.60       82
5            0.39       0.36       0.37       25
6            0.00       0.00       0.00       4

accuracy                           0.66       229
macro avg    0.29       0.30       0.29       229
weighted avg 0.61       0.66       0.63       229

Confusion Matrix:
[[ 0  0  2  0  0  0]
 [ 0  0  7  2  0  0]
 [ 0  0 96  9  2  0]
 [ 0  0 27 46  9  0]
 [ 0  0  2 14  9  0]
 [ 0  0  0  1  3  0]]
```


### 2.6 Multi Label MLP
```
Accuracy: 0.34
Hamming Loss: 0.66
Precision: 0.12
Recall: 0.34
F-1 score: 0.17
```



## 4 AutoEncoders

### 4.2 Train the autoencoder
Loss during Training
```
Epoch 1/100, Loss: 0.8144798690888169
Epoch 2/100, Loss: 0.807900434924758
Epoch 3/100, Loss: 0.8015810317329163
Epoch 4/100, Loss: 0.7955113564487419
Epoch 5/100, Loss: 0.7896815142418266
Epoch 6/100, Loss: 0.7840820023406087
Epoch 7/100, Loss: 0.7787036944979837
Epoch 8/100, Loss: 0.7735378260724286
Epoch 9/100, Loss: 0.7685759797002518
Epoch 10/100, Loss: 0.7638100715355435
Epoch 11/100, Loss: 0.7592323380353359
Epoch 12/100, Loss: 0.7548353232683708
Epoch 13/100, Loss: 0.7506118667267245
Epoch 14/100, Loss: 0.74655509162037
Epoch 15/100, Loss: 0.7426583936355379
Epoch 16/100, Loss: 0.7389154301384984
Epoch 17/100, Loss: 0.7353201098071223
Epoch 18/100, Loss: 0.7318665826732607
Epoch 19/100, Loss: 0.7285492305596768
Epoch 20/100, Loss: 0.7253626578958858
Epoch 21/100, Loss: 0.7223016828979009
Epoch 22/100, Loss: 0.7193613290974551
Epoch 23/100, Loss: 0.7165368172068617
Epoch 24/100, Loss: 0.713823557306207
Epoch 25/100, Loss: 0.7112171413401086
Epoch 26/100, Loss: 0.7087133359117704
Epoch 27/100, Loss: 0.7063080753625569
Epoch 28/100, Loss: 0.7039974551257715
Epoch 29/100, Loss: 0.7017777253437765
Epoch 30/100, Loss: 0.6996452847380187
Epoch 31/100, Loss: 0.6975966747219406
Epoch 32/100, Loss: 0.6956285737471538
Epoch 33/100, Loss: 0.6937377918736302
Epoch 34/100, Loss: 0.6919212655550386
Epoch 35/100, Loss: 0.6901760526306977
Epoch 36/100, Loss: 0.6884993275159624
Epoch 37/100, Loss: 0.6868883765831781
Epoch 38/100, Loss: 0.6853405937256553
Epoch 39/100, Loss: 0.6838534760974088
Epoch 40/100, Loss: 0.682424620021702
Epoch 41/100, Loss: 0.6810517170617032
Epoch 42/100, Loss: 0.6797325502468338
Epoch 43/100, Loss: 0.6784649904486377
Epoch 44/100, Loss: 0.6772469929002497
Epoch 45/100, Loss: 0.6760765938537686
Epoch 46/100, Loss: 0.6749519073700775
Epoch 47/100, Loss: 0.6738711222358564
Epoch 48/100, Loss: 0.6728324990027537
Epoch 49/100, Loss: 0.6718343671438709
Epoch 50/100, Loss: 0.670875122322916
Epoch 51/100, Loss: 0.6699532237715607
Epoch 52/100, Loss: 0.6690671917707119
Epoch 53/100, Loss: 0.6682156052315835
Epoch 54/100, Loss: 0.6673970993726103
Epoch 55/100, Loss: 0.6666103634884111
Epoch 56/100, Loss: 0.6658541388071494
Epoch 57/100, Loss: 0.6651272164327919
Epoch 58/100, Loss: 0.6644284353688995
Epoch 59/100, Loss: 0.6637566806207212
Epoch 60/100, Loss: 0.6631108813724883
Epoch 61/100, Loss: 0.6624900092369277
Epoch 62/100, Loss: 0.6618930765741357
Epoch 63/100, Loss: 0.661319134877061
Epoch 64/100, Loss: 0.6607672732209596
Epoch 65/100, Loss: 0.6602366167742869
Epoch 66/100, Loss: 0.6597263253685904
Epoch 67/100, Loss: 0.6592355921250693
Epoch 68/100, Loss: 0.6587636421355508
Epoch 69/100, Loss: 0.6583097311957327
Epoch 70/100, Loss: 0.6578731445886143
Epoch 71/100, Loss: 0.6574531959161346
Epoch 72/100, Loss: 0.6570492259771018
Epoch 73/100, Loss: 0.6566606016895828
Epoch 74/100, Loss: 0.6562867150559907
Epoch 75/100, Loss: 0.6559269821691776
Epoch 76/100, Loss: 0.6555808422579091
Epoch 77/100, Loss: 0.6552477567701595
Epoch 78/100, Loss: 0.6549272084927287
Epoch 79/100, Loss: 0.654618700705743
Epoch 80/100, Loss: 0.6543217563706559
Epoch 81/100, Loss: 0.654035917350424
Epoch 82/100, Loss: 0.6537607436605786
Epoch 83/100, Loss: 0.653495812749975
Epoch 84/100, Loss: 0.6532407188100388
Epoch 85/100, Loss: 0.6529950721113807
Epoch 86/100, Loss: 0.6527584983666995
Epoch 87/100, Loss: 0.6525306381189248
Epoch 88/100, Loss: 0.652311146153605
Epoch 89/100, Loss: 0.6520996909345789
Epoch 90/100, Loss: 0.6518959540620051
Epoch 91/100, Loss: 0.6516996297518677
Epoch 92/100, Loss: 0.6515104243361041
Epoch 93/100, Loss: 0.6513280557825399
Epoch 94/100, Loss: 0.6511522532338445
Epoch 95/100, Loss: 0.6509827565647551
Epoch 96/100, Loss: 0.6508193159568441
Epoch 97/100, Loss: 0.6506616914901353
Epoch 98/100, Loss: 0.6505096527509024
Epoch 99/100, Loss: 0.650362978455006
Epoch 100/100, Loss: 0.6502214560861559
```

### 4.3 AutoEncoder + KNN
#### a) 
```
Validation Accuracy: 0.12719298245614036
Validation Precision: 0.11967390867990128
Validation Recall: 0.1278611347212864
Validation F1_macro Score: 0.12363212537583165
```
#### b) Comparision

* Assignment 1 result:  

```
1. k: 19, Distance Metric: manhattan, Validation Accuracy:0.18
```

* Assignment 2 result:
```
Validation Accuracy: 0.06385964912280702  
Validation Precision: 0.05657442764395354
Validation Recall: 0.06494332842812192 Validation F1_macro Score: 0.06047069586991169
```

* AutoEncoder Result:

```
Validation Accuracy: 0.12719298245614036
Validation Precision: 0.11967390867990128
Validation Recall: 0.1278611347212864
Validation F1_macro Score: 0.12363212537583165
```

##### Observations:
It is quite evident that the Auto-encoder resulted in a much better accuracy in terms of prediction as compared to the PCA reduced dataset.  
We can conclude from this that Auto-encoder results in a better dimentionality reduction as compared to traditional PCA.

### 4.4 MLP classification

* MLP_class:
```
mlp_classifier = MLPClassifier(input_size=input_size,hidden_layers=hidden_layers,output_size=output_size,       activation='tanh',optimizer='sgd',epochs=100)
```

* Metrics:
```
Validation Accuracy: 0.18140350877192982
Validation Precision: 0.13332388388084004
Validation Recall: 0.1782278132112412
Validation F1_score: 0.15253984808748156
```

As compared to the previous result it can be easily observed that the perormance based on accuracy is better in case of MLP classifier as compared to the Auto-Encoder.   
Infact the result is better than the original KNN model showing how neural networks results in better results.