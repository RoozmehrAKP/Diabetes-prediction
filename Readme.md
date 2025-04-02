
# Diabetes Prediction Project

## Overview
This project aims to predict the likelihood of diabetes based on medical features such as insulin levels, skin thickness, glucose levels, and more. Several machine learning models have been implemented and tested, including **K-Nearest Neighbors (KNN), Support Vector Machine (SVM), Logistic Regression, Random Forest, XGBoost,** and **Artificial Neural Networks (ANN)**. The objective is to compare the models' performance and determine the most accurate model for diabetes prediction.

## Dataset
The dataset used for this project contains medical data that is used to predict the likelihood of diabetes. It includes features such as insulin levels, glucose, BMI, age, and more. You can access the dataset via the following Google Drive link:

**[Google Drive Link to Dataset](Insert your Google Drive link here)**

## Installation and Setup

### Required Libraries
To run the project, the following Python libraries are required:
- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `sklearn`
- `xgboost`
- `tensorflow`
- `keras`

You can install these libraries using the following command:

```bash
pip install -r requirements.txt
```

### Data Loading
Once the required libraries are installed, the dataset can be loaded into the project. Make sure to place the dataset file (e.g., `diabetes.csv`) in the appropriate directory.

```python
import pandas as pd
data = pd.read_csv('path_to_diabetes.csv')
```

## Data Preprocessing

### Handling Missing Values
The dataset includes missing values represented by zeros in certain columns (e.g., **Insulin**, **SkinThickness**). These zeros are replaced with the mean or median values of the corresponding columns, calculated only from rows with non-zero values.

```python
# Replace zeros with the mean of the respective column (ignoring zeros)
data['Insulin'] = data['Insulin'].replace(0, data['Insulin'][data['Insulin'] > 0].mean())
data['SkinThickness'] = data['SkinThickness'].replace(0, data['SkinThickness'][data['SkinThickness'] > 0].mean())
```

### Feature Scaling
To enhance the performance of some models like KNN and ANN, the data is standardized using **RobustScaler** (to handle outliers) and **StandardScaler**.

```python
from sklearn.preprocessing import RobustScaler
scaler = RobustScaler()
scaled_data = scaler.fit_transform(data.drop(columns='Outcome'))
```

## Model Implementation

### 1. K-Nearest Neighbors (KNN)
KNN was used to classify diabetes based on the features of the dataset. The best performance was obtained using **k=14**.

```python
from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier(n_neighbors=14)
model.fit(X_train, y_train)
```

### 2. Support Vector Machine (SVM)
SVM with **RBF kernel** showed promising results. Hyperparameters were tuned using **GridSearchCV** to find the best `C` and `gamma`.

```python
from sklearn.svm import SVC
model = SVC(C=1, gamma=0.01, kernel='rbf')
model.fit(X_train, y_train)
```

### 3. Logistic Regression
A simple Logistic Regression model was implemented as a baseline for comparison.

```python
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train, y_train)
```

### 4. Random Forest
A Random Forest classifier was used to evaluate its performance for diabetes prediction.

```python
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)
```

### 5. XGBoost
XGBoost was applied to improve model accuracy with a gradient boosting approach.

```python
import xgboost as xgb
model = xgb.XGBClassifier()
model.fit(X_train, y_train)
```

### 6. Artificial Neural Network (ANN)
An ANN was built using **Keras** and **TensorFlow**, with optimization through **Dropout** and **EarlyStopping**.

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
model = Sequential([
    Dense(32, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.3),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')
])
```

## Model Evaluation

### Accuracy Comparison
The test accuracy for each model was measured and compared:

| Model                        | Test Accuracy  |
|------------------------------|----------------|
| **KNN (k=14, RobustScaler)**  | 0.7489         |
| **Logistic Regression**       | 0.7316         |
| **Random Forest**             | 0.7316         |
| **SVM (C=1, gamma=0.01, rbf)**| 0.7489         |
| **XGBoost**                   | 0.7403         |
| **ANN (Optimized)**           | 0.7532         |

📌 **Best Model: SVM (RBF kernel) achieved the best results in terms of training accuracy (0.7709) and test accuracy (0.7489).**

## Conclusion

- **Best Model: SVM** achieved the best performance overall with the highest test accuracy.
- **ANN** showed comparable results but was more computationally intensive.
- **XGBoost** and **Random Forest** performed well but did not surpass SVM in terms of accuracy.

Future improvements could include:
- **Hyperparameter tuning** for ANN and Random Forest.
- **Feature engineering** to improve model input features.
- **Ensemble methods** to combine the best aspects of multiple models.

---

**If you would like to receive a copy of the dataset or any additional files, feel free to contact me!**

---
