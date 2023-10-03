# -*- coding: utf-8 -*-
"""Student_Performance.ipynb

#Gradient Boost Tree Classifier
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from xgboost import XGBClassifier, plot_importance
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Import Dataset
df = pd.read_csv('/content/drive/MyDrive/AI/studentPerformance.csv')

# Separate features and target
y = df['30']
X = df.drop(['29', '30', 'COURSE ID', 'GRADE', 'STUDENT ID'], axis=1)

# Encode all features, including numeric ones, as categories
for feature in X.columns:
    le = LabelEncoder()
    X[feature] = le.fit_transform(X[feature])

# Encode the target variable
label_encoder_y = LabelEncoder()
y = label_encoder_y.fit_transform(y)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the XGBoost classifier with use_label_encoder=False
xgb_classifier = XGBClassifier(objective='multi:softmax', num_class=len(np.unique(y)), use_label_encoder=False)

# Define hyperparameters and their possible values for GridSearch
param_grid = {
    'n_estimators': [105, 110, 115],
    'learning_rate': [0.999, 0.1, 0.111],
    'max_depth': [3, 4, 7],
}

# Create GridSearchCV to find the best hyperparameters
grid_search = GridSearchCV(xgb_classifier, param_grid, cv=5, n_jobs=-1)

# Fit the model to the training data
grid_search.fit(X_train, y_train)

# Get the best hyperparameters and model
best_params = grid_search.best_params_
best_model = grid_search.best_estimator_

# Evaluate the model on the test set
y_pred = best_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Actual value for y:", y_test)
print("Predict value of y:", y_pred)
print("\n")
# Print the best hyperparameters and test accuracy
print("Best Hyperparameters:", best_params)
print("Test Accuracy:", accuracy)

# Get feature importances
importance_scores = best_model.feature_importances_

# Print feature importances
for i, score in enumerate(importance_scores):
    print(f"Feature {X.columns[i]} Importance: {score}")

# Visualize feature importances
plot_importance(best_model)
plt.show()

from google.colab import drive
drive.mount('/content/drive')

"""#Random Forest Classifier"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Import Dataset
df = pd.read_csv('/content/drive/MyDrive/AI/studentPerformance.csv')

# Separate features and target
y = df['30']
X = df.drop(['29', '30', 'COURSE ID', 'GRADE', 'STUDENT ID'], axis=1)

# Encode all features, including numeric ones, as categories
for feature in X.columns:
    le = LabelEncoder()
    X[feature] = le.fit_transform(X[feature])

# Encode the target variable
label_encoder_y = LabelEncoder()
y = label_encoder_y.fit_transform(y)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the Random Forest classifier
random_forest_classifier = RandomForestClassifier()

# Define hyperparameters and their possible values for GridSearch
param_grid = {
    'n_estimators': [90, 100, 110],
    'max_depth': [None, 10, 20, 3],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
}

# Create GridSearchCV to find the best hyperparameters
grid_search = GridSearchCV(random_forest_classifier, param_grid, cv=5, n_jobs=-1)

# Fit the model to the training data
grid_search.fit(X_train, y_train)

# Get the best hyperparameters and model
best_params = grid_search.best_params_
best_model = grid_search.best_estimator_

# Evaluate the model on the test set
y_pred = best_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# Print the best hyperparameters and test accuracy
print("Best Hyperparameters:", best_params)
print("Test Accuracy:", accuracy)

# Get feature importances
importance_scores = best_model.feature_importances_

# Print feature importances
for i, score in enumerate(importance_scores):
    print(f"Feature {X.columns[i]} Importance: {score}")

# Visualize feature importances
plt.figure(figsize=(8, 6))
plt.barh(range(len(importance_scores)), importance_scores, tick_label=X.columns)
plt.xlabel('Feature Importance')
plt.title('Random Forest Feature Importance')
plt.show()

from google.colab import drive
drive.mount('/content/drive')

"""#Support Vector Machine Model"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Import Dataset
df = pd.read_csv('/content/drive/MyDrive/AI/studentPerformance.csv')

# Separate features and target
y = df['30']
X = df.drop(['29', '30', 'COURSE ID', 'GRADE', 'STUDENT ID'], axis=1)

# Encode all features, including numeric ones, as categories
for feature in X.columns:
    le = LabelEncoder()
    X[feature] = le.fit_transform(X[feature])

# Encode the target variable
label_encoder_y = LabelEncoder()
y = label_encoder_y.fit_transform(y)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the SVM classifier with a linear kernel
svm_classifier = SVC(kernel='linear')

# Fit the model to the training data
svm_classifier.fit(X_train, y_train)

# Evaluate the model on the test set
y_pred = svm_classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# Print the test accuracy
print("Test Accuracy:", accuracy)

# Get feature coefficients (weights)
feature_weights = svm_classifier.coef_[0]

# Print and visualize feature coefficients
print("Feature Coefficients (Weights):")
for i, weight in enumerate(feature_weights):
    print(f"Feature {X.columns[i]} Weight: {weight}")

# Visualize feature coefficients
plt.figure(figsize=(8, 6))
plt.barh(range(len(feature_weights)), feature_weights, tick_label=X.columns)
plt.xlabel('Feature Coefficient (Weight)')
plt.title('SVM Feature Importance (Coefficients)')
plt.show()

from google.colab import drive
drive.mount('/content/drive')

"""#Stacked Generalization Ensemble"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Import Dataset
df = pd.read_csv('/content/drive/MyDrive/AI/studentPerformance.csv')

# Separate features and target
y = df['30']
X = df.drop(['29', '30', 'COURSE ID', 'GRADE', 'STUDENT ID'], axis=1)

# Encode all features, including numeric ones, as categories
for feature in X.columns:
    le = LabelEncoder()
    X[feature] = le.fit_transform(X[feature])

# Encode the target variable
label_encoder_y = LabelEncoder()
y = label_encoder_y.fit_transform(y)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Base Models
random_forest = RandomForestClassifier(n_estimators = 100, max_depth = None, min_samples_leaf = 2, min_samples_split = 10, random_state=42)
xgboost = XGBClassifier(objective='multi:softmax', num_class=len(np.unique(y)), use_label_encoder=False, random_state=42, learning_rate = 0.1, max_depth = 4, n_estimators = 110)

# Train base models
random_forest.fit(X_train, y_train)
xgboost.fit(X_train, y_train)

# Generate predictions from base models
rf_predictions = random_forest.predict(X_train)
xgb_predictions = xgboost.predict(X_train)

# Create a new DataFrame with base model predictions
stacked_predictions = pd.DataFrame({'RandomForest': rf_predictions, 'XGBoost': xgb_predictions})

# Meta-Model (Random Forest)
meta_model = RandomForestClassifier(random_state=42)

# Train the meta-model on base model predictions
meta_model.fit(stacked_predictions, y_train)

# Generate base model predictions on the test set
rf_test_predictions = random_forest.predict(X_test)
xgb_test_predictions = xgboost.predict(X_test)

# Create a new DataFrame with test set base model predictions
stacked_test_predictions = pd.DataFrame({'RandomForest': rf_test_predictions, 'XGBoost': xgb_test_predictions})

# Generate final predictions using the meta-model
final_predictions = meta_model.predict(stacked_test_predictions)

# Evaluate the stacked ensemble on the test set
accuracy = accuracy_score(y_test, final_predictions)
print("Stacked Ensemble Test Accuracy:", accuracy)

from google.colab import drive
drive.mount('/content/drive')
