import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

# Load the dataset
data = pd.read_csv('D:/University/Course/Grade4/CSDS340/Case Study/code/train.csv')
X = data.iloc[:, 0:-1].values  # features
y = data.iloc[:, -1].values  # label

# Splitting the dataset into training and testing sets for initial assessment
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)

# Defining preprocessing and model pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()), 
    ('classifier', SVC(random_state=1))
])
# Parameters for GridSearchCV
param_grid = {
    'classifier__C': [100], 
    'classifier__gamma': [0.1]
}
# Hyperparameter tuning
grid_search = GridSearchCV(estimator=pipeline, param_grid=param_grid, cv=10, scoring='accuracy')
grid_search.fit(X_train, y_train)
# Best model and its performance
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
test_accuracy = accuracy_score(y_test, y_pred)
print(f"Best Model Parameters: {grid_search.best_params_}\nTest Accuracy: {test_accuracy:.2%}")
