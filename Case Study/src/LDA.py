import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
data_path = 'train.csv'
data = pd.read_csv(data_path)

# Separating features and label
X = data.iloc[:, :-1].values  
y = data.iloc[:, -1].values

# Splitting the dataset into training and testing sets for initial assessment
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)

# Define preprocessing and models pipeline
pipelines = {
    'lr': Pipeline([('scaler', StandardScaler()), ('lda', LinearDiscriminantAnalysis()), ('classifier', LogisticRegression(random_state=100))]),
    'svm': Pipeline([('scaler', StandardScaler()), ('lda', LinearDiscriminantAnalysis()), ('classifier', SVC(probability=True, random_state=100))]),
    'dt': Pipeline([('scaler', StandardScaler()), ('lda', LinearDiscriminantAnalysis()), ('classifier', DecisionTreeClassifier(random_state=100))])
}

# Parameters for GridSearchCV (Optional: Add/modify parameters based on your model's needs)
param_grid = {
    'lr': {'lda__n_components': [None], 'classifier__C': [0.01, 0.1, 1, 10, 100]},
    'svm': {'lda__n_components': [None], 'classifier__C': [0.01, 0.1, 1, 10, 100], 'classifier__gamma': [0.001, 0.01, 0.1, 1, 10]},
    'dt': {'lda__n_components': [None], 'classifier__max_depth': [None, 10, 20, 30, 40, 50], 'classifier__min_samples_split': [2, 5, 10]}
}

# Hyperparameters tuning and model selection
cv_scores = {}
best_estimators = {}
for name, pipeline in pipelines.items():
    grid_search = GridSearchCV(estimator=pipeline, param_grid=param_grid[name], cv=10, scoring='accuracy')
    grid_search.fit(X_train, y_train)
    cv_scores[name] = grid_search.best_score_
    best_estimators[name] = grid_search.best_estimator_

# Identifying the best model
best_model_name = max(cv_scores, key=cv_scores.get)
best_model = best_estimators[best_model_name]

# Evaluate the best model on the test set
y_pred = best_model.predict(X_test)
test_accuracy = accuracy_score(y_test, y_pred)
print(f"Best Model: {best_model_name}\nTest Accuracy: {test_accuracy:.2%}")
