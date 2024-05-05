from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, cross_val_score, validation_curve
from sklearn.metrics import roc_auc_score
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('D:/University/Course/Grade4/CSDS340/Case Study/code/train.csv')

X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

#Standardize
scaler = StandardScaler()
X_standardized = scaler.fit_transform(X)

model = SVC(probability=True)

param_grid = {
    'C': np.logspace(0.1, 1, 5),  
    'gamma': np.logspace(0.1, 1, 5), 
    'kernel': ['rbf']  
}

cv = GridSearchCV(model, param_grid, cv=10, scoring='accuracy')  
cv.fit(X, y)

print("Best parameters found:", cv.best_params_)
print("Best cross-validation accuracy:", cv.best_score_)


