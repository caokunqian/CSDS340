import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, validation_curve
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline

# Load the dataset
data_path = 'D:/University/Course/Grade4/CSDS340/Case Study/code/train.csv'
data = pd.read_csv(data_path)

# Separating features and label
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# Splitting the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)

# Logistic Regression - Validation Curve for 'C'
pipeline_lr = Pipeline([('scaler', StandardScaler()), ('classifier', LogisticRegression(random_state=1))])
param_range_lr = np.logspace(-5, 2, 6)
train_scores_lr, test_scores_lr = validation_curve(
    pipeline_lr, X_train, y_train, param_name='classifier__C', param_range=param_range_lr,
    cv=5, scoring="accuracy", n_jobs=-1)

# SVM - Validation Curve for 'C'
pipeline_svm = Pipeline([('scaler', StandardScaler()), ('classifier', SVC(probability=True, random_state=1))])
param_range_svm = np.logspace(-6, 3, 10)
train_scores_svm, test_scores_svm = validation_curve(
    pipeline_svm, X_train, y_train, param_name='classifier__C', param_range=param_range_svm,
    cv=5, scoring="accuracy", n_jobs=-1)

# Decision Tree - Validation Curve for 'max_depth'
pipeline_dt = Pipeline([('scaler', StandardScaler()), ('classifier', DecisionTreeClassifier(random_state=1))])
param_range_dt = np.arange(1, 15)
train_scores_dt, test_scores_dt = validation_curve(
    pipeline_dt, X_train, y_train, param_name='classifier__max_depth', param_range=param_range_dt,
    cv=5, scoring="accuracy", n_jobs=-1)

# Plotting the validation curve
def plot_validation_curve(train_scores, test_scores, param_range, title, xlabel):
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)

    plt.plot(param_range, train_mean, color='blue', marker='o', markersize=5, label='training accuracy')
    plt.fill_between(param_range, train_mean + train_std, train_mean - train_std, alpha=0.15, color='blue')

    plt.plot(param_range, test_mean, color='green', linestyle='--', marker='s', markersize=5, label='validation accuracy')
    plt.fill_between(param_range, test_mean + test_std, test_mean - test_std, alpha=0.15, color='green')

    plt.grid()
    plt.xscale('log')
    plt.legend(loc='lower right')
    plt.xlabel(xlabel)
    plt.ylabel('Accuracy')
    plt.ylim([0.5, 1.0])
    plt.title(title)
    plt.show()

# Plot validation curve for each classifier
plot_validation_curve(train_scores_lr, test_scores_lr, param_range_lr, 'Validation Curve for Logistic Regression', 'Parameter C')
plot_validation_curve(train_scores_svm, test_scores_svm, param_range_svm, 'Validation Curve for SVM', 'Parameter C')
plot_validation_curve(train_scores_dt, test_scores_dt, param_range_dt, 'Validation Curve for Decision Tree', 'Parameter max_depth')
