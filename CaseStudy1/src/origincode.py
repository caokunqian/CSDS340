import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler, PowerTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

import matplotlib.pyplot as plt
from sklearn.model_selection import validation_curve


# Load the dataset
data = pd.read_csv('D:/University/Course/Grade4/CSDS340/Case Study/code/train.csv')
X = data.iloc[:,0:-1].values  #features
y = data.iloc[:,-1].values #label

# Splitting the dataset into training and testing sets for initial assessment
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)

# Defining preprocessing and models pipeline
pipelines = {
    'lr': Pipeline([('scaler', StandardScaler()), ('classifier', LogisticRegression(random_state=1))]),
    'svm': Pipeline([('scaler', StandardScaler()), ('classifier', SVC(random_state=1))]),
    'dt': Pipeline([('scaler', StandardScaler()), ('classifier', DecisionTreeClassifier(random_state=1))])
}

# Parameters for GridSearchCV
param_grid = {
    'lr': {'classifier__C': [0.01, 0.1, 1, 10, 100]},
    'svm': {'classifier__C': [0.01, 0.1, 1, 10, 100], 'classifier__gamma': [0.001, 0.01, 0.1, 1,10]},
    'dt': {'classifier__max_depth': [None, 10, 20, 30, 40, 50], 'classifier__min_samples_split': [2, 5, 10]}
}

# Hyperparameters tuning and model selection
cv_scores = {}
best_estimators = {}
for name, pipeline in pipelines.items():
    # Using GridSearchCV for hyperparameter tuning
    grid_search = GridSearchCV(estimator=pipeline, param_grid=param_grid[name], cv=10, scoring='accuracy')
    grid_search.fit(X_train, y_train)
    cv_scores[name] = grid_search.best_score_
    best_estimators[name] = grid_search.best_estimator_

# Identifying the best model
best_model_name = max(cv_scores, key=cv_scores.get)
best_model = best_estimators[best_model_name]

# Final evaluation with the test set
y_pred = best_model.predict(X_test)
test_accuracy = accuracy_score(y_test, y_pred)
print(f"Best Model: {best_model_name}\nTest Accuracy: {test_accuracy:.2%}")


for name, pipeline in pipelines.items():
    # Using GridSearchCV for hyperparameter tuning
    grid_search = GridSearchCV(estimator=pipeline, param_grid=param_grid[name], cv=10, scoring='accuracy')
    grid_search.fit(X_train, y_train)
    cv_scores[name] = grid_search.best_score_
    best_estimators[name] = grid_search.best_estimator_
    # Print the best parameters for each model
    print(f"Best parameters for {name}: {grid_search.best_params_}")








def plot_validation_curve_for_best_model(best_model_name, X_train, y_train):
    if best_model_name == 'lr':
        param_name = 'classifier__C'
        param_range = np.logspace(-3, 2, 6)
    elif best_model_name == 'svm':
        param_name = 'classifier__C'
        param_range = np.logspace(-3, 2, 6)
    elif best_model_name == 'dt':
        param_name = 'classifier__max_depth'
        param_range = np.arange(1, 51, 10)
    else:
        raise ValueError("Unknown model type.")

    # Adjusting the pipeline to include the best model directly might not be straightforward because
    # the best model is already fitted. Instead, we use the validation curve to plot against the parameter of interest.
    
    # Extract the classifier from the best_estimator_ for validation curve plotting
    estimator = pipelines[best_model_name]
    
    train_scores, test_scores = validation_curve(
        estimator, X_train, y_train, param_name=param_name,
        param_range=param_range, cv=10, scoring="accuracy", n_jobs=-1)

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.figure()
    plt.title("Validation Curve for the Best Model: " + best_model_name)
    plt.xlabel(param_name)
    plt.ylabel("Score")
    plt.ylim(0.0, 1.1)
    plt.semilogx(param_range, train_scores_mean, label="Training score",
                 color="darkorange", lw=2)
    plt.fill_between(param_range, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, color="darkorange", alpha=0.2)
    plt.semilogx(param_range, test_scores_mean, label="Cross-validation score",
                 color="navy", lw=2)
    plt.fill_between(param_range, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, color="navy", alpha=0.2)
    plt.legend(loc="best")
    plt.show()

# Now, call this function with the best model name
plot_validation_curve_for_best_model(best_model_name, X_train, y_train)
