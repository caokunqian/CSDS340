import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import cycle

# Load the dataset
data_path = 'train.csv'
data = pd.read_csv(data_path)

# Separating features and label
X = data.iloc[:, :-1].values  
y = data.iloc[:, -1].values
n_classes = len(np.unique(y))  

# Splitting the dataset into training and testing sets for initial assessment
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)

# Define preprocessing and models pipeline
pipelines = {
    'lr': Pipeline([('scaler', StandardScaler()), ('classifier', LogisticRegression(random_state=100))]),
    'svm': Pipeline([('scaler', StandardScaler()), ('classifier', SVC(probability=True, random_state=100))]),
    'dt': Pipeline([('scaler', StandardScaler()), ('classifier', DecisionTreeClassifier(random_state=100))])
}

# Parameters for GridSearchCV
param_grid = {
    'lr': {'classifier__C': [0.01, 0.1, 1, 10, 100]},
    'svm': {'classifier__C': [0.01, 0.1, 1, 10, 100], 'classifier__gamma': [0.001, 0.01, 0.1, 1, 10]},
    'dt': {'classifier__max_depth': [None, 10, 20, 30, 40, 50], 'classifier__min_samples_split': [2, 5, 10]}
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


from sklearn.model_selection import validation_curve
from sklearn.model_selection import learning_curve

# Helper function for plotting the validation curve
def plot_validation_curve(estimator, title, X, y, param_name, param_range, cv=10, scoring='accuracy'):
    # If the estimator is a Pipeline, extract the last step as the actual classifier
    if isinstance(estimator, Pipeline):
        estimator_name = list(estimator.named_steps.keys())[-1]
        param_name = f'{estimator_name}__{param_name}'

    train_scores, test_scores = validation_curve(
        estimator, X, y, param_name=param_name, param_range=param_range,
        cv=cv, scoring=scoring, n_jobs=-1)

    # ... (rest of the code remains unchanged)

    
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.title(title)
    plt.xlabel(param_name)
    plt.ylabel(scoring)
    plt.ylim(0.0, 1.1)
    lw = 2
    plt.semilogx(param_range, train_scores_mean, label="Training score",
                 color="darkorange", lw=lw)
    plt.fill_between(param_range, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.2,
                     color="darkorange", lw=lw)
    plt.semilogx(param_range, test_scores_mean, label="Cross-validation score",
                 color="navy", lw=lw)
    plt.fill_between(param_range, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.2,
                     color="navy", lw=lw)
    plt.legend(loc="best")
    plt.show()

# Helper function for plotting the learning curve
def plot_learning_curve(estimator, title, X, y, cv, n_jobs=-1, train_sizes=np.linspace(.1, 1.0, 5)):
    plt.figure()
    plt.title(title)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.grid()
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    plt.show()

# Create the validation and learning curves for each model
# Plot validation curves for each model
for name, model in best_estimators.items():
    model_params = list(param_grid[name].keys())
    for param_name in model_params:
        param_range = np.array(param_grid[name][param_name])
        if isinstance(param_range[0], list):
            param_range = np.array(param_range[0])
        # Plot validation curve only for hyperparameters with a range of values
        if len(param_range) > 1:
            plot_validation_curve(model, f'Validation Curve for {name.upper()}', X_train, y_train,
                                  param_name=param_name.split('__')[1],
                                  param_range=param_range, cv=10, scoring='accuracy')

# Plot learning curves for each model
for name, model in best_estimators.items():
    plot_learning_curve(model, f'Learning Curve for {name.upper()}', X_train, y_train, cv=10, n_jobs=-1)


def plot_validation_curve(estimator, title, X, y, param_name, param_range, cv=10, scoring='accuracy'):
    train_scores, test_scores = validation_curve(
        estimator, X, y, param_name=param_name, param_range=param_range,
        cv=cv, scoring=scoring, n_jobs=-1)
    
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    
    plt.title(title)
    plt.xlabel(param_name)
    plt.ylabel("Score")
    plt.ylim(0.0, 1.1)
    lw = 2
    plt.semilogx(param_range, train_scores_mean, label="Training score",
                 color="darkorange", lw=lw)
    plt.fill_between(param_range, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.2,
                     color="darkorange", lw=lw)
    plt.semilogx(param_range, test_scores_mean, label="Cross-validation score",
                 color="navy", lw=lw)
    plt.fill_between(param_range, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.2,
                     color="navy", lw=lw)
    plt.legend(loc="best")
    plt.show()


from sklearn.model_selection import learning_curve

def plot_learning_curve(estimator, title, X, y, cv=10, n_jobs=-1, train_sizes=np.linspace(.1, 1.0, 5)):
    plt.figure()
    plt.title(title)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    plt.show()


# Plot validation curves for each model
for name, model in best_estimators.items():
    param_range = param_grid[name][f'classifier__{list(param_grid[name].keys())[0]}']
    plot_validation_curve(model, f'Validation Curve for {name.upper()}', X_train, y_train, 
                          param_name=f'classifier__{list(param_grid[name].keys())[0]}',
                          param_range=param_range, cv=10, scoring='accuracy')

# Plot learning curves for each model
for name, model in best_estimators.items():
    plot_learning_curve(model, f'Learning Curve for {name.upper()}', X_train, y_train, cv=10, n_jobs=-1)






