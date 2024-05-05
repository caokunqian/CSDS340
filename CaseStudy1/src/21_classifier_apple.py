import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

'''
# You may uncomment it for testing
path = './Data/test.csv'
data = pd.read_csv(path)
X_test, y_test = data.iloc[:, :-1].values, data.iloc[:, -1].values

'''

# Load dataset
path = './Data/train.csv'
data = pd.read_csv(path)
X, y = data.iloc[:, :-1].values, data.iloc[:, -1].values

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)

# Define and configure the pipeline
pipeline = Pipeline([('scaler', StandardScaler()), ('classifier', SVC(random_state=1))])
param_grid = {'classifier__C': [0.01, 0.1, 1, 10, 100], 'classifier__gamma': [0.001, 0.01, 0.1, 1, 10]}
#we delete two other models, if you want to see it is named ''
grid_search = GridSearchCV(estimator=pipeline, param_grid=param_grid, cv=10, scoring='accuracy')
grid_search.fit(X_train, y_train)

# print results
y_pred = grid_search.predict(X_test)
print(f"Best Model Parameters: {grid_search.best_params_}\nTest Accuracy: {accuracy_score(y_test, y_pred):.2%}")
