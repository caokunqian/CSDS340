import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.cluster import KMeans
import functools
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.metrics import adjusted_rand_score
import matplotlib.pyplot as plt


def hh_mm_ss2seconds(hh_mm_ss):
    return functools.reduce(lambda acc, x: acc*60 + x, map(int, hh_mm_ss.split(':')))


def predictor_baseline(csv_path):
    # load data and convert hh:mm:ss to seconds
    df = pd.read_csv(csv_path, converters={'SEQUENCE_DTTM' : hh_mm_ss2seconds})
    # select features 
    selected_features = ['SEQUENCE_DTTM', 'LAT', 'LON', 'SPEED_OVER_GROUND' ,'COURSE_OVER_GROUND']
    X = df[selected_features].to_numpy()
    # Standardization 
    X = preprocessing.StandardScaler().fit(X).transform(X)
    # k-means with K = number of unique VIDs of set1
    K = 20 
    model = KMeans(n_clusters=K, random_state=123, n_init='auto').fit(X)
    # predict cluster numbers of each sample
    labels_pred = model.predict(X)
    return labels_pred


def get_baseline_score():
    file_names = ['set1.csv', 'set2.csv']
    for file_name in file_names:
        csv_path = './Data/' + file_name
        labels_true = pd.read_csv(csv_path)['VID'].to_numpy()
        labels_pred = predictor_baseline(csv_path)
        rand_index_score = adjusted_rand_score(labels_true, labels_pred)
        print(f'Adjusted Rand Index Baseline Score of {file_name}: {rand_index_score:.4f}')
        #plot_clusters(csv_path, labels_pred)


def evaluate():
    csv_path = './Data/set3.csv'
    labels_true = pd.read_csv(csv_path)['VID'].to_numpy()
    labels_pred = predictor(csv_path)
    rand_index_score = adjusted_rand_score(labels_true, labels_pred)
    print(f'Adjusted Rand Index Score of set3.csv: {rand_index_score:.4f}')
    #plot_clusters(csv_path, labels_pred)

'''
The following are edited code
'''


def get_predict_score_parameter_tuning():
    file_names = ['set1.csv', 'set2.csv']
    best_score = -1
    best_params = (0, 0)
    
    # Define ranges to test
    eps_values = np.arange(0.1, 1.0, 0.1)  # Adjust the range and step as necessary
    min_samples_values = range(2, 10)       # Adjust the range as necessary

    for eps in eps_values:
        for min_samples in min_samples_values:
            for file_name in file_names:
                csv_path = './Data/' + file_name
                labels_true = pd.read_csv(csv_path)['VID'].to_numpy()
                labels_pred = predictor(csv_path, eps, min_samples)
                rand_index_score = adjusted_rand_score(labels_true, labels_pred)
                
                # Store best score and parameters
                if rand_index_score > best_score:
                    best_score = rand_index_score
                    best_params = (eps, min_samples)
                
                print(f'EPS: {eps:.2f}, Min Samples: {min_samples}, File: {file_name}, ARI: {rand_index_score:.4f}')
    
    print(f'Best score: {best_score:.4f} with EPS: {best_params[0]:.2f} and Min Samples: {best_params[1]}')
    return best_params

def load_and_preprocess_data(csv_path):
    df = pd.read_csv(csv_path, converters={'SEQUENCE_DTTM': hh_mm_ss2seconds})
    selected_features = ['SEQUENCE_DTTM', 'LAT', 'LON', 'SPEED_OVER_GROUND', 'COURSE_OVER_GROUND']
    X = df[selected_features].to_numpy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled

def predictor(csv_path, eps, min_samples):
    X_scaled = load_and_preprocess_data(csv_path)
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(X_scaled)
    labels_pred = clustering.labels_
    return labels_pred


'''
Debug print
'''
def plot_clusters(csv_path, labels_pred):
    df = pd.read_csv(csv_path)
    plt.scatter(df['LON'], df['LAT'], c=labels_pred, cmap='viridis')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('Cluster visualization')
    plt.show()

def evaluate_with_best_params(eps, min_samples):
    file_names = ['set1.csv', 'set2.csv']
    for file_name in file_names:
        csv_path = './Data/' + file_name
        labels_true = pd.read_csv(csv_path)['VID'].to_numpy()
        labels_pred = predictor(csv_path, eps, min_samples)
        rand_index_score = adjusted_rand_score(labels_true, labels_pred)
        print(f'Best params on {file_name}: ARI: {rand_index_score:.4f}')
        plot_clusters(csv_path, labels_pred)  # Uncomment if you have implemented plot_clusters


if __name__ == "__main__":
    get_baseline_score()
    best_params = get_predict_score_parameter_tuning()
    # Optionally, run the best configuration again to plot or further analyze
    eps, min_samples = best_params
    evaluate_with_best_params(eps, min_samples)  # You need to implement this if you want to see the best performance separately

# Optional: Function to evaluate with the best parameters and possibly plot the results







    


