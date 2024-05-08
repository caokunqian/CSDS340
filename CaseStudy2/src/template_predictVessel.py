import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.cluster import KMeans
import functools
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
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


def evaluate():
    csv_path = './Data/set3.csv'
    labels_true = pd.read_csv(csv_path)['VID'].to_numpy()
    labels_pred = predictor(csv_path)
    rand_index_score = adjusted_rand_score(labels_true, labels_pred)
    print(f'Adjusted Rand Index Score of set3.csv: {rand_index_score:.4f}')
    print(f'The predicted VID for set3.csv:{labels_pred}')

'''
The following are edited code
'''
def get_predict_score():
    file_names = ['set1.csv', 'set2.csv']
    for file_name in file_names:
        csv_path = './Data/' + file_name
        labels_true = pd.read_csv(csv_path)['VID'].to_numpy()
        labels_pred = predictor(csv_path)
        rand_index_score = adjusted_rand_score(labels_true, labels_pred)
        print(f'Adjusted Rand Index Cluster Predict Score of {file_name}: {rand_index_score:.4f}')
        # you can uncomment the following to see LAT vs LON
        #plot_clusters(csv_path, labels_pred)
        

def load_and_preprocess_data(csv_path):
    df = pd.read_csv(csv_path, converters={'SEQUENCE_DTTM': hh_mm_ss2seconds})
    selected_features = ['SEQUENCE_DTTM', 'LAT', 'LON', 'SPEED_OVER_GROUND', 'COURSE_OVER_GROUND']
    X = df[selected_features].to_numpy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled

def predictor(csv_path):
    X_scaled = load_and_preprocess_data(csv_path)
    # GMM clustering
    gmm = GaussianMixture(n_components=20, random_state=123)  # Adjust n_components as needed  
    gmm.fit(X_scaled)
    labels_pred = gmm.predict(X_scaled)
    return labels_pred


if __name__=="__main__":
    get_baseline_score()
    evaluate()
    get_predict_score()


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