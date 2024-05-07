import pandas as pd
import numpy as np
from sklearn import preprocessing
import functools
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.metrics import adjusted_rand_score
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans, AffinityPropagation, MeanShift, SpectralClustering, AgglomerativeClustering, OPTICS, Birch

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
We only add following code, keep above unchanged
'''
def plot_clusters(csv_path, labels_pred, method_name):
    df = pd.read_csv(csv_path)
    plt.scatter(df['LON'], df['LAT'], c=labels_pred, cmap='viridis')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title(f'{method_name} Cluster Visualization for {csv_path}')
    plt.show()

def load_and_preprocess_data(csv_path):
    df = pd.read_csv(csv_path, converters={'SEQUENCE_DTTM': hh_mm_ss2seconds})
    selected_features = ['SEQUENCE_DTTM', 'LAT', 'LON', 'SPEED_OVER_GROUND', 'COURSE_OVER_GROUND']
    X = df[selected_features].to_numpy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled

'''
DBSCAN
'''
def predictor_dbscan(csv_path):
    X_scaled = load_and_preprocess_data(csv_path)
    # DBSCAN clustering
    clustering = DBSCAN(eps=0.1, min_samples=2).fit(X_scaled)
    labels_pred = clustering.labels_
    return labels_pred
'''
Gaussian Mixture Models (GMM)
'''
def predictor_gmm(csv_path):
    X_scaled = load_and_preprocess_data(csv_path)
    # GMM clustering
    gmm = GaussianMixture(n_components=20, random_state=123)  # Adjust n_components as needed  
    gmm.fit(X_scaled)
    labels_pred = gmm.predict(X_scaled)
    return labels_pred

'''
Hierarchical Clustering
'''
def predictor_hierarchical(csv_path):
    X_scaled = load_and_preprocess_data(csv_path)
    # Hierarchical clustering
    clustering = AgglomerativeClustering(n_clusters=10)  # Adjust n_clusters as needed
    labels_pred = clustering.fit_predict(X_scaled)
    return labels_pred

'''
Spectral Clustering
'''
def predictor_spectral(csv_path):
    X_scaled = load_and_preprocess_data(csv_path)
    # Spectral clustering
    clustering = SpectralClustering(n_clusters=20, random_state=123, affinity='nearest_neighbors')
    labels_pred = clustering.fit_predict(X_scaled)
    return labels_pred

'''
Affinity Propagation
'''
def predictor_affinity_propagation(csv_path):
    X_scaled = load_and_preprocess_data(csv_path)
    clustering = AffinityPropagation(damping=0.9, preference=-50)
    labels_pred = clustering.fit_predict(X_scaled)
    return labels_pred

'''
Mean-Shift
'''
def predictor_mean_shift(csv_path):
    X_scaled = load_and_preprocess_data(csv_path)
    clustering = MeanShift(bandwidth=None)  # None lets the algorithm estimate the bandwidth
    labels_pred = clustering.fit_predict(X_scaled)
    return labels_pred

'''
OPTICS
'''
def predictor_optics(csv_path):
    X_scaled = load_and_preprocess_data(csv_path)
    clustering = OPTICS(min_samples=5, max_eps=2.0)
    labels_pred = clustering.fit_predict(X_scaled)
    return labels_pred

'''
BIRCH
'''
def predictor_birch(csv_path):
    X_scaled = load_and_preprocess_data(csv_path)
    clustering = Birch(n_clusters=20)
    labels_pred = clustering.fit_predict(X_scaled)
    return labels_pred

'''
Bisecting K-Means (Simulated)
'''
def predictor_bisecting_kmeans(csv_path, depth=5):
    X_scaled = load_and_preprocess_data(csv_path)
    n_clusters = 2 ** depth  # Determines the number of final clusters
    labels_pred = KMeans(n_clusters=n_clusters, random_state=123).fit_predict(X_scaled)
    return labels_pred

'''
The following secction is for debugging
You can uncomment plot() to see LAT,LON graph
You can uncomment # to see those methods, there are 9 in total
'''

def get_predict_score1():
    file_names = ['set1.csv', 'set2.csv']
    methods = {
        #Those method have high acc
        #'DBSCAN': predictor_dbscan, 
        'GMM': predictor_gmm,
        'Hierarchical': predictor_hierarchical,
       # 'BIRCH': predictor_birch,
        #'Bisecting K-Means': predictor_bisecting_kmeans  
    }
    for file_name in file_names:
        csv_path = './Data/' + file_name
        labels_true = pd.read_csv(csv_path)['VID'].to_numpy()
        for method_name, predictor in methods.items():
            labels_pred = predictor(csv_path)
            rand_index_score = adjusted_rand_score(labels_true, labels_pred)
            print(f'{method_name} Adjusted Rand Index Score of {file_name}: {rand_index_score:.4f}')
            #plot_clusters(csv_path, labels_pred, method_name)


def get_predict_score2():
    file_names = ['set1.csv', 'set2.csv']
    methods = {
        #Those method have low acc, Affinity and Mean-shift need long time to run
        'Affinity Propagation': predictor_affinity_propagation,
        'Mean-Shift': predictor_mean_shift,
        'OPTICS': predictor_optics,
        'Spectral': predictor_spectral,
    }
    for file_name in file_names:
        csv_path = './Data/' + file_name
        labels_true = pd.read_csv(csv_path)['VID'].to_numpy()
        for method_name, predictor in methods.items():
            labels_pred = predictor(csv_path)
            rand_index_score = adjusted_rand_score(labels_true, labels_pred)
            print(f'{method_name} Adjusted Rand Index Score of {file_name}: {rand_index_score:.4f}')
            #plot_clusters(csv_path, labels_pred, method_name)

if __name__=="__main__":
    get_baseline_score()
    #evaluate()
    get_predict_score1()
    


