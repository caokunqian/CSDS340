import pandas as pd
import numpy as np
from sklearn import preprocessing
import functools
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.preprocessing import StandardScaler
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
The following are edited code
'''

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
The evaluatation
'''
def evaluate_DBSCAN_parameters():
    file_names = ['set1.csv', 'set2.csv']
    best_score = -1
    best_params = (0, 0)
    
    # Define ranges to test
    eps_values = np.arange(0.1, 1.0, 0.1)  # Adjust the range and step as necessary
    min_samples_values = range(5, 50)       # Adjust the range as necessary

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

def evaluate_gmm_parameters():
    file_names = ['set1.csv', 'set2.csv']
    best_score = -1
    best_params = (0, '')
    
    n_components_range = range(2, 50)  # Example range for number of clusters
    covariance_type_options = ['full', 'tied', 'diag', 'spherical']
    
    for file_name in file_names:
        csv_path = './Data/' + file_name
        data = load_and_preprocess_data(csv_path)
        labels_true = pd.read_csv(csv_path)['VID'].to_numpy()
        
        for n_components in n_components_range:
            for covariance_type in covariance_type_options:
                gmm = GaussianMixture(n_components=n_components, covariance_type=covariance_type, random_state=42)
                gmm.fit(data)
                labels_pred = gmm.predict(data)
                rand_index_score = adjusted_rand_score(labels_true, labels_pred)
                
                if rand_index_score > best_score:
                    best_score = rand_index_score
                    best_params = (n_components, covariance_type)
                
                print(f'GMM Parameters - n_components: {n_components}, Covariance Type: {covariance_type}, File: {file_name}, ARI: {rand_index_score:.4f}')
    
    print(f'Best GMM score: {best_score:.4f} with Parameters: {best_params}')
    return best_params

def evaluate_hierarchical_parameters():
    file_names = ['set1.csv', 'set2.csv']
    best_score = -1
    best_params = (0, '')
    
    n_clusters_range = range(2, 100)  # Example range for number of clusters
    linkage_options = ['ward', 'complete', 'average', 'single']
    
    for file_name in file_names:
        csv_path = './Data/' + file_name
        data = load_and_preprocess_data(csv_path)
        labels_true = pd.read_csv(csv_path)['VID'].to_numpy()
        
        for n_clusters in n_clusters_range:
            for linkage in linkage_options:
                clustering = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
                labels_pred = clustering.fit_predict(data)
                rand_index_score = adjusted_rand_score(labels_true, labels_pred)
                
                if rand_index_score > best_score:
                    best_score = rand_index_score
                    best_params = (n_clusters, linkage)
                
                print(f'Agglomerative Parameters - n_clusters: {n_clusters}, Linkage: {linkage}, File: {file_name}, ARI: {rand_index_score:.4f}')
    
    print(f'Best Hierarchical score: {best_score:.4f} with Parameters: {best_params}')
    return best_params

def evaluate_birch_parameters():
    file_names = ['set1.csv', 'set2.csv']
    best_score = -1
    best_params = (0, 0)  # Initialize best parameters
    
    # Parameter ranges to explore
    n_clusters_range = range(2, 70)  # Range for number of clusters
    threshold_range = np.linspace(0.1, 0.5, 5)  # Range for threshold values
    
    for file_name in file_names:
        csv_path = './Data/' + file_name
        data = load_and_preprocess_data(csv_path)
        labels_true = pd.read_csv(csv_path)['VID'].to_numpy()
        
        for n_clusters in n_clusters_range:
            for threshold in threshold_range:
                birch = Birch(n_clusters=n_clusters, threshold=threshold)
                labels_pred = birch.fit_predict(data)
                rand_index_score = adjusted_rand_score(labels_true, labels_pred)
                
                if rand_index_score > best_score:
                    best_score = rand_index_score
                    best_params = (n_clusters, threshold)
                
                print(f'BIRCH Parameters - n_clusters: {n_clusters}, Threshold: {threshold:.2f}, File: {file_name}, ARI: {rand_index_score:.4f}')
    
    print(f'Best BIRCH score: {best_score:.4f} with Parameters: {best_params}')
    return best_params

def evaluate_bisecting_kmeans_parameters():
    file_names = ['set1.csv', 'set2.csv']
    best_score = -1
    best_params = 0  # Initialize best depth
    
    # Parameter range to explore
    max_depth_range = range(1, 6)  # Corresponds to depth in a binary tree
    
    for file_name in file_names:
        csv_path = './Data/' + file_name
        data = load_and_preprocess_data(csv_path)
        labels_true = pd.read_csv(csv_path)['VID'].to_numpy()
        
        for max_depth in max_depth_range:
            n_clusters = 2 ** max_depth
            kmeans = KMeans(n_clusters=n_clusters, random_state=123)
            labels_pred = kmeans.fit_predict(data)
            rand_index_score = adjusted_rand_score(labels_true, labels_pred)
            
            if rand_index_score > best_score:
                best_score = rand_index_score
                best_params = max_depth
            
            print(f'Bisecting K-Means Parameters - Max Depth: {max_depth}, Equivalent Clusters: {n_clusters}, File: {file_name}, ARI: {rand_index_score:.4f}')
    
    print(f'Best Bisecting K-Means score: {best_score:.4f} with Max Depth: {best_params}')
    return best_params

'''
Debug print
Uncomment eva to get output for certain function
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
        #plot_clusters(csv_path, labels_pred)  # Uncomment if you have implemented plot_clusters




if __name__ == "__main__":
    get_baseline_score()
    #evaluate_DBSCAN_parameters()
    #evaluate_gmm_parameters()
    #evaluate_hierarchical_parameters()
    #evaluate_birch_parameters()
    evaluate_bisecting_kmeans_parameters()
    # Optionally, run the best configuration again to plot or further analyze
    #eps, min_samples = best_params
    #evaluate_with_best_params(eps, min_samples)  # You need to implement this if you want to see the best performance separately









    


