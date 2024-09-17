import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import os
import sys
import time
import seaborn as sns
import pyarrow as pa
import pyarrow.feather as feather
from sklearn.mixture import GaussianMixture
import scipy.cluster.hierarchy  as hc
import warnings
warnings.filterwarnings('ignore')

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from models.k_means.k_means import K_Means

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from models.gmm.gmm import GMM

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from models.pca.pca import PCA

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from models.kNearestN.knn import KNN, Model_Evaluation


#======================================functions============================================
def print_clusters(labels: np.ndarray, cluster_assignments: np.ndarray):
    clusters = defaultdict(list)
    for label, cluster in zip(labels, cluster_assignments):
        clusters[cluster].append(label)
    
    for cluster in sorted(clusters.keys()):
        print(f"Cluster {int(cluster)}:")
        print(", ".join(str(item[0]) for item in clusters[cluster]))
        print("-" * 40)

def plot_k_vs_cost(embeddings:np.ndarray, outPath:str) -> None:
    k_max=21
    costs=[]
    for i in range(1, k_max):
        kmeans = K_Means(k=i)
        kmeans.fit(embeddings)
        cost = kmeans.getCost()
        costs.append(float(cost))

    k_values = [int(i) for i in range(1, k_max)]
    plt.figure(figsize=(8,6))
    plt.plot(k_values, costs, marker='o', ms=5)
    plt.xticks(np.arange(min(k_values), max(k_values)+1, 1))  # Step size 1 for integers
    plt.grid(False)
    plt.xlabel('Number of Clusters')
    plt.ylabel('Within-Cluster Sum of Squares(WCSS)')
    plt.savefig(outPath)
    # plt.show()

def calculate_aic_and_bic_score(num_components:int, X:np.ndarray, log_likelihood:float) -> tuple[float, float]:
        num_samples, num_features = X.shape
        num_parameters = num_components * (num_features + ((num_features * (num_features + 1)) // 2)) + num_components - 1
        aic_score = 2 * num_parameters - 2 * log_likelihood
        bic_score = num_parameters * np.log(num_samples) - 2 * log_likelihood
        return aic_score, bic_score

def plot_AIC_BIC_vs_k(embeddings:np.ndarray, outPath:str) -> None:
        k_max = 11
        aic_scores=[]
        bic_scores=[]
        cluster_values=np.arange(1,k_max)
        for i in range(1,k_max):
            gmm = GMM(num_components=i)
            gmm.fit(X=embeddings)
            aic, bic = calculate_aic_and_bic_score(i, embeddings, gmm.getLikelihood(embeddings))
            aic_scores.append(aic)
            bic_scores.append(bic)
            likelihood = gmm.getLikelihood(embeddings)

        plt.figure(figsize=(8, 6))

        plt.plot(cluster_values, aic_scores, label='AIC score', marker='o')
        plt.plot(cluster_values, bic_scores, label='BIC score', marker='o')

        plt.title('AIC and BIC v/s Number of clusters')
        plt.xlabel('Number of Clusters')
        plt.ylabel('AIC/BIC scores')
        plt.legend()
        plt.savefig(outPath)
        plt.show()


def read_word_embeddings_data() -> tuple[np.ndarray, np.ndarray]: 
    df = pd.read_feather('./data/external/word-embeddings.feather') #dim : 200x2 where second coloumn is 512 dim vector embedding
    data = df.to_numpy()
    embeddings = np.vstack(data[:,-1])
    labels = np.vstack(data[:,:1])
    embeddings = embeddings.astype(np.float64)
    return embeddings, labels

def z_score_normalization(data:np.ndarray) -> np.ndarray:
    std_dev = np.std(data, axis=0)
    mean = np.mean(data, axis=0)
    std_dev = np.where(std_dev < 1e-6, 1e-6, std_dev)
    normalized_data = (data - mean) / std_dev
    return normalized_data

def plot_scree_plot(data: np.ndarray, n_components:int, outPath:str):
    data = data - np.mean(data, axis=0)
    cov_mat = np.cov(data, rowvar=False) 
    eigen_values, eigen_vectors = np.linalg.eig(cov_mat)

    eigen_values = eigen_values.real
    eigen_vectors = eigen_vectors.real
    sorted_indices = np.argsort(eigen_values)[::-1]
    sorted_eigenvalues = eigen_values[sorted_indices]
    sorted_eigenvalues = sorted_eigenvalues[:n_components]

    plt.figure(figsize=(15, 12))
    plt.plot(np.arange(1, n_components + 1), sorted_eigenvalues, 'o-', color='b', label='Explained Variance')
    plt.title('Scree Plot')
    plt.xlabel('Principal Component')
    plt.ylabel('Corresponding Eigenvalue')
    plt.xticks(np.arange(1, n_components + 1))
    plt.tight_layout()
    plt.savefig(outPath)
    # plt.show()

def plot_dendrogram(data:np.ndarray, dist_metric, dist_method):
    linkage_matrix = hc.linkage(y=data, method=dist_method, metric=dist_metric)
    fig = plt.figure(figsize=(15, 12))
    dgm = hc.dendrogram(linkage_matrix)
    plt.subplots_adjust(left=0.10, right=0.90, top=0.90, bottom=0.10)
    plt.tight_layout()
    plt.savefig(f'hc_dgm_metric={dist_metric}_method={dist_method}')
    # plt.show()

def measure_inference_time(model, x_test) -> float:
    start_time = time.time()
    for x in x_test:
        x = x.reshape(1, -1)
        model.predict(x)
    end_time = time.time()
    return end_time - start_time

def plot_inference_time_for_different_models(x_train_red:np.ndarray, x_train:np.ndarray, y_train:np.ndarray, x_val:np.ndarray, x_val_red:np.ndarray) -> None:
    knn_complete_data = KNN(k=11,distance_method='manhattan')   
    knn_reduced_data = KNN(k=11,distance_method='manhattan')  

    knn_complete_data.fit(x_train, y_train)
    knn_reduced_data.fit(x_train_red, y_train)

    completed_data_time = measure_inference_time(knn_complete_data, x_val)
    reduced_data_time = measure_inference_time(knn_reduced_data, x_val_red)

    models = ['KNN on complete dataset', 'KNN on Reduced Dataset']
    inference_times = [completed_data_time, reduced_data_time]

    plt.figure(figsize=(8, 6))
    plt.bar(models, inference_times, color=['blue','orange'])

    plt.title('Inference Time of KNN models')
    plt.xlabel('KNN Model')
    plt.ylabel('Inference Time (seconds)')
    plt.savefig('figures/knn_models_inf_time.png')

#======================================3: K-Means============================================
DATA = "./data/external/"
df = feather.read_feather(DATA + "word-embeddings.feather")
# df dim : 200 x 2 where second coloumn is 512 dim vector embedding
data = df.to_numpy()
embeddings = np.vstack(data[:,-1])
labels = np.vstack(data[:,:1])
vit_embeddings = embeddings.astype(np.float64)

## 3.1 WCSS v/s k plot
# plot_k_vs_cost(embeddings=vit_embeddings, outPath='wcss_vs_k_plot_q1.png')

## Kmeans for Optimal Number of Clusters
k_kmeans1 = 7
kmeans = K_Means(k=k_kmeans1)
kmeans.fit(vit_embeddings)
cluster_assigments = kmeans.predict(vit_embeddings)
print_clusters(labels=labels, cluster_assignments=cluster_assigments)

#======================================4: GMM============================================
# plot_AIC_BIC_vs_k(vit_embeddings, 'gmm_aic_bic_plot.png')

gmm = GMM(num_components=10)
gmm.fit(X=vit_embeddings)

gmm_skl = GaussianMixture(n_components=10)
gmm_skl.fit(X=vit_embeddings)

### plotting AIC,BIC v/s K 
# plot_AIC_BIC_vs_k(embeddings=vit_embeddings,outPath='gmm_aic_bic_vs_k.png')

# -------------------- GMM using k_gmm1 ----------------
k_gmm1 = 1
gmm = GMM(num_components=k_gmm1)
gmm.fit(X=vit_embeddings)
likelihood = gmm.getLikelihood(embeddings)
membership_mat = gmm.getMembership()
cluster_assignments = np.argmax(membership_mat, axis=1)
print_clusters(labels=labels, cluster_assignments=cluster_assignments)



#======================================5: PCA============================================

## testing on 2D Dataset
df_2d = pd.read_csv('./data/interim/a2/test.csv')
data = df_2d.to_numpy()
labels = data[:,-1]
data = data[:, :2]

num_clusters = 3
gmm = GMM(num_components=num_clusters)
gmm.fit(X=data)
membership_probs = gmm.getMembership()
print(membership_probs)
print(gmm.getLikelihood(data))

##  plotting the clusters with max prob
cluster_assigns = np.argmax(membership_probs, axis=1)
plt.figure(figsize=(8, 6))
for j in range(num_clusters):  # assuming k=10
    cluster_points = data[cluster_assigns == j]
    plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'Cluster {j}')

plt.title('K-Means Clustering')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
# plt.savefig(f'gmm_clustering_k={num_clusters}.png')
plt.show()




#======================================6: PCA+CLustering============================================
embeddings, labels = read_word_embeddings_data()

## 6.1 : K-Means based on 2D Visualization
# k2 = 4
# kmeans = K_Means(k=k2)
# kmeans.fit(features=embeddings)
# cluster_assignments = kmeans.predict(data_features=embeddings) 
# print_clusters(labels, cluster_assignments)

# ## 6.2 : PCA + K-Means Clustering
# plot_scree_plot(embeddings, n_components=50, outPath='embeddings_scree_plot_6_2.png')
# opt_dims = 5
# pca = PCA(num_components=opt_dims)
# pca.fit(embeddings)
# transform_feats = pca.transform(embeddings)

# plot_k_vs_cost(embeddings=embeddings, outPath='pca_clustering_wcss_vs_k_6_2.png')
# k_kmeans3 = 3
# kmeans = K_Means(k=k_kmeans3)
# kmeans.fit(features=transform_feats)
# cluster_assignments = kmeans.predict(data_features=transform_feats) 
# print_clusters(labels, cluster_assignments)

# ## 6.3 : GMM based on 2D Visualization
# gmm = GMM(num_components=k2)
# gmm.fit(X=embeddings)
# cluster_memberships = gmm.getMembership()

# ## 6.4 : PCA + GMMs
# plot_AIC_BIC_vs_k(embeddings=transform_feats, outPath='pca_gmm_aic_bic_vs_k_6_4.png')
# k_kgmm3 = 1
# gmm = GMM(num_components=k_kgmm3)
# gmm.fit(X=embeddings)
# cluster_memberships = gmm.getMembership()


#======================================8: Herarcy CLustering===========================================
print("==================HC=================")
k_best1 = 6
k_best2 = 2

linkage_matrix = hc.linkage(y=embeddings, method='ward', metric='euclidean')

clusters_hc_kbest1 = hc.fcluster(linkage_matrix, t=k_best1, criterion='maxclust')
clusters_hc_kbest2 = hc.fcluster(linkage_matrix, t=k_best2, criterion='maxclust')

print_clusters(labels=labels, cluster_assignments=clusters_hc_kbest1)

# dist_metrics=['euclidean', 'cosine', 'minkowski']
# dist_methods=['single', 'complete', 'average', 'weighted', 'centroid', 'median', 'ward']

# for mthd in dist_methods:
#     if mthd == 'centroid' or mthd == 'median' or mthd == 'ward':
#         plot_dendrogram(data=embeddings, dist_metric='euclidean', dist_method=mthd)
#     else:
#         for met in dist_metrics:
#             plot_dendrogram(data=embeddings, dist_metric=met, dist_method=mthd)

#======================================9: KNN+PCA===========================================
## ploting scree plot for spotify dataset
df = pd.read_csv('./data/external/spotify.csv')
df = df.drop_duplicates(subset='track_id', keep='first')
df = df.drop(columns=['Unnamed: 0','track_id','artists','album_name','track_name'])
data = df.to_numpy()
np.random.shuffle(data)
features = data[:, :-1].astype(np.float64)
labels = data[:,-1]
classes_list = np.unique(labels)

norm_features = z_score_normalization(features)
plot_scree_plot(norm_features, n_components=15, outPath='spotify_screen_plot.png')  

## Dimensionality Reduction on Spotify Dataset
opt_dims = 5
pca = PCA(num_components=opt_dims)
pca.fit(norm_features)
transformed_features = pca.transform(norm_features)

train_ratio = 0.8
val_ratio = 0.2
train_size = int(train_ratio * len(data))
val_size = int(val_ratio * len(data))

x_train = transformed_features[:train_size]
y_train = labels[:train_size]

x_val = transformed_features[train_size:train_size + val_size]
y_val = labels[train_size:train_size + val_size]

k_best = 11
best_dist_metric = 'manhattan'
knn = KNN(k=k_best, distance_method=best_dist_metric)
knn.fit(x_train, y_train)
y_pred = np.array([knn.predict(x) for x in x_val])

evaluation = Model_Evaluation(y_true=y_val, y_pred=y_pred, classes_list=classes_list)
print(f'Accuracy : {evaluation.accuracy_score()}')
print(f'Precision (macro) : {evaluation.precision_score(method="macro")}')
print(f'Recall (macro): {evaluation.recall_score(method="macro")}')
print(f'F1-Score (macro): {evaluation.f1_score(method="macro")}')
print(f'Precision (micro) : {evaluation.precision_score(method="micro")}')
print(f'Recall (micro): {evaluation.recall_score(method="micro")}')
print(f'F1-Score (micro): {evaluation.f1_score(method="micro")}')


x_train_org = norm_features[:train_size]
x_val_org = norm_features[train_size:train_size + val_size]
# plot_inference_time_for_different_models(x_train=x_train_org, x_train_red=x_train, y_train=y_train, x_val=x_val_org, x_val_red=x_val)