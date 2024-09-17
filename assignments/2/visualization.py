import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import seaborn as sns
import pyarrow as pa
import pyarrow.feather as feather
import warnings
warnings.filterwarnings('ignore')

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from models.k_means.k_means import K_Means

# -----------------------Visualizing Dataset-------------------------------
DATA = "./data/external/"
df = feather.read_feather(DATA + "word-embeddings.feather")
print(df.head())
print(df.info())
# OBDERVATION: looks like vit column contains vector representation of words
# df has 200 rows and 2 cols
# df.isnull().sum() # find missing values in dataset


vit_embeddings = np.array(df['vit'].tolist()) # converting vit column to numpy array

print(f"Shape of embeddings: {vit_embeddings.shape}")
vit_mean = np.mean(vit_embeddings, axis=0)
vit_std = np.std(vit_embeddings, axis=0)
vit_standardized = (vit_embeddings - vit_mean) / vit_std

cov_matrix = np.cov(vit_standardized.T)

eigen_values, eigen_vectors = np.linalg.eig(cov_matrix)

# Sort eigenvalues and eigenvectors
idx = eigen_values.argsort()[::-1]
eigen_values = eigen_values[idx]
eigen_vectors = eigen_vectors[:, idx]

# Transform data into 2D
vit_pca = np.dot(vit_standardized, eigen_vectors[:, :2])

# Create a new DataFrame with reduced dimensions
data_pca = pd.DataFrame(vit_pca, columns=['PC1', 'PC2'])
data_pca['word'] = df['words']

# Step 2: Visualize
plt.figure(figsize=(10, 7))
plt.scatter(data_pca['PC1'], data_pca['PC2'], alpha=0.7)
# for i in range(len(data_pca)):
#     plt.text(data_pca['PC1'][i], data_pca['PC2'][i], data_pca['word'][i], fontsize=9)
plt.title('2D Visualization of Word Embeddings using PCA')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.grid(True)
plt.tight_layout()
plt.show()
# plt.savefig("./assignments/2/figures/PCA-feathers.png")
# plt.close()

def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    return dot_product / (norm_vec1 * norm_vec2)

# Compute cosine similarities for all pairs of words
cosine_similarities = np.zeros((len(vit_embeddings), len(vit_embeddings)))
for i in range(len(vit_embeddings)):
    for j in range(len(vit_embeddings)):
        cosine_similarities[i][j] = cosine_similarity(vit_embeddings[i], vit_embeddings[j])

# Create a DataFrame for the heatmap
cosine_sim_df = pd.DataFrame(cosine_similarities, index=df['words'], columns=df['words'])

# Plot heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(cosine_sim_df, cmap='coolwarm', annot=False)
plt.title('Cosine Similarity Heatmap of Word Embeddings')
plt.tight_layout()
plt.show()
# plt.savefig("./assignments/2/figures/heatmap-feeather.png")
# plt.close()

plt.figure(figsize=(12, 6))
sns.boxplot(data=data_pca, orient='h')
plt.title('Boxplot of PCA Components')
plt.xlabel('Value')
plt.ylabel('Principal Components')
plt.grid(True)
plt.show()
# plt.savefig("./assignments/2/figures/boxplot-feather.png")