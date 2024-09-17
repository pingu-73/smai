<center>

# **Assignment 2 Report**

</center>

## **Table of Contents**
3. [K-Means](#KMeans)

4. [GMM](#GMM)

5. [PCA](#PCA)
 
6. [PCA + K-Means](#PCAKMeans)

7. [Cluster Analysis](#ClusterAnalysis)
  
8. [Hierarchical Clustering](#HC)
 
9. [Nearest Neighbor Search - Spotify Dataset](#Spotify)

---

<p id = "KMeans"> </p>

<p id = "ElbowPlot512"> </p>

## **3 Optimal Number of Clusters for 512 dimensions using K-Means**

<center>

![Elbow Plot](../2/figures/wcss_vs_k_plot_q1.png)

*Figure 1: Elbow plot for the original 512 dimensional dataset*

</center>

**Elbow plot** helps in determining the optimal number of clusters for K-means clustering by plotting the within-cluster sum of squares (WCSS) against number of clusters. The optimal `K` is typically found at the "elbow" point, where adding more clusters yields only marginal improvements.

$$ \text{WCSS} = \sum_{i=1}^K \sum_{x \in C_i} \| x - \mu_i \|^2 $$

where:
- $K$ is the number of clusters,
- $C_i$ is the set of points in cluster $i$,
- $\mu_i$ is the centroid of cluster $i$,
- $x$ is a data point in cluster $i$.

> From the plot we can see that the elbow comes around $k = 7$ hence $k_{kmeans1} = 7$

>Note: Not always the best method: The elbow method might <span style="color: red;"><u>not be suitable</u></span> for all datasets, especially for those with <span style="color: red;"><u>high dimensionality</u></span> or clusters of irregular shapes. Hence if you see the graph it is very difficult to identify the elbow point.
---

<p id="GMM"></p>


## **4 GMM**
>The implemented GMM class doesn't work for high dimension data whereas  sckit-learn's GMM class applies some sophesticated techniques/algorithms to handle the case when covariance matrix leads to singular matrix. <span style="color: red;"><u>My model works well and in parallel to the sklearn's model for lower dimentional data.</u></span>
<center>

![Elbow Plot](../2/figures/gmm_aic_bic_vs_k.png)
</center>

**Akaike Information Criterion (AIC)** and **Bayesian Information Criterion (BIC)** are used for model selection in Gaussian Mixture Models (GMMs) to <span style="color: green;">balance goodness of fit</span> with <span style="color: red;">model complexity</span>.

- **AIC**: Measures the relative quality of a statistical model, <u>penalizing for the number of parameters</u> to <span style="color: green;">avoid overfitting</span>.
  $$\text{AIC} = 2k - 2 \ln(\hat{L})$$
  where $k$ is the number of parameters and $\hat{L}$ is the maximum likelihood of the model.

- **BIC**: Similar to AIC but with a <u>stronger penalty for model complexity</u>, suitable for larger sample sizes.
  $$\text{BIC} = \ln(n)k - 2 \ln(\hat{L})$$
  where $n$ is the number of observations and $k$ is the number of parameters.

Both criteria help in selecting the model with the best trade-off between fit and complexity. For selecting the suitable value of $k$ we want to <span style="color: green;">minimise both AIC and BIC.</span>

> From the plot we can see that both AIC and BIC are minimum for $k = 1$, hence $k_{gmm1} = 1$


---

<p id="PCA"></p>

## **5 PCA**
<center>

![2dplot](../2/figures/pca_2d.png)
![3d plot](../2/figures/pca_3d.jpeg)
</center>

---

<p id="PCAKMeans"></p>


## **6 PCA + K-Mean**

### **6.2**
**Scree Plot for determining optimal number of dim**
<center>

![screeplot](../2/figures/spotify_screen_plot.png)

![pca clustering wcss vs k](../2/figures/pca_clustering_wcss_vs_k_6_2.png)
> From  we can see that the graph flattens out starting from PC 5 hence we take optimal number of dimensions to be 4
</center>

### **6.4 PCA + GMM**
<center>

![pca+gmm](../2/figures/pca_gmm_aic_bic_vs_k_6_4.png)
Fig: AIC, BIC are reduced dataset with dim=4
> From the plot we can see that both AIC abnd BIC are min for k~=4, hence $k_{gmm3}=1$
</center> 

<p id="ClusterAnalysis"></p>


## 7.1 K-Means Cluster Analysis
> On manual inspection of word clouds of clusters for different values of k as well as on inspecting the inertial (WCSS) and silhouette scores for K-Means clustering we can conclude that $k_{kmeans}=3$ since there is not much difference in inertia (considering 512 dimensions) and also it gives the highest silhouette score

---



## 7.2 GMM Cluster Analysis


> On manual inspection of word clouds of clusters for different values of k as well as on inspecting the inertial (WCSS) and silhouette scores for GMM clustering we can conclude that $k_{gmm}=4$ since it gives the lowest inertial and highest silhouette score.



## 7.3 GMM and K-Means Comparison
> On manual inspection of word clouds of clusters formed for k_kmeans and k_gmm and by visually inspecting the clusters formed (fig 22 and 23) we see that K-Means produces better clusters (based on similarity between clusters and distance betweeen individual clusters)

---

<p id="HC"> </p>

## 8 Hierarchical Clustering
Linkage Methods in Hierarchical Clustering:
- Single Linkage: Minimum distance between points in two clusters.

$$d_{single}(C_i, C_j) = \min_{x \in C_i, y \in C_j} d(x, y)$$

- Complete Linkage: Maximum distance between points in two clusters.

$$d_{complete}(C_i, C_j) = \max_{x \in C_i, y \in C_j} d(x, y)$$

- Median Linkage: Distance between median points of two clusters.

$$d_{median}(C_i, C_j) = \|\tilde{x}_i - \tilde{x}_j\|^2$$

where $\tilde{x}_i$ is the median of cluster $C_i$

- Average Linkage: Average distance between all pairs of points in two clusters.

$$d_{average}(C_i, C_j) = \frac{1}{|C_i||C_j|} \sum_{x \in C_i} \sum_{y \in C_j} d(x, y)$$

- Ward's Method: Minimizes the increase in total within-cluster variance after merging.

$$d_{ward}(C_i, C_j) = \sqrt{\frac{2|C_i||C_j|}{|C_i|+|C_j|}} \|\bar{x}_i - \bar{x}_j\|$$

where $\bar{x}_i$ is the centroid of cluster $C_i$

- Centroid Linkage: Distance between centroids of two clusters.

$$d_{centroid}(C_i, C_j) = \|\bar{x}_i - \bar{x}_j\|^2$$

where $\bar{x}_i$ is the centroid of cluster $C_i$

<p id="Dendrograms"> </p>

#### <u>Below are all the dendrograms produced:</u>

<center>

![](../2/figures/hc_dgm_metric=cosine_method=average.png)
![](../2/figures/hc_dgm_metric=cosine_method=complete.png)
![](../2/figures/hc_dgm_metric=cosine_method=single.png)
![](../2/figures/hc_dgm_metric=cosine_method=weighted.png)
![](../2/figures/hc_dgm_metric=euclidean_method=average.png)
![](../2/figures/hc_dgm_metric=euclidean_method=centroid.png)
![](../2/figures/hc_dgm_metric=euclidean_method=complete.png)
![](../2/figures/hc_dgm_metric=euclidean_method=median.png)
![](../2/figures/hc_dgm_metric=euclidean_method=single.png)
![](../2/figures/hc_dgm_metric=euclidean_method=ward.png)
![](../2/figures/hc_dgm_metric=euclidean_method=weighted.png)
![](../2/figures/hc_dgm_metric=minkowski_method=average.png)
![](../2/figures/hc_dgm_metric=minkowski_method=complete.png)
![](../2/figures/hc_dgm_metric=minkowski_method=single.png)
![](../2/figures/hc_dgm_metric=minkowski_method=weighted.png)

</center>

> On inspecting all the above dendrograms we can see that the best linkage method is `ward`


---

<p id="Spotify"></p>

## Nearest Neighbor Search - Spotify Dataset



<center>

![Scree Plot of Spotify Dataset](../2/figures/spotify_screen_plot.png)

*Figure: Scree plot of the complete original Spotify Dataset*

</center>

> From the Scree plot we can clearly see that the plot flattens out after k = 4, hence we take the optimal number of dimensions to reduce to as 3

<p id="Evaluation"></p>

### KNN on reduced dataset using the best_k and best_metric obtained in A1

Metrics obtained for the complete spotify dataset in A1 are as follows:
<center>

![](../2/figures/model_evluations_q9.png)
</center>

### **Inference time plot**
<center>

![](../2/figures/knn_models_inf_time.png)
</center>

