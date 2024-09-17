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

## **3.2 Optimal Number of Clusters for 512 dimensions using K-Means**

<center>

![Elbow Plot](../2/figures/wcss_vs_k_plot_q1.png)

*Figure 1: Elbow plot for the original 512 dimensional dataset*

</center>

**Elbow plot** helps in determining the optimal number of clusters for K-means clustering by plotting the within-cluster sum of squares (WCSS) against number of clusters. The optimal `K` is typically found at the "elbow" point, where adding more clusters yields only marginal improvements.

$$ \text{WCSS} = \sum_{i=1}^K \sum_{x \in C_i} \| x - \mu_i \|^2 $$

where:
- $ K $ is the number of clusters,
- $ C_i $ is the set of points in cluster $ i $,
- $ \mu_i $ is the centroid of cluster $ i $,
- $ x $ is a data point in cluster $ i $.

> From the plot we can see that the elbow comes around $k = 7$ hence $k_{kmeans1} = 7$

>Note: Not always the best method: The elbow method might <span style="color: red;"><u>not be suitable</u></span> for all datasets, especially for those with <span style="color: red;"><u>high dimensionality</u></span> or clusters of irregular shapes. Hence if you see the graph it is very difficult to identify the elbow point.
---

<p id="GMM"></p>

<p id="GMM512"></p>

## **4.2 GMM**

### Issue Summary with Custom GMM Implementation

