import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt

# Load and preprocess data
print("Loading data...")
data = pd.read_csv('rideshare_data_cleaned.csv')
features = ['price', 'distance', 'hour', 'surge_multiplier']
X = data[features]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 1. K-means Clustering
print("Running K-means...")
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans_clusters = kmeans.fit_predict(X_scaled)

# 2. Gaussian Mixture Model
print("Running Gaussian Mixture Model...")
gmm = GaussianMixture(n_components=3, random_state=42)
gmm_clusters = gmm.fit_predict(X_scaled)

# 3. PCA
print("Running PCA...")
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Save results
data['kmeans_cluster'] = kmeans_clusters
data['gmm_cluster'] = gmm_clusters
data['pca_1'] = X_pca[:, 0]
data['pca_2'] = X_pca[:, 1]

# Calculate cluster statistics
def get_cluster_stats(data, cluster_column):
    stats = data.groupby(cluster_column)[features].agg([
        'mean', 'std', 'min', 'max'
    ]).round(2)
    return stats

# Get statistics for each technique
kmeans_stats = get_cluster_stats(data, 'kmeans_cluster')
gmm_stats = get_cluster_stats(data, 'gmm_cluster')

# Print results
print("\n=== Analysis Results ===")

print("\nK-means cluster sizes:")
print(pd.Series(kmeans_clusters).value_counts().to_dict())

print("\nGMM cluster sizes:")
print(pd.Series(gmm_clusters).value_counts().to_dict())

print("\nPCA explained variance ratio:")
print(pca.explained_variance_ratio_)

# Save detailed statistics to CSV
kmeans_stats.to_csv('kmeans_cluster_stats.csv')
gmm_stats.to_csv('gmm_cluster_stats.csv')

# Create basic plots
plt.figure(figsize=(10, 6))
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=kmeans_clusters, cmap='viridis', alpha=0.5)
plt.title('K-means Clustering')
plt.xlabel('Price (scaled)')
plt.ylabel('Distance (scaled)')
plt.savefig('kmeans_plot.png')
plt.close()

plt.figure(figsize=(10, 6))
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=gmm_clusters, cmap='viridis', alpha=0.5)
plt.title('Gaussian Mixture Model Clustering')
plt.xlabel('Price (scaled)')
plt.ylabel('Distance (scaled)')
plt.savefig('gmm_plot.png')
plt.close()

plt.figure(figsize=(10, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.5)
plt.title('PCA: First Two Principal Components')
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
plt.savefig('pca_plot.png')
plt.close()

# Save all results to a new CSV
data.to_csv('rideshare_data_with_all_analysis.csv', index=False)

print("\nAll analysis completed! Check the saved CSV files and plots.")
print("1. rideshare_data_with_all_analysis.csv - Contains all original data with cluster assignments")
print("2. kmeans_cluster_stats.csv - Detailed statistics for K-means clusters")
print("3. gmm_cluster_stats.csv - Detailed statistics for GMM clusters")
print("4. Three visualization plots saved: kmeans_plot.png, gmm_plot.png, pca_plot.png")