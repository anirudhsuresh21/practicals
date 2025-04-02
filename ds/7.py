import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

# Load and prepare data
data = pd.read_csv("C:\\Niraj\\Practical\\Sem 6\\DS\\program\\7\\dataset.csv")
features = ["Relative Compactness", "Surface Area", "Wall Area", "Roof Area", 
           "Overall Height", "Orientation", "Glazing Area", "Glazing Area Distribution", 
           "Heating Load", "Cooling Load"]
X = StandardScaler().fit_transform(data[features])

# Elbow Method
inertias = [KMeans(n_clusters=k, random_state=42).fit(X).inertia_ for k in range(1, 11)]
plt.plot(range(1, 11), inertias, 'bo-')
plt.title('Elbow Method')
plt.show()

# Silhouette Analysis
for k in range(2, 5):
    score = silhouette_score(X, KMeans(n_clusters=k, random_state=42).fit_predict(X))
    print(f"Silhouette Score for k={k}: {round(score * 100, 2)}%")

# Final Clustering and Visualization
kmeans = KMeans(n_clusters=3, random_state=42)
labels = kmeans.fit_predict(X)

# PCA Visualization
X_pca = PCA(n_components=2).fit_transform(X)
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], 
           marker='x', color='red', s=200)
plt.title('Clusters Visualization')
plt.show()
