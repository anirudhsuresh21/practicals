import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

# Load and prepare data
data = pd.read_csv("diabetes.csv")

# Convert categorical variables to numeric
# data['famhist'] = data['famhist'].map({'Present': 1, 'Absent': 0})
# data['chd'] = data['chd'].map({'Si': 1, 'No': 0})

# Select numerical features for clustering
features = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
X = StandardScaler().fit_transform(data[features])

# Elbow Method
inertias = [KMeans(n_clusters=k, random_state=42).fit(X).inertia_ for k in range(1, 11)]
plt.plot(range(1, 11), inertias, 'bo-')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia')
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
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)
centers_pca = pca.transform(kmeans.cluster_centers_)

plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='viridis')
plt.scatter(centers_pca[:, 0], centers_pca[:, 1], marker='x', s=200, linewidths=3, color='red', label='Centroids')
plt.title('Clusters Visualization using PCA')
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
plt.legend()
plt.show()
