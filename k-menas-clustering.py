import numpy as np

# Data points
X = np.array([
    [2, 2],
    [4, 4],
    [6, 6],
    [8, 8]
])

# Number of clusters
K = 2

#Step 1: Initialize Centroids
centroids = np.array([
    [2, 2],
    [8, 8]
])

print("Initial Centroids:")
print(centroids)

def euclidean_distance(a, b):
    return np.sqrt(np.sum((a - b) ** 2))

for iteration in range(1, 6):
    print(f"\n--- Iteration {iteration} ---")
    
    clusters = {i: [] for i in range(K)}
    
    # Assignment step
    for point in X:
        distances = [euclidean_distance(point, centroid) for centroid in centroids]
        cluster_id = np.argmin(distances)
        clusters[cluster_id].append(point)
        print(f"Point {point} distances: {distances} -> Cluster {cluster_id+1}")
    
    new_centroids = []
    
    # Update step
    for i in clusters:
        cluster_points = np.array(clusters[i])
        new_centroid = cluster_points.mean(axis=0)
        new_centroids.append(new_centroid)
        print(f"Updated Centroid {i+1}: {new_centroid}")
    
    new_centroids = np.array(new_centroids)
    
    # Convergence check
    if np.allclose(centroids, new_centroids):
        print("\nCentroids unchanged. Convergence reached.")
        break
    
    centroids = new_centroids
