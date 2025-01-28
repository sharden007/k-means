# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import json

# Step 1: Load the data from an external JSON file
def load_data(file_path):
    """
    Load customer data from a JSON file into a Pandas DataFrame.
    """
    try:
        with open(file_path, 'r') as file:
            data = json.load(file)
        return pd.DataFrame(data)
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        exit()

# Step 2: Preprocess the data
def preprocess_data(df):
    """
    Preprocess the data by selecting features and scaling them.
    """
    # Select numerical features for clustering
    features = ['Age', 'Frequency (for visits)', 'Coffee', 'Cold drinks', 
                'Jaws chip', 'Pastries', 'Juices', 'Sandwiches', 'cake']
    X = df[features]
    
    # Scale the features using StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled

# Step 3: Determine the optimal number of clusters using the Elbow Method
def determine_optimal_clusters(X):
    """
    Use the Elbow Method to determine the optimal number of clusters.
    """
    inertia = []
    K_range = range(1, 11)
    
    for k in K_range:
        kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
        kmeans.fit(X)
        inertia.append(kmeans.inertia_)
    
    # Plot the Elbow Curve
    plt.figure(figsize=(8, 5))
    plt.plot(K_range, inertia, marker='o')
    plt.title('Elbow Method for Optimal Clusters')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Inertia')
    plt.grid(True)
    plt.show()

# Step 4: Apply K-Means clustering
def apply_kmeans(X, n_clusters):
    """
    Apply K-Means clustering to the scaled data.
    """
    kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
    labels = kmeans.fit_predict(X)
    
    # Calculate silhouette score for evaluation
    silhouette_avg = silhouette_score(X, labels)
    
    print(f"Silhouette Score for {n_clusters} clusters: {silhouette_avg:.2f}")
    
    return labels

# Step 5: Save clustered data to a CSV file
def save_clustered_data(df, labels, output_file):
    """
    Save the original DataFrame with cluster labels to a CSV file.
    """
    df['Cluster'] = labels
    df.to_csv(output_file, index=False)
    print(f"Clustered data saved to {output_file}")

# Step 6: Visualize clusters (using PCA for dimensionality reduction if needed)
def visualize_clusters(X, labels):
    """
    Visualize clusters in a 2D space using PCA.
    """
    from sklearn.decomposition import PCA
    
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    
    plt.figure(figsize=(8, 6))
    
    for cluster in np.unique(labels):
        plt.scatter(
            X_pca[labels == cluster, 0], 
            X_pca[labels == cluster, 1], 
            label=f"Cluster {cluster}"
        )
    
    plt.title("Customer Segments (PCA Reduced)")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    
    plt.legend()
    plt.grid(True)
    plt.show()

# Main function to execute the workflow
def main():
    # Load data from external JSON file
    file_path = 'starbucks.json'  # Ensure this file exists in the same directory
    df = load_data(file_path)

    # Preprocess the data
    X_scaled = preprocess_data(df)

    # Determine the optimal number of clusters (via Elbow Method)
    determine_optimal_clusters(X_scaled)

    # Apply K-Means clustering with a chosen number of clusters (e.g., k=4)
    n_clusters = 4  # You can adjust this based on the Elbow Method results
    labels = apply_kmeans(X_scaled, n_clusters)

    # Save clustered data to a CSV file
    save_clustered_data(df, labels, 'clustered_customers.csv')

    # Visualize the clusters
    visualize_clusters(X_scaled, labels)

if __name__ == "__main__":
    main()
