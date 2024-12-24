# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
import tkinter as tk
from tkinter import filedialog, messagebox

# Function to load data
def load_data(filepath):
    data = pd.read_csv(filepath)
    return data

# Function to preprocess data (handle missing values and scale data)
def preprocess_data(data):
    numeric_cols = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']  # List numeric columns
    data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].mean())  # Fill missing values with column mean
    categorical_cols = ['Gender']  # List categorical columns
    for col in categorical_cols:
        data[col] = data[col].fillna(data[col].mode()[0])  # Fill missing categorical values with the mode (most frequent)
    
    # Normalize the data using StandardScaler
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data[numeric_cols])
    return scaled_data

# Elbow method for KMeans to determine optimal number of clusters
def elbow_method(data):
    inertia = []  # List to store inertia values (sum of squared distances)
    for k in range(1, 11):  # Try different numbers of clusters (1 to 10)
        kmeans = KMeans(n_clusters=k, random_state=42)  # KMeans model
        kmeans.fit(data)  # Fit the model to the data
        inertia.append(kmeans.inertia_)  # Inertia is the sum of squared distances from points to their cluster centers
    
    # Plot the Elbow curve
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, 11), inertia, marker='o')
    plt.title('Elbow Method')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Inertia')
    plt.show()

# Silhouette Score for determining cluster quality
def silhouette_analysis(data):
    scores = []  # List to store silhouette scores
    for k in range(2, 11):  # Try different numbers of clusters (2 to 10)
        kmeans = KMeans(n_clusters=k, random_state=42)  # KMeans model
        kmeans.fit(data)  # Fit the model to the data
        score = silhouette_score(data, kmeans.labels_)  # Calculate silhouette score
        scores.append(score)  # Append the score
    
    # Plot the Silhouette scores
    plt.figure(figsize=(8, 6))
    plt.plot(range(2, 11), scores, marker='o')
    plt.title('Silhouette Score for K-Means')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Silhouette Score')
    plt.show()

# Function to perform KMeans clustering
def apply_kmeans(data, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)  # Initialize KMeans model with n_clusters
    labels = kmeans.fit_predict(data)  # Apply KMeans and get cluster labels for each data point
    return labels, kmeans

# Function to perform DBSCAN clustering
def apply_dbscan(data, eps=0.5, min_samples=5):
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)  # Initialize DBSCAN model
    labels_dbscan = dbscan.fit_predict(data)  # Apply DBSCAN and get cluster labels
    return labels_dbscan

# Function to visualize KMeans clusters in 2D
def plot_kmeans_2d(data, labels):
    pca = PCA(n_components=2)  # Use PCA for dimensionality reduction (to 2D)
    pca_data = pca.fit_transform(data)  # Reduce dimensions of the data to 2D
    plt.figure(figsize=(8, 6))
    plt.scatter(pca_data[:, 0], pca_data[:, 1], c=labels, cmap='viridis', s=50)  # Scatter plot of clusters
    plt.title('KMeans Clusters (2D)')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.show()

# Function to visualize DBSCAN clusters in 2D
def plot_dbscan_2d(data, labels):
    pca = PCA(n_components=2)  # Use PCA for dimensionality reduction (to 2D)
    pca_data = pca.fit_transform(data)  # Reduce dimensions of the data to 2D
    plt.figure(figsize=(8, 6))
    plt.scatter(pca_data[:, 0], pca_data[:, 1], c=labels, cmap='plasma', s=50)  # Scatter plot of clusters
    plt.title('DBSCAN Clusters (2D)')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.show()

# GUI for customer segmentation app
class CustomerSegmentationApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Customer Segmentation using Clustering")
        
        self.filepath = None  # To store the file path
        self.n_clusters = 5  # Default number of clusters

        # Button to upload dataset
        self.upload_btn = tk.Button(root, text="Upload Dataset", command=self.upload_file, width=20, bg="blue", fg="white")
        self.upload_btn.pack(pady=20)

        # Button to perform K-means clustering
        self.kmeans_btn = tk.Button(root, text="Perform KMeans", command=self.perform_kmeans, width=20, bg="green", fg="white")
        self.kmeans_btn.pack(pady=20)

        # Button to perform DBSCAN clustering
        self.dbscan_btn = tk.Button(root, text="Perform DBSCAN", command=self.perform_dbscan, width=20, bg="purple", fg="white")
        self.dbscan_btn.pack(pady=20)

        # Label to display results
        self.result_label = tk.Label(root, text="", font=("Arial", 12))
        self.result_label.pack(pady=20)
    
    def upload_file(self):
    self.filepath = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
    if self.filepath:
        print(f"File loaded: {self.filepath}")  # Debugging line
        messagebox.showinfo("Success", "Dataset uploaded successfully!")
    else:
        messagebox.showwarning("Error", "No file selected!")


    def perform_kmeans(self):
        if not self.filepath:
            messagebox.showwarning("Error", "Please upload a dataset first!")
            return
        
        # Load and preprocess data
        data = load_data(self.filepath)
        scaled_data = preprocess_data(data)

        # Elbow method to determine optimal clusters
        elbow_method(scaled_data)
        
        # Silhouette analysis to check for cluster quality
        silhouette_analysis(scaled_data)
        
        # Apply KMeans (using a predefined number of clusters, e.g., 5)
        labels, model = apply_kmeans(scaled_data, self.n_clusters)

        # Visualize clusters in 2D
        plot_kmeans_2d(scaled_data, labels)
        self.result_label.config(text=f"KMeans Clusters Created with {self.n_clusters} clusters.")

    def perform_dbscan(self):
        if not self.filepath:
            messagebox.showwarning("Error", "Please upload a dataset first!")
            return
        
        # Load and preprocess data
        data = load_data(self.filepath)
        scaled_data = preprocess_data(data)

        # Apply DBSCAN
        labels_dbscan = apply_dbscan(scaled_data)

        # Visualize DBSCAN results in 2D
        plot_dbscan_2d(scaled_data, labels_dbscan)
        self.result_label.config(text="DBSCAN Clusters Created.")
