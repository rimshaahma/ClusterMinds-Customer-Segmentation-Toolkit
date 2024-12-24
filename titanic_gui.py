import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

class CustomerSegmentationApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Customer Segmentation with Clustering")

        self.filepath = None  # To store the selected file path

        # Button to upload CSV file
        self.upload_btn = tk.Button(root, text="Upload CSV", command=self.upload_csv, width=20, bg="blue", fg="white")
        self.upload_btn.pack(pady=20)

        # Button to perform KMeans clustering
        self.kmeans_btn = tk.Button(root, text="Perform KMeans Clustering", command=self.perform_kmeans, width=20, bg="green", fg="white")
        self.kmeans_btn.pack(pady=20)

        # Button to perform DBSCAN clustering
        self.dbscan_btn = tk.Button(root, text="Perform DBSCAN Clustering", command=self.perform_dbscan, width=20, bg="orange", fg="white")
        self.dbscan_btn.pack(pady=20)

        # Label to display results
        self.result_label = tk.Label(root, text="", font=("Arial", 12))
        self.result_label.pack(pady=20)

    def upload_csv(self):
        # Let the user choose the dataset file
        self.filepath = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if self.filepath:
            messagebox.showinfo("Success", "CSV file uploaded successfully!")
        else:
            messagebox.showwarning("Error", "No file selected!")

    def preprocess_data(self, data):
        """ Preprocess data by filling missing values and normalizing. """
        data = data.fillna(data.mean())  # Filling missing values with mean
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(data)
        return scaled_data

    def perform_kmeans(self):
        """ Perform KMeans clustering on the uploaded data. """
        if not self.filepath:
            messagebox.showwarning("Error", "Please upload a CSV file first!")
            return
        
        # Load data
        data = pd.read_csv(self.filepath)
        # Preprocess data
        scaled_data = self.preprocess_data(data)

        # Perform KMeans clustering
        kmeans = KMeans(n_clusters=3, random_state=42)
        kmeans.fit(scaled_data)
        labels = kmeans.labels_

        # Calculate the Silhouette score
        silhouette_avg = silhouette_score(scaled_data, labels)
        
        # Plot the clusters in 2D space
        pca = PCA(n_components=2)
        pca_data = pca.fit_transform(scaled_data)
        
        plt.figure(figsize=(8,6))
        plt.scatter(pca_data[:, 0], pca_data[:, 1], c=labels, cmap='viridis')
        plt.title("KMeans Clustering (Silhouette Score: {:.2f})".format(silhouette_avg))
        plt.xlabel("PCA Component 1")
        plt.ylabel("PCA Component 2")
        plt.show()

        # Display result
        result_text = f"KMeans Clustering performed\nSilhouette Score: {silhouette_avg:.4f}"
        self.result_label.config(text=result_text)

    def perform_dbscan(self):
        """ Perform DBSCAN clustering on the uploaded data. """
        if not self.filepath:
            messagebox.showwarning("Error", "Please upload a CSV file first!")
            return
        
        # Load data
        data = pd.read_csv(self.filepath)
        # Preprocess data
        scaled_data = self.preprocess_data(data)

        # Perform DBSCAN clustering
        dbscan = DBSCAN(eps=0.5, min_samples=5)
        dbscan_labels = dbscan.fit_predict(scaled_data)

        # Plot the clusters in 2D space
        pca = PCA(n_components=2)
        pca_data = pca.fit_transform(scaled_data)

        plt.figure(figsize=(8,6))
        plt.scatter(pca_data[:, 0], pca_data[:, 1], c=dbscan_labels, cmap='viridis')
        plt.title("DBSCAN Clustering")
        plt.xlabel("PCA Component 1")
        plt.ylabel("PCA Component 2")
        plt.show()

        # Display result
        unique_labels = set(dbscan_labels)
        result_text = f"DBSCAN Clustering performed\nUnique Labels: {len(unique_labels)}"
        self.result_label.config(text=result_text)

# Run the GUI application
if __name__ == "__main__":
    root = tk.Tk()
    app = CustomerSegmentationApp(root)
    root.mainloop()
