# K-Means Clustering and Gaussian Mixture Model Analysis

## Overview
This project implements K-means clustering and Gaussian Mixture Models (GMM) using the PySpark MLlib library. The analysis focuses on evaluating the clustering performance by calculating silhouette scores across varying values of K (number of clusters) and comparing the results of K-means with GMM.

## Requirements
- Databricks environment or a local Spark installation.
- Python 3.x
- PySpark
- Matplotlib (for plotting)

## Dataset
This implementation uses the Iris dataset, which is a classic dataset for clustering tasks. The dataset contains measurements of iris flowers from three different species.

## Installation Instructions
1. **Clone the Repository (if applicable)**:
   ```bash
   git clone https://github.com/venkatbiyyapu/K-Means-GMM-Clustering.git
   cd K-Means-GMM-Clustering
   ```

2. **Set Up the Databricks Environment**:
   - Log in to your Databricks account or set up your local Spark environment.

3. **Upload the Iris Dataset**:
   - If using Databricks, upload the `Iris.csv` file to the `/FileStore/tables/` directory.

## Running the Code
1. **Start a New Notebook**:
   - Open a new notebook in Databricks or your local Spark IDE.

2. **Copy the Code**:
   - Upload the code from the repo named KNN-GMM.py

3. **Run the Notebook**:
   - Execute the cells in your notebook sequentially to perform the clustering and visualize the silhouette scores.

## Analysis
### Part A: K-means Clustering
- **Silhouette Score Graph**: The graph displays the silhouette scores for K values ranging from 2 to 10.
- **Choosing K**: A good value for K is typically chosen based on the peak silhouette score. Analyze the graph to determine where the silhouette score is maximized, indicating well-defined clusters.

### Part B: Model Comparison
- **Silhouette Scores**: The silhouette scores for both K-means and GMM are printed. 
- **Performance Analysis**: Compare the two scores to determine which model has better clustering performance. A higher silhouette score indicates that the clusters are well-separated and distinct.

### Conclusion
Based on the silhouette scores obtained from both models, conclude which clustering approach is more effective for the dataset. Provide reasoning based on the scores observed.

## Output
The output of the program includes:
- A plot of silhouette scores against different values of K.
- Silhouette scores for K-means and Gaussian Mixture Model for the chosen K.
