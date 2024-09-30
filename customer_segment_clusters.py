import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import numpy as np
import pandas as pd
def perform_clustering(pca_ds,df1):
    # Perform Hierarchical Clustering
    hierarchical_model = AgglomerativeClustering(n_clusters=4)
    hierarchical_labels = hierarchical_model.fit_predict(pca_ds)

    # Store cluster labels in the DataFrame
    pca_ds["Hierarchical_Clusters"] = hierarchical_labels
    df1["Hierarchical_Clusters"] = hierarchical_labels

    # Calculate clustering scores
    silhouette = silhouette_score(pca_ds, hierarchical_labels)
    davies_bouldin = davies_bouldin_score(pca_ds, hierarchical_labels)
    calinski_harabasz = calinski_harabasz_score(pca_ds, hierarchical_labels)

    # Store important scores in a dictionary
    clustering_scores = {
        'Silhouette Score': silhouette,
        'Davies-Bouldin Index': davies_bouldin,
        'Calinski-Harabasz Index': calinski_harabasz
    }
    
    return clustering_scores

def plot_cluster_distribution(df1):
    plt.figure(figsize=(13, 8))
    pl = sns.countplot(x=df1['Hierarchical_Clusters'])
    pl.set_title('Distribution Of The Clusters')
    plt.show()
    

def plot_income_spending_relationship(df1):
    df1.head()
    plt.figure(figsize=(10, 8))
    #df1['Spent'] = df1[['MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts', 'MntGoldProds']].apply(lambda x: np.sum(x.dropna()), axis=1)
    #df1['Spent'] = df1['MntWines'] + df1['MntFruits'] + df1['MntMeatProducts'] + df1['MntFishProducts'] + df1['MntSweetProducts'] + df1['MntGoldProds']
    pl = sns.scatterplot(data=df1, x='Spent', y='Income',cmap='coolwarm',s=0)
    pl.set_title("Cluster's Profile Based on Income and Spending")
    pl.set_xlabel('Total Spending')
    pl.set_ylabel('Income')
    # Adjust axis limits for better clarity
    pl.xlim(0, 2500)  # Set x-axis limits
    pl.ylim(0, 160000)  # Set y-axis limits
    # Show legend
    pl.legend()
    plt.show() 
def describe_clusters(df1,cluster_column):
    cluster_descriptions = {}
    for label in range(0,4):
        selected_cluster = df1[df1[cluster_column] == label]
        cluster_descriptions[f'Cluster {label}'] = selected_cluster.describe().T
    return cluster_descriptions



if __name__ == "__main__":
    # Load PCA dataset and perform clustering
    with open('eda.pkl','rb') as f:    
        pca_ds = pickle.load(f)
    with open('dataframe.pkl','rb') as f:    
        df1 = pickle.load(f)
     # Add a new column 'Spent' to the DataFrame by summing the specified columns

    # Perform clustering
    clustering_scores = perform_clustering(pca_ds,df1)

    # Pickle the clustering scores
    with open('clustering_scores.pkl', 'wb') as f:
        pickle.dump(clustering_scores, f)

        
    # Plot cluster distribution and save the plot
    #plot_cluster_distribution(df1)
    

    # Plot income vs spending relationship and save the plot
    #plot_income_spending_relationship(df1)
    
    describe_clusters(df1)