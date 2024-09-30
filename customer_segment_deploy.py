import streamlit as st
import pandas as pd
from customer_segment_eda import perform_eda
from customer_segment_clusters import perform_clustering,plot_cluster_distribution,plot_income_spending_relationship,describe_clusters
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('agg')

# Define a function to perform EDA and clustering
def perform_analysis(file):
    # Perform EDA
    pca_ds,df1= perform_eda(file)
    # Perform clustering
    clustering_scores = perform_clustering(pca_ds,df1)
    st.write("Clustering Scores:")
    st.write(clustering_scores)
    st.write("the distribution of the clusters is ")
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.pyplot(plot_cluster_distribution(df1))
    #st.write("income vs spenting graph ")
    #st.set_option('deprecation.showPyplotGlobalUse', False)
    #st.pyplot(plot_income_spending_relationship(df1))
    cluster_descriptions = describe_clusters(df1, 'Hierarchical_Clusters')
    for cluster_label, description in cluster_descriptions.items():
      st.write(f"Cluster {cluster_label}:")
      st.table(description)


# Create a file uploader widget
file = st.file_uploader("Upload a file", type=["xlsx"])

if file is not None:
    # Call the perform_analysis function
    perform_analysis(file)
    
    

