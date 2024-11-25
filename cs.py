import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
import plotly.graph_objs as go
import warnings

# Ignore warnings
warnings.filterwarnings("ignore")

# Set Streamlit page configuration
st.set_page_config(page_title="Customer Segmentation", layout="wide")

# Load data
@st.cache
def load_data():
    file_path = r"C:\Users\rumjhum\Desktop\Customer Segmentation\Mall_Customers.csv"
    return pd.read_csv(file_path)

df = load_data()

# Sidebar for navigation
st.sidebar.title("Navigation")
options = st.sidebar.radio(
    "Select Section:",
    ["Data Overview", "Visualizations", "Clustering Analysis"]
)

# Data Overview
if options == "Data Overview":
    st.title("Customer Segmentation - Data Overview")
    st.write("### Dataset Preview")
    st.write(df.head())

    st.write("### Dataset Statistics")
    st.write(df.describe())

    st.write("### Missing Values")
    st.write(df.isnull().sum())

# Visualizations
elif options == "Visualizations":
    st.title("Visualizations")
    st.write("### Distribution Plots")
    fig, axes = plt.subplots(1, 3, figsize=(15, 6))
    for i, col in enumerate(['Age', 'Annual Income (k$)', 'Spending Score (1-100)']):
        sns.histplot(df[col], bins=20, kde=True, ax=axes[i])
        axes[i].set_title(f"Distribution of {col}")
    st.pyplot(fig)

    st.write("### Gender Count")
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.countplot(data=df, y="Gender", ax=ax)
    st.pyplot(fig)

    st.write("### Pairwise Relationships")
    fig, ax = plt.subplots(figsize=(10, 10))
    sns.pairplot(df[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']])
    st.pyplot(fig)

# Clustering Analysis
elif options == "Clustering Analysis":
    st.title("Clustering Analysis")
    
    # KMeans clustering on 'Age' and 'Spending Score'
    st.write("### Clustering on Age and Spending Score")
    X1 = df[['Age', 'Spending Score (1-100)']].values
    inertia = []
    for n in range(1, 11):
        model = KMeans(n_clusters=n, init='k-means++', random_state=111)
        model.fit(X1)
        inertia.append(model.inertia_)
    
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(range(1, 11), inertia, 'o-', label="Inertia")
    ax.set_title("Elbow Method")
    ax.set_xlabel("Number of Clusters")
    ax.set_ylabel("Inertia")
    ax.legend()
    st.pyplot(fig)

    kmeans = KMeans(n_clusters=4, init='k-means++', random_state=111)
    kmeans.fit(X1)
    labels1 = kmeans.labels_
    centroids1 = kmeans.cluster_centers_

    fig, ax = plt.subplots(figsize=(10, 7))
    scatter = ax.scatter(X1[:, 0], X1[:, 1], c=labels1, cmap='viridis', s=50, alpha=0.7)
    ax.scatter(centroids1[:, 0], centroids1[:, 1], s=200, c='red', marker='X')
    ax.set_title("Clusters (Age vs Spending Score)")
    ax.set_xlabel("Age")
    ax.set_ylabel("Spending Score (1-100)")
    st.pyplot(fig)

    # 3D Clustering Visualization
    st.write("### 3D Visualization of Clusters")
    X3 = df[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']].values
    kmeans3 = KMeans(n_clusters=5, init='k-means++', random_state=111)
    labels3 = kmeans3.fit_predict(X3)

    trace = go.Scatter3d(
        x=X3[:, 0],
        y=X3[:, 1],
        z=X3[:, 2],
        mode='markers',
        marker=dict(
            size=10,
            color=labels3,
            colorscale='Viridis',
            opacity=0.8
        )
    )

    layout = go.Layout(
        title="3D Clusters",
        scene=dict(
            xaxis_title="Age",
            yaxis_title="Annual Income (k$)",
            zaxis_title="Spending Score (1-100)"
        )
    )

    fig = go.Figure(data=[trace], layout=layout)
    st.plotly_chart(fig)
