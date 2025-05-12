import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
from io import StringIO
import json
from datetime import datetime
import subprocess
import os
import tempfile
import requests

# Set page config
st.set_page_config(
    page_title="Restaurant Recommender Data Quality Dashboard",
    page_icon="🍽️",
    layout="wide"
)

# Function to load data from Chameleon Object Storage using rclone
@st.cache_data
def load_data():
    try:
        # Create a temporary directory to store the data
        temp_dir = tempfile.mkdtemp()
        temp_file_path = os.path.join(temp_dir, "als_transformed_data.csv")
        
        # Use rclone to copy the file from the object storage
        container = "object-persist-project23"
        path_in_container = "als"  # Directory inside the container
        filename = "training_data.csv"  # Use the appropriate filename from your storage
        
        # Run rclone command to copy the file
        cmd = [
            "rclone", 
            "copy", 
            f"chi_tacc:{container}/{path_in_container}/{filename}",
            temp_file_path,
            "--progress"
        ]
        
        st.info(f"Downloading data from chi_tacc:{container}/{path_in_container}/{filename}")
        
        process = subprocess.run(
            cmd, 
            capture_output=True, 
            text=True
        )
        
        # Check if command was successful
        if process.returncode != 0:
            st.error(f"Failed to download data: HTTP status 404")
            st.error(f"Error details: NoSuchBucket")
            st.error(f"Command output: {process.stderr}")
            
            # Try listing the available files to help debug
            list_cmd = ["rclone", "ls", f"chi_tacc:{container}/{path_in_container}"]
            list_process = subprocess.run(list_cmd, capture_output=True, text=True)
            
            if list_process.returncode == 0:
                st.info("Available files in the container:")
                st.code(list_process.stdout)
            else:
                st.error(f"Could not list files in the container: {list_process.stderr}")
            
            return None
            
        # Read only first 20000 rows
        df = pd.read_csv(temp_file_path, nrows=20000)
        
        # Clean up the temporary file
        os.unlink(temp_file_path)
        os.rmdir(temp_dir)
        
        return df

    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        import traceback
        st.error(traceback.format_exc())
        return None

# Main title
st.title("🍽️ Restaurant Recommender Data Quality Dashboard")
st.markdown("---")

# Load the data
with st.spinner('Loading data from storage...'):
    df = load_data()

if df is not None:
    # Sidebar
    st.sidebar.header("Dashboard Navigation")
    page = st.sidebar.radio(
        "Select a page",
        ["Data Overview", "Data Quality", "Distribution Analysis", "Correlation Analysis"]
    )

    # Data Overview
    if page == "Data Overview":
        st.header("Data Overview")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Records", len(df))
        with col2:
            st.metric("Total Features", len(df.columns))
        with col3:
            st.metric("Memory Usage", f"{df.memory_usage().sum() / 1024**2:.2f} MB")
        
        st.subheader("Sample Data")
        st.dataframe(df.head())
        
        st.subheader("Data Information")
        buffer = StringIO()
        df.info(buf=buffer)
        st.text(buffer.getvalue())

    # Data Quality
    elif page == "Data Quality":
        st.header("Data Quality Analysis")
        
        # Missing values analysis
        st.subheader("Missing Values Analysis")
        missing_values = df.isnull().sum()
        missing_percentage = (missing_values / len(df)) * 100
        
        missing_df = pd.DataFrame({
            'Missing Values': missing_values,
            'Percentage': missing_percentage
        }).reset_index()
        
        fig = px.bar(
            missing_df,
            x='index',
            y='Percentage',
            title='Missing Values Percentage by Feature'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Duplicates analysis
        st.subheader("Duplicate Records Analysis")
        duplicates = len(df) - len(df.drop_duplicates())
        st.metric("Number of Duplicate Records", duplicates)
        
        # Data type distribution
        st.subheader("Data Types Distribution")
        dtype_counts = df.dtypes.value_counts()
        fig = px.pie(
            values=dtype_counts.values,
            names=dtype_counts.index.astype(str),
            title='Distribution of Data Types'
        )
        st.plotly_chart(fig, use_container_width=True)

    # Distribution Analysis
    elif page == "Distribution Analysis":
        st.header("Distribution Analysis")
        
        # Select numerical columns only
        numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
        
        # Feature selector
        selected_feature = st.selectbox(
            "Select a feature to analyze",
            numerical_cols
        )
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Histogram
            fig = px.histogram(
                df,
                x=selected_feature,
                title=f'Distribution of {selected_feature}'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Box plot
            fig = px.box(
                df,
                y=selected_feature,
                title=f'Box Plot of {selected_feature}'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Basic statistics
        st.subheader(f"Statistics for {selected_feature}")
        stats = df[selected_feature].describe()
        st.dataframe(stats)

    # Correlation Analysis
    elif page == "Correlation Analysis":
        st.header("Correlation Analysis")
        
        # Select numerical columns
        numerical_df = df.select_dtypes(include=['float64', 'int64'])
        
        # Compute correlation matrix
        corr_matrix = numerical_df.corr()
        
        # Create heatmap
        fig = px.imshow(
            corr_matrix,
            title='Correlation Matrix Heatmap',
            color_continuous_scale='RdBu'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Feature correlation selector
        st.subheader("Detailed Correlation Analysis")
        feature1 = st.selectbox("Select first feature", numerical_df.columns, key='feat1')
        feature2 = st.selectbox("Select second feature", numerical_df.columns, key='feat2')
        
        # Scatter plot
        fig = px.scatter(
            numerical_df,
            x=feature1,
            y=feature2,
            title=f'Scatter Plot: {feature1} vs {feature2}'
        )
        st.plotly_chart(fig, use_container_width=True)
else:
    st.error("Failed to load data. Please check your connection and try again.") 