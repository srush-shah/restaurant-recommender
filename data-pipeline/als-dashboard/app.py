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
from collections import Counter

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
        # Define the container and file path in the object storage
        container = "object-persist-project23"
        file_path = "als/training_data.csv"
        
        st.info(f"Loading data from chi_tacc:{container}/{file_path}")
        
        # First, check if rclone config is available and list remotes
        check_cmd = ["rclone", "listremotes"]
        check_process = subprocess.run(check_cmd, capture_output=True, text=True)
        
        if check_process.returncode != 0:
            st.error(f"Failed to list rclone remotes: {check_process.stderr}")
            return None
        
        st.info(f"Available rclone remotes: {check_process.stdout}")
        
        # Check if chi_tacc remote is configured
        if "chi_tacc:" not in check_process.stdout:
            st.error("chi_tacc remote is not configured in rclone")
            st.info("Please check rclone configuration. You can configure it using 'rclone config'")
            return None
        
        # List contents of container to verify access
        list_cmd = ["rclone", "ls", f"chi_tacc:{container}/als"]
        list_process = subprocess.run(list_cmd, capture_output=True, text=True)
        
        if list_process.returncode != 0:
            st.error(f"Failed to list container contents: {list_process.stderr}")
            st.info(f"Command output: {list_process.stdout}")
            return None
        
        st.info(f"Files in container: {list_process.stdout}")
        
        # Instead of loading the entire file at once, we'll use pandas chunking
        # First, we need to get a file size estimate
        size_cmd = ["rclone", "size", f"chi_tacc:{container}/{file_path}"]
        size_process = subprocess.run(size_cmd, capture_output=True, text=True)
        
        if size_process.returncode != 0:
            st.warning(f"Couldn't get file size, proceeding anyway: {size_process.stderr}")
        else:
            st.info(f"File size information: {size_process.stdout}")
        
        # Create a temporary file to store a limited amount of data
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.csv')
        temp_path = temp_file.name
        temp_file.close()
        
        # Now run rclone cat and pipe to the temporary file
        # Using a subprocess with pipes to avoid loading everything into memory
        st.info("Starting to download data in chunks...")
        
        cat_cmd = ["rclone", "cat", f"chi_tacc:{container}/{file_path}"]
        with open(temp_path, 'w') as f:
            process = subprocess.Popen(cat_cmd, stdout=subprocess.PIPE, text=True)
            
            # Set up progress tracking
            progress_bar = st.progress(0)
            line_count = 0
            total_rows = 10001  # Header + 10000 rows
            
            # Read and write header
            header = process.stdout.readline()
            f.write(header)
            line_count += 1
            
            # Process data line by line up to 10,000 rows
            while line_count < total_rows:
                line = process.stdout.readline()
                if not line:  # End of file
                    break
                    
                f.write(line)
                line_count += 1
                
                # Update progress
                if line_count % 100 == 0:
                    progress_bar.progress(min(line_count / total_rows, 1.0))
                    
            # Terminate the process if we've reached our limit
            if line_count >= total_rows:
                process.terminate()
            else:
                process.wait()
                
            progress_bar.progress(1.0)
        
        st.success(f"Successfully loaded {line_count-1} rows (limited to 10,000)")
        
        # Now read the temporary file with pandas
        df = pd.read_csv(temp_path)
        
        # Clean up
        os.unlink(temp_path)
        
        # Display information about the loaded data
        st.info(f"Data loaded with {len(df)} rows and {len(df.columns)} columns")
        return df

    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        st.exception(e)  # This will display the full traceback
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
        ["Data Overview", "Data Quality", "Stars Analysis", "Cities Analysis", "Distribution Analysis", "Correlation Analysis"]
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

    # Stars Analysis
    elif page == "Stars Analysis":
        st.header("Restaurant Ratings Analysis")
        
        # Check if 'stars' column exists
        if 'stars' in df.columns:
            # Basic statistics for stars
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Average Rating", f"{df['stars'].mean():.2f}")
            with col2:
                st.metric("Most Common Rating", f"{df['stars'].mode().iloc[0]}")
            with col3:
                st.metric("Rating Variance", f"{df['stars'].var():.2f}")
            
            # Stars distribution
            st.subheader("Distribution of Star Ratings")
            
            # Create a histogram with customized bins
            fig = px.histogram(
                df, 
                x='stars',
                nbins=10,
                title='Distribution of Star Ratings',
                labels={'stars': 'Rating', 'count': 'Number of Reviews'},
                opacity=0.8,
                color_discrete_sequence=['#ff6b6b']
            )
            
            # Add a vertical line for the mean
            fig.add_vline(
                x=df['stars'].mean(), 
                line_dash="dash", 
                line_color="green",
                annotation_text=f"Mean: {df['stars'].mean():.2f}",
                annotation_position="top right"
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Rating counts table
            st.subheader("Rating Counts")
            rating_counts = df['stars'].value_counts().sort_index().reset_index()
            rating_counts.columns = ['Rating', 'Count']
            rating_counts['Percentage'] = (rating_counts['Count'] / rating_counts['Count'].sum() * 100).round(2)
            
            # Show table and pie chart side by side
            col1, col2 = st.columns(2)
            
            with col1:
                st.dataframe(rating_counts)
            
            with col2:
                fig = px.pie(
                    rating_counts, 
                    values='Count', 
                    names='Rating',
                    title='Rating Distribution',
                    hole=0.4,
                    color_discrete_sequence=px.colors.sequential.RdBu
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Ratings over time (if date column exists)
            if 'date' in df.columns:
                st.subheader("Ratings Trends Over Time")
                
                # Convert date column to datetime
                df['date'] = pd.to_datetime(df['date'])
                df['year'] = df['date'].dt.year
                df['month'] = df['date'].dt.month
                
                # Group by year and month to calculate average rating
                time_trend = df.groupby(['year', 'month'])['stars'].mean().reset_index()
                time_trend['year_month'] = time_trend['year'].astype(str) + '-' + time_trend['month'].astype(str)
                
                fig = px.line(
                    time_trend, 
                    x='year_month', 
                    y='stars',
                    title='Average Rating by Month',
                    labels={'stars': 'Average Rating', 'year_month': 'Year-Month'},
                    markers=True
                )
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.error("Stars column not found in the dataset.")
    
    # Cities Analysis
    elif page == "Cities Analysis":
        st.header("Cities Analysis")
        
        # Check if the city column exists
        city_column = None
        for col in ['city', 'City', 'business_city']:
            if col in df.columns:
                city_column = col
                break
        
        if city_column:
            # Count frequency of each city
            city_counts = df[city_column].value_counts().reset_index()
            city_counts.columns = ['City', 'Count']
            
            # Calculate percentage
            city_counts['Percentage'] = (city_counts['Count'] / city_counts['Count'].sum() * 100).round(2)
            
            # Display top 10 cities
            st.subheader("Top Cities by Number of Restaurants")
            top_cities = city_counts.head(10)
            
            fig = px.bar(
                top_cities,
                x='City',
                y='Count',
                title='Top 10 Cities by Number of Restaurants',
                color='Count',
                color_continuous_scale=px.colors.sequential.Viridis
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Show cities with the highest and lowest average ratings if stars column exists
            if 'stars' in df.columns:
                st.subheader("Cities by Average Rating")
                
                # Group by city and calculate average rating
                city_ratings = df.groupby(city_column)['stars'].agg(['mean', 'count']).reset_index()
                city_ratings.columns = ['City', 'Average Rating', 'Review Count']
                
                # Filter cities with more than a minimum number of reviews
                min_reviews = st.slider("Minimum number of reviews", 5, 100, 20)
                filtered_cities = city_ratings[city_ratings['Review Count'] >= min_reviews]
                
                # Sort by average rating
                top_rated = filtered_cities.sort_values('Average Rating', ascending=False).head(10)
                lowest_rated = filtered_cities.sort_values('Average Rating').head(10)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("#### Top Rated Cities")
                    st.dataframe(top_rated)
                
                with col2:
                    st.markdown("#### Lowest Rated Cities")
                    st.dataframe(lowest_rated)
                
                # Map view if coordinates are available
                if 'latitude' in df.columns and 'longitude' in df.columns:
                    st.subheader("Restaurant Locations")
                    
                    # Sample data to avoid overwhelming the map
                    map_sample = df.sample(min(1000, len(df)))
                    
                    fig = px.scatter_mapbox(
                        map_sample, 
                        lat='latitude', 
                        lon='longitude',
                        color='stars',
                        size_max=15, 
                        zoom=3,
                        mapbox_style="carto-positron",
                        title="Restaurant Locations"
                    )
                    st.plotly_chart(fig, use_container_width=True)
        else:
            st.error("No city column found in the dataset.")

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