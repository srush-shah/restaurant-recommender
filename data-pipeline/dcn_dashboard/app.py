import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import logging
from datetime import datetime
import time
import swiftclient
from config import SWIFT_CONFIG, DATA_PATHS, CONTAINER_NAME
import io

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set page configuration
st.set_page_config(
    page_title="Interactive Restaurant Recommendation Dashboard",
    page_icon="🍽️",
    layout="wide"
)

@st.cache_resource
def get_swift_connection():
    """Create Swift client connection"""
    return swiftclient.Connection(
        authurl=SWIFT_CONFIG['auth_url'],
        user=SWIFT_CONFIG['user_id'],
        key=SWIFT_CONFIG['application_credential_secret'],
        os_options={
            "region_name": SWIFT_CONFIG['region_name'],
            "application_credential_id": SWIFT_CONFIG['application_credential_id'],
            "application_credential_secret": SWIFT_CONFIG['application_credential_secret']
        },
        auth_version='3'
    )

@st.cache_data(ttl=3600)
def load_data(dataset_type='train'):
    """Load first chunk of data from object store"""
    try:
        # Get Swift connection
        conn = get_swift_connection()
        
        # Get object
        obj = conn.get_object(CONTAINER_NAME, DATA_PATHS[dataset_type])
        
        # Read CSV data into DataFrame
        df = pd.read_csv(io.BytesIO(obj[1]), nrows=10000)  # Read only first 10000 rows
        
        # Convert date column if it exists
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            
        return df
        
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        st.error(f"Error loading data: {str(e)}")
        return None
    finally:
        if 'conn' in locals():
            conn.close()

def calculate_statistics(df):
    """Calculate dataset statistics"""
    stats = {
        "Total Reviews": len(df),
        "Unique Users": df['user_id'].nunique(),
        "Unique Restaurants": df['business_id'].nunique(),
        "Average Rating": df['rating'].mean(),
        "Rating Std Dev": df['rating'].std()
    }
    return stats

def plot_rating_distribution(df, title):
    """Plot rating distribution"""
    fig = px.histogram(
        df, 
        x='rating',
        nbins=10,
        title=f"Rating Distribution - {title}",
        labels={'rating': 'Rating', 'count': 'Number of Reviews'},
        color_discrete_sequence=['#1f77b4']
    )
    fig.update_layout(bargap=0.1)
    return fig

def plot_reviews_per_user(df, title):
    """Plot reviews per user distribution"""
    reviews_per_user = df['user_id'].value_counts()
    fig = px.histogram(
        reviews_per_user,
        nbins=50,
        title=f"Reviews per User Distribution - {title}",
        labels={'value': 'Number of Reviews', 'count': 'Number of Users'},
        color_discrete_sequence=['#2ca02c']
    )
    fig.update_layout(bargap=0.1)
    return fig

def plot_temporal_distribution(df, title):
    """Plot temporal distribution of reviews"""
    if 'date' in df.columns:
        daily_reviews = df.groupby(df['date'].dt.date).size()
        fig = px.line(
            daily_reviews,
            title=f"Temporal Distribution of Reviews - {title}",
            labels={'index': 'Date', 'value': 'Number of Reviews'},
            color_discrete_sequence=['#ff7f0e']
        )
        return fig
    return None

def plot_city_distribution(df, title, top_n=20):
    """Plot city distribution"""
    if 'city' in df.columns:
        city_counts = df['city'].value_counts().head(top_n)
        fig = px.bar(
            city_counts,
            title=f"Top {top_n} Cities by Review Count - {title}",
            labels={'index': 'City', 'value': 'Number of Reviews'},
            color_discrete_sequence=['#9467bd']
        )
        fig.update_layout(xaxis_tickangle=-45)
        return fig
    return None

def main():
    st.title("Interactive Restaurant Recommendation Dashboard")
    st.write("This dashboard provides insights into the restaurant review dataset (First 10,000 records).")
    
    try:
        # Add a button to clear cache if needed
        if st.sidebar.button("Clear Cache and Reload Data"):
            st.cache_data.clear()
            st.experimental_rerun()
        
        # Dataset type selector
        dataset_type = st.sidebar.selectbox(
            "Select Dataset",
            options=['train', 'validation', 'production'],
            index=0
        )
        
        # Load data with better progress reporting
        with st.spinner("Loading data from object store..."):
            start_time = time.time()
            df = load_data(dataset_type=dataset_type)
            if df is None:
                st.error("Failed to load data")
                st.stop()
            load_time = time.time() - start_time
            st.sidebar.info(f"Data loaded in {load_time:.2f} seconds")
        
        # Add object store connection status
        st.sidebar.success("✅ Connected to Object Store")
        
        # Display data sizes
        st.sidebar.subheader("Dataset Size")
        st.sidebar.info(f"Total Reviews: {len(df):,} rows")
        
        # Display statistics
        st.subheader("Dataset Statistics")
        stats = calculate_statistics(df)
        cols = st.columns(4)
        for i, (metric, value) in enumerate(stats.items()):
            cols[i % 4].metric(metric, f"{value:,.2f}" if isinstance(value, float) else f"{value:,}")
        
        # Display plots
        col1, col2 = st.columns(2)
        
        with col1:
            st.plotly_chart(plot_rating_distribution(df, "First 10K Reviews"), use_container_width=True)
            city_plot = plot_city_distribution(df, "First 10K Reviews")
            if city_plot:
                st.plotly_chart(city_plot, use_container_width=True)
        
        with col2:
            st.plotly_chart(plot_reviews_per_user(df, "First 10K Reviews"), use_container_width=True)
            temporal_plot = plot_temporal_distribution(df, "First 10K Reviews")
            if temporal_plot:
                st.plotly_chart(temporal_plot, use_container_width=True)
        
        # Additional analysis
        st.subheader("Data Quality Analysis")
        
        # Check for missing values
        missing_values = df.isnull().sum()
        if missing_values.any():
            st.warning("Missing Values Detected")
            st.write(missing_values[missing_values > 0])
        else:
            st.success("No missing values found in the dataset!")
        
        # Display sample data
        st.subheader("Sample Data")
        st.dataframe(df.head(10))
        
    except Exception as e:
        st.error(f"Error: {str(e)}")
        logger.error(f"Error in main: {str(e)}", exc_info=True)

if __name__ == "__main__":
    main() 