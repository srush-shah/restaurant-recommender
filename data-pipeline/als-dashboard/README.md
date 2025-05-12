# Restaurant Recommender Data Quality Dashboard

This Streamlit dashboard provides comprehensive data quality analysis for the ALS ETL transformed data from the Restaurant Recommender project.

## Features

- Data Overview
- Data Quality Analysis
- Distribution Analysis
- Correlation Analysis
- Interactive visualizations
- Sample data preview
- Missing values analysis
- Duplicate records detection
- Data type distribution

## Running with Docker (Recommended)

1. Create a `.streamlit/secrets.toml` file with your Chameleon credentials:

```toml
user_id = "YOUR_USER_ID"
application_credential_id = "APP_CRED_ID"
application_credential_secret = "APP_CRED_SECRET"
```

2. Build and run using Docker Compose:

```bash
docker-compose up --build
```

The dashboard will be available at `http://localhost:8501`

To stop the dashboard:

```bash
docker-compose down
```

## Running Locally (Alternative)

1. Create a virtual environment (optional but recommended):

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install required packages:

```bash
pip install -r requirements.txt
```

3. Create a `.streamlit/secrets.toml` file with your Chameleon credentials:

```toml
user_id = "YOUR_USER_ID"
application_credential_id = "APP_CRED_ID"
application_credential_secret = "APP_CRED_SECRET"
```

4. Run the dashboard:

```bash
streamlit run app.py
```

The dashboard will be available at `http://localhost:8501`

## Data Loading

The dashboard automatically loads the first 20,000 rows of the transformed data from Chameleon Object Storage for analysis. This limit is in place to ensure smooth performance while still providing meaningful insights.

## Notes

- Make sure you have proper access credentials for the Chameleon Object Storage
- The dashboard caches data to improve performance
- All visualizations are interactive and can be downloaded as PNG files
- When running with Docker, any changes to the secrets file will be immediately reflected in the container due to volume mounting
