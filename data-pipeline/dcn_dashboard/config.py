import os
from dotenv import load_dotenv

# Load environment variables from .env file if it exists
load_dotenv()

# Chameleon Cloud Swift configuration
SWIFT_CONFIG = {
    'auth_url': os.getenv('CC_AUTH_URL', 'https://chi.tacc.chameleoncloud.org:5000/v3'),
    'user_id': os.getenv('CC_USER_ID', '59b49300deb6f11832f7764d9f2c3451ba51e578bb556d58e02bf4c64bd89a31'),
    'application_credential_id': os.getenv('CC_APP_CRED_ID', 'aba2c756cb1649b6bd47c572f67a9528'),
    'application_credential_secret': os.getenv('CC_APP_CRED_SECRET', 'HnUZC1Dvhnw1A5-19LiDQORiWURpL1IflqK3tk506NXJp8Tb0HSW4m0V9Ch4zEhwV9WR7tTxiAPTcTGHMi2SKw'),
    'region_name': os.getenv('CC_REGION', 'CHI@TACC')
}

# Data file paths in object store - Updated to use ALS data
DATA_PATHS = {
    'train': 'als/training_data.csv',
    'validation': 'als/validation_data.csv',
    'production': 'als/production_data.csv'
}

# Container (bucket) name
CONTAINER_NAME = os.getenv('CC_CONTAINER_NAME', 'object-persist-project23') 