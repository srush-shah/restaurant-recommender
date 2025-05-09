from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
from datetime import datetime
import os

def transform():
    os.system("python3 scripts/transform_yelp.py")

with DAG("yelp_offline_etl", start_date=datetime(2024, 1, 1), schedule_interval=None, catchup=False) as dag:
    
    extract_data = BashOperator(
        task_id="extract_data",
        bash_command="rclone copy chi_tacc:$RCLONE_CONTAINER/raw/ data/raw/ --progress"
    )

    transform_data = PythonOperator(
        task_id="transform_data",
        python_callable=transform
    )

    load_data = BashOperator(
        task_id="load_data",
        bash_command="bash scripts/upload_to_object_store.sh"
    )

    extract_data >> transform_data >> load_data
