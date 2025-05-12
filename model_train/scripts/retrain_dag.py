from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator 
from airflow.utils.dates import days_ago 
from datetime import datetime, timedelta


def get_new_data(**kwargs):
    ti = kwargs['ti'] # TaskInstance object
    print("Fetching new data for retraining...")
    # Simulate fetching data and returning a path or identifier
    data_path = "/tmp/new_retraining_data.csv"
    
    # Run Maneesh Data fetching code here
    
    with open(data_path, 'w') as f:
        f.write("feature1,feature2,target\n1,2,0\n3,4,1\n") # Dummy data
    print(f"Data fetched and saved to {data_path}")
    ti.xcom_push(key="data_path", value=data_path) # Share data_path with other tasks

def preprocess_data(**kwargs):
    ti = kwargs['ti']
    data_path = ti.xcom_pull(task_ids="get_new_data_task", key="data_path")
    if not data_path:
        raise ValueError("No data path found from previous task.")
    print(f"Preprocessing data from: {data_path}...")
    
    # Run Maneesh preprocessing code here
    
    processed_data_path = "/tmp/processed_retraining_data.csv"
    with open(processed_data_path, 'w') as f:
        f.write("processed_feature1,processed_feature2,target\n0.1,0.2,0\n0.3,0.4,1\n") # Dummy processed data
    print(f"Data preprocessed and saved to {processed_data_path}")
    ti.xcom_push(key="processed_data_path", value=processed_data_path)

def train_model(**kwargs):
    ti = kwargs['ti']
    processed_data_path = ti.xcom_pull(task_ids="preprocess_data_task", key="processed_data_path")
    if not processed_data_path:
        raise ValueError("No processed data path found from previous task.")
    print(f"Training model with data from: {processed_data_path}...")
    
    # Run both of Russel's model training scripts here
    
    model_path = "/tmp/retrained_model.pkl"

    with open(model_path, 'w') as f:
        f.write("dummy model artifact")
    print(f"Model trained and saved to {model_path}")
    ti.xcom_push(key="model_path", value=model_path)

def evaluate_model(**kwargs):
    ti = kwargs['ti']
    model_path = ti.xcom_pull(task_ids="train_model_task", key="model_path")
    if not model_path:
        raise ValueError("No model path found from previous task.")
    print(f"Evaluating model from: {model_path}...")
    
    # Run Russel's model evaluation scripts here
    
    print(f"Model evaluation complete. Model metrics: {metrics}")
    ti.xcom_push(key="metrics", value=metrics)

def deploy_model(**kwargs):
    ti = kwargs['ti']
    model_path = ti.xcom_pull(task_ids="train_model_task", key="model_path")
    is_model_good_enough = ti.xcom_pull(task_ids="evaluate_model_task", key="is_model_good_enough")

    if not model_path or not is_model_good_enough:
        print("Model not deployed (either path missing or evaluation failed).")
        
        # Run Srushti's model deployment and serving scripts here
        
        return

    print(f"Deploying model from: {model_path}...")
    print("Model deployed successfully.")

def notify_success(**kwargs):
    print("Model retraining pipeline completed successfully!")
    # Add notification logic (e.g., send email, Slack message)

def notify_failure(context):
    dag_run = context.get('dag_run')
    task_instance = context.get('task_instance')
    print(f"Model retraining pipeline failed at task: {task_instance.task_id} in DAG run: {dag_run}")
    # Add detailed failure notification logic


default_args = {
    'owner': 'airflow_admin',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1, 
    'retry_delay': timedelta(minutes=5),
    'on_failure_callback': notify_failure, 
}

# --- Define the DAG ---
# The DAG object; this is what Airflow will register and schedule
with DAG(
    dag_id='ml_model_retraining_every_two_weeks',
    default_args=default_args,
    description='A DAG to retrain an ML model every two weeks',
    schedule_interval=timedelta(days=14),
    start_date=days_ago(1),
    catchup=False,
    tags=['ml', 'retraining', 'bi-weekly'],
) as dag:

    # Define Tasks using Operators
    get_new_data_task = PythonOperator(
        task_id='get_new_data_task',
        python_callable=get_new_data,
    )

    preprocess_data_task = PythonOperator(
        task_id='preprocess_data_task',
        python_callable=preprocess_data,
    )

    train_model_task = PythonOperator(
        task_id='train_model_task',
        python_callable=train_model,
    )

    evaluate_model_task = PythonOperator(
        task_id='evaluate_model_task',
        python_callable=evaluate_model,
    )

    deploy_model_task = PythonOperator(
        task_id='deploy_model_task',
        python_callable=deploy_model,
    )

    notify_success_task = PythonOperator(
        task_id='notify_success_task',
        python_callable=notify_success,
        trigger_rule='all_success', # Only run if all upstream tasks succeeded
    )

    # --- Define Task Dependencies (the order of execution) ---
    get_new_data_task >> preprocess_data_task >> train_model_task >> \
    evaluate_model_task >> deploy_model_task >> notify_success_task