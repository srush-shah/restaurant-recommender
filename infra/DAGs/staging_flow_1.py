from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import os
import requests
import subprocess

default_args = {
    'owner': 'airflow',
    'retry_delay': timedelta(minutes=5),
    'retries': 2,
}

def trigger_training_job():
    ip = os.environ['TRAINING_NODE_IP']
    resp = requests.post(f"http://{ip}:9090/trigger-training")
    resp.raise_for_status()
    return resp.json().get('new_model_version')

def trigger_argo_workflow(template, parameters):
    base_cmd = ["argo", "submit", "--from", f"workflowtemplate/{template}"]
    for key, val in parameters.items():
        base_cmd.extend(["-p", f"{key}={val}"])
    subprocess.run(base_cmd, check=True)

def build_container(model_version: str):
    trigger_argo_workflow("build-container-image", {"model-version": model_version})

def deploy_to_staging(model_version: str):
    trigger_argo_workflow("deploy-container-image", {"env": "staging", "model-version": model_version})

def promote_model(src_env, tgt_env, model_version):
    trigger_argo_workflow("promote-model", {
        "source-env": src_env,
        "target-env": tgt_env,
        "model-version": model_version
    })

with DAG(
    dag_id="ml_lifecycle_pipeline",
    default_args=default_args,
    start_date=datetime(2024, 1, 1),
    schedule_interval=None,
    catchup=False,
    description="Full ML model lifecycle from training to production",
) as dag:

    trigger_train = PythonOperator(
        task_id="trigger_training_job",
        python_callable=trigger_training_job
    )

    build_container_task = PythonOperator(
        task_id="build_container",
        python_callable=build_container,
        op_args=["{{ ti.xcom_pull(task_ids='trigger_training_job') }}"]
    )

    deploy_staging_task = PythonOperator(
        task_id="deploy_to_staging",
        python_callable=deploy_to_staging,
        op_args=["{{ ti.xcom_pull(task_ids='trigger_training_job') }}"]
    )

    promote_to_canary_task = PythonOperator(
        task_id="promote_to_canary",
        python_callable=promote_model,
        op_args=["staging", "canary", "{{ ti.xcom_pull(task_ids='trigger_training_job') }}"]
    )

    promote_to_production_task = PythonOperator(
        task_id="promote_to_production",
        python_callable=promote_model,
        op_args=["canary", "production", "{{ ti.xcom_pull(task_ids='trigger_training_job') }}"]
    )

    trigger_train >> build_container_task >> deploy_staging_task >> promote_to_canary_task >> promote_to_production_task
