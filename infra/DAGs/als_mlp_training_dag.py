from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime, timedelta

# Run every 2 weeks
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=10),
}

with DAG(
    dag_id='als_mlp_ray_training',
    default_args=default_args,
    description='Submit ALS and MLP training jobs to Ray every two weeks',
    schedule_interval='@biweekly',  # or '0 0 */14 * *' (every 14 days)
    start_date=datetime(2025, 5, 12),
    catchup=False,
    tags=['ray', 'ml-training'],
) as dag:

    train_als = BashOperator(
        task_id='submit_als_training',
        bash_command=(
            'ray job submit '
            '--runtime-env /home/jovyan/work/scripts/runtime.json '
            '--entrypoint-num-cpus 2 '
            '--verbose '
            '--working-dir /home/jovyan/work/scripts '
            '--name als-train-job '
            '-- python als_train.py'
        )
    )

    train_mlp = BashOperator(
        task_id='submit_mlp_training',
        bash_command=(
            'ray job submit '
            '--runtime-env /home/jovyan/work/scripts/runtime.json '
            '--entrypoint-num-cpus 2 '
            '--verbose '
            '--working-dir /home/jovyan/work/scripts '
            '--name mlp-train-job '
            '-- python mlp_train.py'
        )
    )

    train_als >> train_mlp  # Run ALS first, then MLP
