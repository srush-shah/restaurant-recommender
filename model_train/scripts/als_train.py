import ray
from ray import train
from ray.train import Trainer
from ray.train.spark import SparkTrainer
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, countDistinct
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator
import sys
import mlflow
import mlflow.spark
import os

# Initialize ray to connect to the existing ray cluster on the node
ray.init(address="auto")

# Set up MLflow tracking URI and experiment name
mlflow.set_tracking_uri("http://129.114.108.166:8000")
experiment_name = "restaurant-rating-als"
mlflow.set_experiment(experiment_name)

def train_als_model(config):
    """Training function to be executed on Ray workers"""
    # Set up Spark session inside the worker
    spark = (
        SparkSession.builder 
        .appName("ALSTrainingOnRay") 
        .config("spark.driver.memory", "4g")
        .config("spark.executor.memory", "4g")
        .config("spark.sql.shuffle.partitions", "200")
        .getOrCreate()
    )
    print("Spark Session created for training")
    
    # Get data paths and configuration from config
    user_id_mapping_path = config["user_id_mapping_path"]
    business_id_mapping_path = config["business_id_mapping_path"]
    file_path = config["file_path"]
    rank = config["rank"]
    maxIter = config["maxIter"]
    regParam = config["regParam"]
    
    # Load the data
    user_id_mapping_loaded = spark.read.parquet(user_id_mapping_path)
    business_id_mapping_loaded = spark.read.parquet(business_id_mapping_path)
    print("ID mappings loaded successfully")
    
    df = spark.read.csv(file_path, header=True, inferSchema=True)
    
    als_input_df_loaded = df.join(user_id_mapping_loaded, "user_id", "inner").drop("user_id")
    als_input_df_loaded = als_input_df_loaded.join(business_id_mapping_loaded, "business_id", "inner").drop("business_id")
    
    als_input_df_loaded = als_input_df_loaded.select(
        col("userCol").cast("int"),
        col("itemCol").cast("int"),
        col("stars").cast("float").alias("rating")
    )

    als_input_df_loaded.printSchema()
    print("ALS training data loaded and prepared")
    
    # Data splitting
    seed = 42 
    (prelim_train_df, prelim_test_df) = als_input_df_loaded.randomSplit([0.8, 0.2], seed=seed)
    
    print(f"Preliminary training data count: {prelim_train_df.count()}")
    print(f"Preliminary test data count: {prelim_test_df.count()}")
    
    # Cache dataframes
    prelim_train_df.cache()
    prelim_test_df.cache()
    
    # Filter users and items
    known_users_in_train = prelim_train_df.select("userCol").distinct()
    known_items_in_train = prelim_train_df.select("itemCol").distinct()
    
    print(f"Distinct users in preliminary training set: {known_users_in_train.count()}")
    print(f"Distinct items in preliminary training set: {known_items_in_train.count()}")
    
    known_users_in_train.cache()
    known_items_in_train.cache()
    
    test_df_known_users = prelim_test_df.join(known_users_in_train, "userCol", "inner")
    final_test_df = test_df_known_users.join(known_items_in_train, "itemCol", "inner")
    
    print(f"Final test data count (after filtering for known users/items): {final_test_df.count()}")
    final_test_df.cache()
    
    final_train_df = prelim_train_df
    print(f"Final training data count: {final_train_df.count()}")
    
    # Start MLflow run
    with mlflow.start_run(run_name="ALS_Ray_Train_Run") as run:
        run_id = run.info.run_id
        print(f"MLflow Run ID: {run_id}")
        
        # Model definition
        als = ALS(
            rank=rank,
            maxIter=maxIter,
            regParam=regParam,
            userCol="userCol",
            itemCol="itemCol",
            ratingCol="rating",
            coldStartStrategy="drop",
            nonnegative=True,
            implicitPrefs=False,
            seed=42
        )
        
        # Log parameters
        print("Logging parameters to MLflow...")
        mlflow.log_param("rank", rank)
        mlflow.log_param("maxIter", maxIter)
        mlflow.log_param("regParam", regParam)
        mlflow.log_param("training_data_count", final_train_df.count())
        mlflow.log_param("test_data_count", final_test_df.count())
        
        # Train model
        print("Training the ALS model...")
        model = als.fit(final_train_df)
        print("Model training complete.")
        
        # Make predictions
        print("Making predictions on the test set...")
        predictions = model.transform(final_test_df)
        predictions_cleaned = predictions.na.drop(subset=[als.getPredictionCol()])
        
        if predictions.count() != predictions_cleaned.count():
            print(f"Dropped {predictions.count() - predictions_cleaned.count()} rows with NaN predictions.")
        
        # Evaluate model
        print("Evaluating the model...")
        evaluator_rmse = RegressionEvaluator(
            metricName="rmse", labelCol="rating", predictionCol=als.getPredictionCol()
        )
        rmse = evaluator_rmse.evaluate(predictions_cleaned)
        print(f"Root Mean Squared Error (RMSE) on test data = {rmse}")
        
        evaluator_mae = RegressionEvaluator(
            metricName="mae", labelCol="rating", predictionCol=als.getPredictionCol()
        )
        mae = evaluator_mae.evaluate(predictions_cleaned)
        print(f"Mean Absolute Error (MAE) on test data = {mae}")
        
        evaluator_r2 = RegressionEvaluator(
            metricName="r2", labelCol="rating", predictionCol=als.getPredictionCol()
        )
        r2 = evaluator_r2.evaluate(predictions_cleaned)
        print(f"R-squared (R2) on test data = {r2}")
        
        # Log metrics
        print("Logging metrics to MLflow...")
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2", r2)
        
        # Log model
        print("Logging model to MLflow...")
        mlflow.spark.log_model(
            spark_model=model,
            artifact_path="als-model",
        )
        print("Model logged.")
        
        print(f"\nMLflow Run completed. Check the MLflow UI for run ID: {run_id}")
        print(f"Experiment ID: {run.info.experiment_id}")
        
        # Report results to Ray Train
        train.report({
            "rmse": rmse,
            "mae": mae,
            "r2": r2
        })
        
        return {
            "rmse": rmse,
            "mae": mae,
            "r2": r2,
            "run_id": run_id
        }

# Define the configuration for our training run
config = {
    "user_id_mapping_path": "/home/jovyan/work/id_mappings/user_id_mapping",
    "business_id_mapping_path": "/home/jovyan/work/id_mappings/business_id_mapping",
    "file_path": "data/als/training_data.csv",
    "rank": 10,
    "maxIter": 15,
    "regParam": 0.1
}

# Initialize the Ray Trainer
trainer = Trainer(
    backend="spark", 
    num_workers=2,  # Adjust based on your Ray cluster size
    use_gpu=False,  # Change if you have GPUs
    resources_per_worker={"CPU": 32}  # Adjust based on resources per worker
)

# Start the trainer and run the training function
print("Starting Ray Train job...")
trainer.start()
results = trainer.run(train_als_model, config=config)
trainer.shutdown()

print("Training completed!")
print("Results:", results)