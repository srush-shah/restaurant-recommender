## Restaurant Recommendation System

<!--
Discuss: Value proposition: Your will propose a machine learning system that can be
used in an existing business or service. (You should not propose a system in which
a new business or service would be developed around the machine learning system.)
Describe the value proposition for the machine learning system. What's the (non-ML)
status quo used in the business or service? What business metric are you going to be
judged on? (Note that the "service" does not have to be for general users; you can
propose a system for a science problem, for example.)
-->

### Contributors

<!-- Table of contributors and their roles.
First row: define responsibilities that are shared by the team.
Then, each row after that is: name of contributor, their role, and in the third column,
you will link to their contributions. If your project involves multiple repos, you will
link to their contributions in all repos here. -->

## Team Contributions

| Name         | Responsible for | Link to their commits in this repo                                                                 |
| ------------ | --------------- | -------------------------------------------------------------------------------------------------- |
| Maneesh      | Units - 8       | [Commits](https://github.com/srush-shah/restaurant-recommender/commits/main/?author=Maneeshk11)    |
| Ritesh Ojha  | Units - 3       | [Commits](https://github.com/srush-shah/restaurant-recommender/commits/main/?author=ritzzi23)      |
| Russel Sy    | Units - 4,5     | [Commits](https://github.com/srush-shah/restaurant-recommender/commits/main/?author=russelgabriel) |
| Srushti Shah | Units - 6,7     | [Commits](https://github.com/srush-shah/restaurant-recommender/commits/main/?author=srush-shah)    |

### System diagram

<img src="./assets/architecture_v1.png"/>

<!-- Overall digram of system. Doesn't need polish, does need to show all the pieces.
Must include: all the hardware, all the containers/software platforms, all the models,
all the data. -->

### Summary of outside materials

<!-- In a table, a row for each dataset, foundation model.
Name of data/model, conditions under which it was created (ideally with links/references),
conditions under which it may be used. -->

|                   | How it was created                                                                                                                                                                                                                                                                                                       | Conditions of use                                                                                           |
| ----------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------- |
| Yelp Open Dataset | The Yelp Open Dataset is a subset of Yelp data that is intended for educational use. It provides real-world data related to businesses including reviews, photos, check-ins, and attributes like hours, parking availability, and ambience.                                                                              | [See detailed ToS here](https://github.com/srush-shah/restaurant-recommender/tree/main/assets/yelp_tos.pdf) |
| SBERT Transformer | Hugging Face used the pretrained microsoft/mpnet-base model and fine-tuned in on a 1B sentence pairs dataset. They used a contrastive learning objective: given a sentence from the pair, the model should predict which out of a set of randomly sampled other sentences, was actually paired with it in their dataset. | [Hugging Face ToS](https://huggingface.co/terms-of-service)                                                 |

### Summary of infrastructure requirements

## Summary of Infrastructure Requirements (Chameleon)

| Requirement                         | How many / When                                      | Justification                                                                                        |
| ----------------------------------- | ---------------------------------------------------- | ---------------------------------------------------------------------------------------------------- |
| **m1.medium VMs**                   | 3 for entire project duration                        | Used for ETL jobs, FastAPI server, Redis, and Ray head node                                          |
| **gpu_a100 or gpu_mi100**           | 4-hour block twice a week                            | Heavy model training (DCN, ALS, SBERT embedding) on large Yelp data                                  |
| **Floating IPs**                    | 1 static IP for entire project duration, 1 on-demand | One for public-facing FastAPI; additional one for staging/monitoring access during canary tests      |
| **Block Storage (100GB)**           | Persistent volume throughout project                 | Store processed Yelp data, user/restaurant embeddings, cached features                               |
| **Object Storage (S3-like)**        | Persistent throughout project                        | Store MLflow artifacts, model checkpoints, and logs                                                  |
| **Docker Registry Access**          | Continuous                                           | For storing/retrieving containerized services (ETL, training, serving)                               |
| **gpu_small VMs (optional)**        | 2 hours weekly (as-needed backup to big GPU)         | Light GPU experimentation or embedding refreshes if gpu_mi100 unavailable                            |
| **Kubernetes Cluster (bare-metal)** | 1 cluster with 3 nodes (2 CPU + 1 GPU)               | To deploy microservices (ETL API, model training jobs, model serving) and support canary deployments |
| **Internal Network**                | Throughout project                                   | For communication between Redis, model server, dashboard, MLflow tracker, etc.                       |

### Rough break down

## Component-to-Node Mapping

| Component                              | Needs?             | Recommended Node Type                                |
| -------------------------------------- | ------------------ | ---------------------------------------------------- |
| **ETL (SBERT embeddings)**             | CPU (GPU optional) | m1.medium (or gpu_small if SBERT is GPU-accelerated) |
| **Model Training (ALS/DCN)**           | GPU-intensive      | gpu_mi100 or gpu_a100                                |
| **Model Serving (FastAPI + RayServe)** | CPU                | m1.medium                                            |
| **Redis (caching)**                    | CPU                | m1.small or m1.medium                                |
| **MLflow (tracking + registry)**       | CPU                | m1.small                                             |
| **Dashboard (Grafana/Prometheus)**     | CPU                | m1.small                                             |
| **Canary / Staging Env**               | CPU                | m1.medium (on-demand/scheduled)                      |
| **Load Testing (Optional)**            | CPU                | Ephemeral VM (as-needed only)                        |

## VM Breakdown

| VM Purpose                    | VM Type   | Count               | Notes                                           |
| ----------------------------- | --------- | ------------------- | ----------------------------------------------- |
| **Ray Cluster Head Node**     | m1.medium | 1                   | Controls Ray tasks, does some orchestration     |
| **Ray Worker Node (CPU)**     | m1.medium | 1–2                 | For ETL, inference, lightweight model serving   |
| **GPU Training Node**         | gpu_mi100 | On demand (2x/week) | For SBERT/DCN/ALS training (can be preemptible) |
| **Redis & MLflow**            | m1.small  | 1                   | Can be co-hosted if needed                      |
| **Canary/Testing Node**       | m1.medium | 1 (as needed)       | Used only during staged testing                 |
| **Dashboard Node (optional)** | m1.small  | 1                   | Optional unless you're monitoring live stats    |

Note: It is subject to change as we implement.

### Detailed design plan

<!-- In each section, you should describe (1) your strategy, (2) the relevant parts of the
diagram, (3) justification for your strategy, (4) relate back to lecture material,
(5) include specific numbers. -->

#### Model training and training platforms

**Unit 4 Requirements:**

1. **Train and Re-train**:

   - Primary training: Alternating Least Squares (ALS) for collaborative filtering
   - Deep Neural Network (DCN) for feature-based recommendations
   - Continuous retraining pipeline using production feedback data
   - Model artifacts stored in versioned storage for reproducibility

2. **Modeling Choices**:
   - ALS for sparse user-restaurant interaction matrix
   - DCN for handling complex feature interactions between user and restaurant profiles
   - Multi-GPU training support for both models using Ray
   - Hybrid recommendation approach combining both models' predictions

**Unit 5 Requirements:**

1. **Experiment Tracking**:

   - MLflow deployment on Chameleon for experiment management
   - Tracking of model metrics, hyperparameters, and artifacts
   - Automated logging of training metrics and model performance
   - Version control of model artifacts and configurations

2. **Training Job Scheduling**:
   - Ray cluster deployment for distributed training
   - GPU resource management (NVIDIA and AMD)
   - Automated job scheduling and resource allocation
   - Integration with continuous training pipeline

**Unit 5 Difficulty Points:**

1. **Ray Train Implementation**:

   - Fault-tolerant training with automatic checkpointing
   - Remote artifact storage integration
   - Distributed training across GPU nodes
   - Automatic failover and recovery

2. **Hyperparameter Tuning**:
   - Ray Tune integration for automated optimization
   - Population-based training for efficient search
   - Multi-objective optimization for latency-accuracy trade-offs
   - Parallel trial execution across available GPUs

#### Model serving and monitoring platforms

##### Model Serving

Our model serving pipeline implements all required components from Unit 6:

1. **Serving from an API Endpoint**:

   - Model is deployed using **FastAPI** and **Ray Serve** for efficient, scalable inference.
   - Optimized API endpoints for real-time and batch recommendations.

2. **Identifying Deployment Requirements**:

   - **Model Size Considerations**: Ensure model fits within serving infrastructure constraints.
   - **Throughput Optimization**: Designed for high-volume batch inference.
   - **Latency Constraints**: Ensuring minimal response time for real-time recommendations.
   - **Concurrency Management**: Handling multiple requests efficiently in a cloud environment.

3. **Model Optimizations**:

   - Graph optimizations for execution efficiency.
   - Quantization and reduced precision for performance improvements.
   - Hardware-optimized operators for both **CPU** and **GPU** deployments.

4. **System Optimizations**:

   - Load balancing for efficient request distribution.
   - Optimized resource allocation to meet scaling demands.

5. **Multiple Serving Options**:
   - Deployment comparisons across:
     - **Server-grade GPU** for high-performance inference.
     - **Server-grade CPU** for cost-efficient serving.
     - **On-device inference** for edge applications.
   - Performance and cost trade-off analysis for each approach.

This serving strategy ensures fast, reliable, and scalable model deployment for restaurant recommendations.

---

##### Evaluation & Monitoring

Our evaluation and monitoring pipeline implements all required components from Unit 7:

1. **Offline Evaluation**:

   - **Automated Model Evaluation** post-training, with results logged in **MLflow**.
   - **Comprehensive Testing** covering:
     - Standard and domain-specific test cases.
     - Fairness and bias evaluations across user populations.
     - Failure mode testing.
     - Unit tests with predefined templates.
   - **Model Registry Automation** to track performance over iterations.

2. **Load Testing in Staging**:

   - Performance benchmarking before full deployment.
   - Stress testing model under varying loads.

3. **Online Evaluation in Canary Environment**:

   - Deploying in a controlled test environment.
   - Simulated user interactions for real-world performance validation.
   - Behavioral analysis to refine recommendation strategies.

4. **Closing the Feedback Loop**:

   - **User Feedback Collection** through interactions, ratings, and labeled annotations.
   - **Production Data Storage** for continuous model improvement and retraining.

5. **Business-Specific Evaluation**:

   - Defining success metrics aligned with business goals.
   - Tracking real-world impact of recommendations.

6. **Advanced Monitoring Features**:
   - **Data Drift Detection**:
     - Identifying shifts in user behavior and recommendation relevance.
   - **Model Degradation Monitoring**:
     - Performance tracking and alerts for degradation.
     - Automated retraining with fresh production data.

This robust evaluation pipeline ensures continuous monitoring, improvement, and adaptation of our recommendation model.

<!-- Make sure to clarify how you will satisfy the Unit 6 and Unit 7 requirements,
and which optional "difficulty" points you are attempting. -->

#### Data pipeline

**Unit 8:**

1. **Persistent Storage**:

   - Dedicated Chameleon persistent storage volumes for:
     - Raw Yelp dataset and processed data
     - Model training artifacts and checkpoints
     - SBERT embeddings and feature transformations
     - Container images for deployment
   - Managed through infrastructure-as-code for automated provisioning and attachment

2. **Offline Data Management**:

   - Primary data source: Yelp Dataset (structured JSON)
     - Restaurant business data: attributes, categories, locations
     - User data: ratings, reviews, user metadata
   - Data repository: Distributed storage system for efficient access
   - Regular snapshots for reproducibility

3. **ETL Pipeline**:

   - Data Ingestion:
     - Yelp Dataset import pipeline
     - SBERT Transformer for text processing using Ray
     - Distributed processing for scalability
   - Transformation:
     - User profile generation (rating vectors, embeddings)
     - Restaurant profile creation (business attributes, embeddings)
     - Data validation and quality checks
   - Loading:
     - Structured storage (on Chameleon) for model training
     - Feature store (on Chameleon) for online serving

4. **Online Data Pipeline**:

   - Real-time data processing:
     - User interaction streaming pipeline
     - Real-time feature computation after storing data on redis(or an alternative for fast access)
   - Data Simulation:
     - Synthetic user interaction generator (script) based on user patterns

5. **Monitoring and Quality**:
   - Interactive dashboard for data pipeline metrics
   - Automated data quality validation

#### Continuous X

Implementations of Continuous X automated workflows for our use-case:

1. **Continuous Integration/Deployment**:

   - Automated model training pipeline using MLflow and Ray
   - Containerized deployment using FastAPI and RayServe
   - Canary testing for new model versions before production deployment

2. **Continuous Monitoring**:

   - Real-time tracking of operational metrics (batch throughput, inference latency)
   - Model performance metrics (Precision@K, Recall@K, Mean Average Precision, RMSE)
   - User engagement and retention metrics
   - Automated model drift detection

3. **Continuous Learning**:
   - Integration of production feedback into training pipeline
   - Automated model retraining based on performance thresholds
   - A/B testing between new and old system versions

This satisfies Unit 3 requirements through automated testing, deployment, and monitoring pipelines.
