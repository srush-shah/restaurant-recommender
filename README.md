## Restaurant Recommendation System

<!--
Discuss: Value proposition: Your will propose a machine learning system that can be
used in an existing business or service. (You should not propose a system in which
a new business or service would be developed around the machine learning system.)
Describe the value proposition for the machine learning system. What’s the (non-ML)
status quo used in the business or service? What business metric are you going to be
judged on? (Note that the “service” does not have to be for general users; you can
propose a system for a science problem, for example.)
-->

### Contributors

<!-- Table of contributors and their roles.
First row: define responsibilities that are shared by the team.
Then, each row after that is: name of contributor, their role, and in the third column,
you will link to their contributions. If your project involves multiple repos, you will
link to their contributions in all repos here. -->

| Name             | Responsible for | Link to their commits in this repo |
| ---------------- | --------------- | ---------------------------------- |
| All team members |                 |                                    |
| Maneesh          |                 |                                    |
| Ritesh Ojha      |                 |                                    |
| Russel Sy        |                 |                                    |
| Srushti Shah     |                 |                                    |

### System diagram

<!-- Overall digram of system. Doesn't need polish, does need to show all the pieces.
Must include: all the hardware, all the containers/software platforms, all the models,
all the data. -->

### Summary of outside materials

<!-- In a table, a row for each dataset, foundation model.
Name of data/model, conditions under which it was created (ideally with links/references),
conditions under which it may be used. -->

|              | How it was created | Conditions of use |
| ------------ | ------------------ | ----------------- |
| Data set 1   |                    |                   |
| Data set 2   |                    |                   |
| Base model 1 |                    |                   |
| etc          |                    |                   |

### Summary of infrastructure requirements

<!-- Itemize all your anticipated requirements: What (`m1.medium` VM, `gpu_mi100`),
how much/when, justification. Include compute, floating IPs, persistent storage.
The table below shows an example, it is not a recommendation. -->

| Requirement     | How many/when                                     | Justification |
| --------------- | ------------------------------------------------- | ------------- |
| `m1.medium` VMs | 3 for entire project duration                     | ...           |
| `gpu_mi100`     | 4 hour block twice a week                         |               |
| Floating IPs    | 1 for entire project duration, 1 for sporadic use |               |
| etc             |                                                   |               |

### Detailed design plan

<!-- In each section, you should describe (1) your strategy, (2) the relevant parts of the
diagram, (3) justification for your strategy, (4) relate back to lecture material,
(5) include specific numbers. -->

#### Model training and training platforms

<!-- Make sure to clarify how you will satisfy the Unit 4 and Unit 5 requirements,
and which optional "difficulty" points you are attempting. -->

#### Model serving and monitoring platforms

**Unit 6: Model Serving**

Our serving architecture implements a tiered approach to balance performance and cost. For primary online inference, we deploy quantized models via Ray Serve on GPU nodes, optimized for low-latency predictions. A CPU-based FastAPI fallback handles batch processing using ONNX-optimized models, while edge devices leverage further compressed models for offline scenarios. The system incorporates Redis caching and Nginx load balancing to improve scalability, aligning with Unit 6's focus on model quantization and system-level optimizations. This multi-platform strategy ensures we meet varying latency requirements while maintaining cost efficiency across deployment scenarios.

**Unit 7: Evaluation & Monitoring**

We establish a comprehensive evaluation pipeline beginning with offline tests for accuracy, bias detection, and failure mode analysis. After passing staging load tests, canary deployments use synthetic user profiles to validate real-world performance before full production rollout. Continuous monitoring tracks data drift and model degradation, with explicit/implicit user feedback feeding a retraining loop. This phased approach implements Unit 7's core requirements while addressing difficulty points through automated drift detection and production data recycling for model updates.

<!-- Make sure to clarify how you will satisfy the Unit 6 and Unit 7 requirements,
and which optional "difficulty" points you are attempting. -->

#### Data pipeline

Our data pipeline implements all required components from Unit 8:

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
   - Version control for data using DVC (Data Version Control)
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
     - Structured storage for model training
     - Feature store for online serving

4. **Online Data Pipeline**:

   - Real-time data processing:
     - User interaction streaming pipeline
     - Real-time feature computation
     - Redis caching for fast access
   - Data Simulation:
     - Synthetic user interaction generator
     - Configurable rate and pattern simulation
     - Realistic user behavior patterns including:
       - Restaurant browsing patterns
       - Rating frequencies
       - Review text generation
       - Time-of-day variations

5. **Monitoring and Quality**:
   - Interactive dashboard for data pipeline metrics
   - Automated data quality validation
   - Data drift detection
   - Pipeline performance monitoring

This comprehensive data infrastructure ensures reliable data management, efficient processing, and robust monitoring capabilities, satisfying all Unit 8 requirements while implementing additional features for improved reliability and scalability.

#### Continuous X

Our Continuous X pipeline implements several automated workflows as shown in the system diagram:

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

This satisfies Unit 3 requirements through automated testing, deployment, and monitoring pipelines. We address additional difficulty points through implementation of canary testing and automated model retraining based on drift detection.
