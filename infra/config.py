# General infra settings
PROJECT_NAME = "restaurant-recommender-project"

# VM Setup (for regular service)
VM_NAME = "vm-recommender"
VM_FLAVOR = "m1.medium"
IMAGE_NAME = "CC-Ubuntu22.04"
KEYPAIR = "your-keypair-name"
FLOATING_IP_NAME = "fip-recommender"

# Volume + Network
VOLUME_NAME = "volume-recommender"
VOLUME_SIZE = 50  # in GB
NETWORK_NAME = "net-recommender"
SUBNET_CIDR = "192.168.0.0/24"

# 🧠 GPU Node Setup (for model training / deep learning)
GPU_SERVER_NAME = "node-llm"
GPU_IMAGE = "CC-Ubuntu24.04-CUDA"
LEASE_NAME = "llm_ro2283"  # Replace with your lease name

# Container Info
CONTAINER_IMAGE = "quay.io/jupyter/pytorch-notebook:cuda12-pytorch-2.5.1"
CONTAINER_NAME = "torchnb"
JUPYTER_PORT = 8888
