# bringup_gpu.py

from chi import context, lease, server
from config import *
import os
import time

# -----------------------------
# 1. Initialize Chameleon Context
# -----------------------------
print("🌐 Setting up Chameleon context...")
context.version = "1.0"
context.choose_project()
context.choose_site(default="CHI@UC")

# -----------------------------
# 2. Fetch Lease
# -----------------------------
print(f"📦 Fetching lease: {LEASE_NAME}")
l = lease.get_lease(LEASE_NAME)
l.show()

# -----------------------------
# 3. Launch GPU Server
# -----------------------------
username = os.getenv('USER', 'cc')  # fallback to 'cc' if undefined
gpu_server_name = f"{GPU_SERVER_NAME}-{username}"

print(f"🚀 Launching GPU server: {gpu_server_name}")
s = server.Server(
    name=gpu_server_name,
    reservation_id=l.node_reservations[0]["id"],
    image_name=GPU_IMAGE
)
s.submit(idempotent=True)

# Wait for boot (alternatively use s.wait_for_active())
print("⏳ Waiting 60 seconds for server to boot...")
time.sleep(60)
s.refresh()

# -----------------------------
# 4. Assign Floating IP
# -----------------------------
print("🌍 Associating floating IP...")
s.associate_floating_ip()
s.refresh()
s.check_connectivity()
s.show()

# -----------------------------
# 5. Install Docker
# -----------------------------
print("🐳 Installing Docker...")
s.execute("curl -sSL https://get.docker.com/ | sudo sh")
s.execute("sudo groupadd -f docker; sudo usermod -aG docker $USER")
s.execute("newgrp docker")  # optional, makes docker immediately usable
s.execute("docker run hello-world")

# -----------------------------
# 6. Install NVIDIA Container Toolkit
# -----------------------------
print("🔧 Installing NVIDIA container toolkit...")
s.execute("""
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg && \
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
""")
s.execute("sudo apt update")
s.execute("sudo apt-get install -y nvidia-container-toolkit")
s.execute("sudo nvidia-ctk runtime configure --runtime=docker")
s.execute("sudo systemctl restart docker")

# -----------------------------
# 7. Test GPU Access
# -----------------------------
print("🧪 Testing GPU access inside container...")
s.execute("docker run --rm --gpus all ubuntu nvidia-smi")

# -----------------------------
# 8. Pull and Run Jupyter Notebook
# -----------------------------
print(f"📥 Pulling container image: {CONTAINER_IMAGE}")
s.execute(f"docker pull {CONTAINER_IMAGE}")

print("🚀 Starting Jupyter notebook container...")
s.execute(
    f"docker run -d -p {JUPYTER_PORT}:{JUPYTER_PORT} --gpus all "
    f"--name {CONTAINER_NAME} {CONTAINER_IMAGE}"
)

# -----------------------------
# 9. Print Tunnel Instructions
# -----------------------------
print("\n✅ GPU Server is up and running!")
print("🔗 You can now create an SSH tunnel to access the Jupyter Notebook:")
print("------------------------------------------------------------")
print(f"ssh -L 8888:127.0.0.1:8888 -i ~/.ssh/id_rsa_chameleon cc@{s.floating_ip}")
print("------------------------------------------------------------")
print(f"Once connected, run the following to get the Jupyter token:\n")
print(f"s.execute('docker logs {CONTAINER_NAME}')")
print("------------------------------------------------------------")
print("Then go to http://localhost:8888 in your browser.")
