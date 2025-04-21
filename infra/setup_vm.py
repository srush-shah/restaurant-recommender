from chi import set_project
from chi.compute import create_server
from chi.networking import get_network
from config import *

set_project(PROJECT_NAME)

net = get_network(NETWORK_NAME)
server = create_server(
    name=VM_NAME,
    image_name=IMAGE_NAME,
    flavor_name=VM_FLAVOR,
    networks=[net],
    key_name=KEYPAIR,
    floating_ip_name=FLOATING_IP_NAME
)

print(f"✅ VM '{VM_NAME}' created with floating IP assigned.")
