from chi import set_project
from chi.networking import create_network, create_subnet
from config import *

set_project(PROJECT_NAME)

net = create_network(name=NETWORK_NAME)
subnet = create_subnet(name=f"{NETWORK_NAME}-subnet", network=net, cidr=SUBNET_CIDR)

print("✅ Network and Subnet created.")
