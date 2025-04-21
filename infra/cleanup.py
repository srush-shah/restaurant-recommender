from chi import set_project
from chi.compute import delete_server
from chi.networking import delete_network
from chi.storage import delete_volume
from config import *

set_project(PROJECT_NAME)

delete_server(name=VM_NAME)
delete_volume(name=VOLUME_NAME)
delete_network(name=NETWORK_NAME)

print("🧼 Cleanup complete.")
