from chi import set_project
from chi.storage import create_volume
from config import *

set_project(PROJECT_NAME)

vol = create_volume(name=VOLUME_NAME, size=VOLUME_SIZE)
print("✅ Volume created.")
