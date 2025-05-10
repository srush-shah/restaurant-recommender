#!/usr/bin/env bash
set -euo pipefail

source "$HOME/openrc"

# 3) Lookup IDs
SERVER_ID=$(
  curl -s http://169.254.169.254/openstack/latest/meta_data.json \
    | python3 -c "import sys, json; print(json.load(sys.stdin)['uuid'])"
)

VOLUME_NAME="block-persist-project23_2"

VOLUME_ID=$(openstack volume list \
  --format value -c ID -c Name \
  | awk "\$2==\"$VOLUME_NAME\" {print \$1}")


echo "Attaching volume $VOLUME_NAME ($VOLUME_ID) → server $SERVER_ID"
openstack server add volume "$SERVER_ID" "$VOLUME_ID"


sudo mkdir -p /mnt/block
sudo mount /dev/vdb1 /mnt/block

sudo chown -R cc /mnt/block
sudo chgrp -R cc /mnt/block

df -h
