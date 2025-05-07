#!/bin/bash

set -e

echo "📦 Checking if /dev/vdb exists..."
lsblk | grep vdb || { echo "❌ /dev/vdb not found. Is the volume attached?"; exit 1; }

echo "🧱 Creating partition table and partition on /dev/vdb..."
sudo parted -s /dev/vdb mklabel gpt
sudo parted -s /dev/vdb mkpart primary ext4 0% 100%

echo "⏳ Waiting for /dev/vdb1 to become available..."
sleep 2
lsblk

echo "🧼 Formatting /dev/vdb1 as ext4..."
sudo mkfs.ext4 /dev/vdb1

echo "📁 Mounting /dev/vdb1 to /mnt/block..."
sudo mkdir -p /mnt/block
sudo mount /dev/vdb1 /mnt/block

echo "🔐 Fixing permissions for user 'cc'..."
sudo chown -R cc /mnt/block
sudo chgrp -R cc /mnt/block

echo "✅ Mounted volume. Current disk usage:"
df -h | grep vdb1