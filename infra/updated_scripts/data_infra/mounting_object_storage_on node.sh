curl https://rclone.org/install.sh | sudo bash

sudo sed -i '/^#user_allow_other/s/^#//' /etc/fuse.conf

mkdir -p ~/.config/rclone

# Write your rclone.conf
cat > ~/.config/rclone/rclone.conf <<'EOF'
[chi_tacc]
type = swift
user_id = 59b49300deb6f11832f7764d9f2c3451ba51e578bb556d58e02bf4c64bd89a31
application_credential_id = aba2c756cb1649b6bd47c572f67a9528
application_credential_secret = HnUZC1Dvhnw1A5-19LiDQORiWURpL1IflqK3tk506NXJp8Tb0HSW4m0V9Ch4zEhwV9WR7tTxiAPTcTGHMi2SKw
auth = https://chi.tacc.chameleoncloud.org:5000/v3
region = CHI@TACC
EOF

rclone lsd chi_tacc:

export RCLONE_CONTAINER=object-persist-project23_2

sudo mkdir -p /mnt/object
sudo chown -R cc /mnt/object
sudo chgrp -R cc /mnt/object


rclone mount chi_tacc:object-persist-project23_2 /mnt/object --read-only --allow-other --daemon

ls /mnt/object
