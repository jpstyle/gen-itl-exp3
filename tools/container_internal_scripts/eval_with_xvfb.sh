#!/bin/bash
source tools/container_internal_scripts/setup_gpu_vulkan_env.sh
bash tools/container_internal_scripts/start_xvfb.sh
eval $*
rsync --archive --recursive --update --compress --info=progress2 outputs/ /mnt/data_volume/outputs/