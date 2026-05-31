#!/bin/bash
# arg1: Name of the container
# arg2: CUDA device index to use
# arg3: Docker volume to mount
# Remaining args: Command to run with the container

# Needed to expose Nvidia GPUs to Vulkan renderer
# Locations of Nvidia icd files may vary across hosts and cannot be baked in 
ICD_SEARCH_LOCATIONS=(
    /usr/local/etc/vulkan/icd.d
    /usr/local/share/vulkan/icd.d
    /etc/vulkan/icd.d
    /usr/share/vulkan/icd.d
    /etc/glvnd/egl_vendor.d
    /usr/share/glvnd/egl_vendor.d
)
ICD_MOUNTS=( )
for filename in $(find "${ICD_SEARCH_LOCATIONS[@]}" -name "*nvidia*.json" 2> /dev/null); do
    ICD_MOUNTS+=( --volume "${filename}":"${filename}":ro )
done

# Bind-mount NVIDIA user-space libraries from the host so they match the driver
# (e.g. 595.71). ICD JSON alone is not enough — libGLX_nvidia needs matching
# libnvidia-gpucomp.so.*. Do not mount /usr/lib/wsl/lib (often older builds).
NVIDIA_LIB_DIRS=(
    /lib/x86_64-linux-gnu
    /usr/lib/x86_64-linux-gnu
)
LIB_MOUNTS=( )
for dir in "${NVIDIA_LIB_DIRS[@]}"; do
    [[ -d "$dir" ]] || continue
    for f in "$dir"/libnvidia*.so* "$dir"/libGLX_nvidia*.so*; do
        [[ -e "$f" ]] || continue
        LIB_MOUNTS+=( --volume "${f}":"${f}":ro )
    done
done

# Uncomment to debug only with virtual display
docker run -d --name $1 --gpus "device=$2" \
    --volume $3:/mnt/data_volume \
    ${ICD_MOUNTS[@]} \
    ${LIB_MOUNTS[@]} \
    jpstyle92/semantic-assembler "${@:4}"

# Uncomment to debug with local (linux) machine display
# docker run -d --name $1 --gpus "device=$2" \
#     --volume $3:/mnt/data_volume \
#     --env DISPLAY=$DISPLAY --volume /tmp/.X11-unix:/tmp/.X11-unix \
#     ${ICD_MOUNTS[@]} \
#     jpstyle92/semantic-assembler "${@:4}"
