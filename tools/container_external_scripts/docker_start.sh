#!/bin/bash
# arg1: Name of the container
# arg2: CUDA device index to use
# arg3: Docker volume to mount
# Remaining args: Command to run with the container
#
# On hosts where rootless podman does not pass GPUs, run with sudo:
#   sudo bash tools/container_external_scripts/docker_start.sh NAME GPU_IDX DATA_VOL ...

set -euo pipefail

CONTAINER_NAME="$1"
GPU_INDEX="$2"
DATA_VOLUME="$3"
shift 3

IMAGE="${DOCKER_IMAGE:-jpstyle92/semantic-assembler}"
RUNTIME="${CONTAINER_RUNTIME:-podman}"

# --- NVIDIA Vulkan ICD (bind-mount at same absolute path as on host) ---
ICD_SEARCH_LOCATIONS=(
    /usr/local/etc/vulkan/icd.d
    /usr/local/share/vulkan/icd.d
    /etc/vulkan/icd.d
    /usr/share/vulkan/icd.d
)
ICD_MOUNTS=( )
NVIDIA_ICDS=( )
while IFS= read -r filename; do
    [[ -n "$filename" ]] || continue
    ICD_MOUNTS+=( --volume "${filename}:${filename}:ro" )
    NVIDIA_ICDS+=( "$filename" )
done < <(find "${ICD_SEARCH_LOCATIONS[@]}" -name '*nvidia*.json' 2>/dev/null | sort -u)

VK_ICD_ENV=()
if ((${#NVIDIA_ICDS[@]} > 0)); then
    VK_ICD_FILENAMES=$(IFS=:; echo "${NVIDIA_ICDS[*]}")
    VK_ICD_ENV=( --env "VK_ICD_FILENAMES=${VK_ICD_FILENAMES}" )
fi

# --- NVIDIA user-space libraries (must match host driver version) ---
NVIDIA_LIB_DIRS=( /lib/x86_64-linux-gnu /usr/lib/x86_64-linux-gnu )
LIB_MOUNTS=( )
for dir in "${NVIDIA_LIB_DIRS[@]}"; do
    [[ -d "$dir" ]] || continue
    for f in "$dir"/libnvidia*.so* "$dir"/libGLX_nvidia*.so* "$dir"/libEGL_nvidia*.so*; do
        [[ -e "$f" ]] || continue
        LIB_MOUNTS+=( --volume "${f}:${f}:ro" )
    done
done

if [[ ! -e "/dev/nvidia${GPU_INDEX}" ]]; then
    echo "docker_start.sh: error: host has no /dev/nvidia${GPU_INDEX}." >&2
    exit 1
fi

"${RUNTIME}" rm -f "${CONTAINER_NAME}" 2>/dev/null || true

"${RUNTIME}" run -d --name "${CONTAINER_NAME}" \
    --privileged \
    --security-opt=label=disable \
    --volume "${DATA_VOLUME}:/mnt/data_volume" \
    "${ICD_MOUNTS[@]}" \
    "${LIB_MOUNTS[@]}" \
    --env XDG_RUNTIME_DIR=/tmp/runtime-nonroot \
    --env "NVIDIA_VISIBLE_DEVICES=${GPU_INDEX}" \
    --env NVIDIA_DRIVER_CAPABILITIES=all \
    "${VK_ICD_ENV[@]}" \
    "${IMAGE}" \
    "$@"

if ! "${RUNTIME}" exec "${CONTAINER_NAME}" test -e "/dev/nvidia${GPU_INDEX}"; then
    echo "docker_start.sh: ERROR — /dev/nvidia${GPU_INDEX} missing inside container." >&2
    "${RUNTIME}" rm -f "${CONTAINER_NAME}" 2>/dev/null || true
    exit 1
fi

echo "docker_start.sh: started ${CONTAINER_NAME} (/dev/nvidia${GPU_INDEX} OK)." >&2
