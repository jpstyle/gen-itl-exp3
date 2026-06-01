#!/bin/bash
# arg1: Name of the container
# arg2: CUDA device index to use
# arg3: Docker volume to mount
# Remaining args: Command to run with the container
#
# GPU_MODE=auto (default): rootless CDI if /etc/cdi/nvidia.yaml exists, else --privileged (sudo).
# One-time: sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml
# Fallback: GPU_MODE=privileged sudo bash $0 ...

set -euo pipefail

CONTAINER_NAME="$1"
GPU_INDEX="$2"
DATA_VOLUME="$3"
shift 3

IMAGE="${DOCKER_IMAGE:-jpstyle92/semantic-assembler}"
RUNTIME="${CONTAINER_RUNTIME:-podman}"
GPU_MODE="${GPU_MODE:-auto}"

_cdi_available() {
    [[ -f /etc/cdi/nvidia.yaml ]] \
        && command -v nvidia-ctk >/dev/null 2>&1 \
        && nvidia-ctk cdi list 2>/dev/null | grep -q 'nvidia.com/gpu='
}

# --- Vulkan ICD + driver libs (privileged path only; CDI injects driver stack) ---
ICD_SEARCH_LOCATIONS=(
    /usr/local/etc/vulkan/icd.d
    /usr/local/share/vulkan/icd.d
    /etc/vulkan/icd.d
    /usr/share/vulkan/icd.d
)
ICD_MOUNTS=( )
NVIDIA_ICDS=( )
LIB_MOUNTS=( )
VK_ICD_ENV=()

_use_cdi=false
case "${GPU_MODE}" in
    cdi) _use_cdi=true ;;
    privileged) _use_cdi=false ;;
    auto)
        if _cdi_available; then
            _use_cdi=true
        elif [[ "${EUID:-$(id -u)}" -eq 0 ]]; then
            _use_cdi=false
        else
            echo "docker_start.sh: rootless needs /etc/cdi/nvidia.yaml or:" >&2
            echo "  GPU_MODE=privileged sudo bash $0 ..." >&2
            exit 1
        fi
        ;;
    *)
        echo "docker_start.sh: unknown GPU_MODE=${GPU_MODE}." >&2
        exit 1
        ;;
esac

if [[ "${_use_cdi}" != true ]]; then
    while IFS= read -r filename; do
        [[ -n "$filename" ]] || continue
        ICD_MOUNTS+=( --volume "${filename}:${filename}:ro" )
        NVIDIA_ICDS+=( "$filename" )
    done < <(find "${ICD_SEARCH_LOCATIONS[@]}" -name '*nvidia*.json' 2>/dev/null | sort -u)
    if ((${#NVIDIA_ICDS[@]} > 0)); then
        VK_ICD_FILENAMES=$(IFS=:; echo "${NVIDIA_ICDS[*]}")
        VK_ICD_ENV=( --env "VK_ICD_FILENAMES=${VK_ICD_FILENAMES}" )
    fi
    for dir in /lib/x86_64-linux-gnu /usr/lib/x86_64-linux-gnu; do
        [[ -d "$dir" ]] || continue
        for f in "$dir"/libnvidia*.so* "$dir"/libGLX_nvidia*.so* "$dir"/libEGL_nvidia*.so*; do
            [[ -e "$f" ]] || continue
            LIB_MOUNTS+=( --volume "${f}:${f}:ro" )
        done
    done
fi

if [[ ! -e "/dev/nvidia${GPU_INDEX}" ]]; then
    echo "docker_start.sh: error: host has no /dev/nvidia${GPU_INDEX}." >&2
    exit 1
fi

PRIVILEGED_FLAG=( )
SECURITY_OPTS=( --security-opt=label=disable )
GPU_DEVICE_ARGS=( )
RUNTIME_ENV=( --env XDG_RUNTIME_DIR=/tmp/runtime-nonroot )

if [[ "${_use_cdi}" == true ]]; then
    GPU_DEVICE_ARGS=( --device nvidia.com/gpu=all )
    echo "docker_start.sh: GPU via CDI (nvidia.com/gpu=all)." >&2
else
    PRIVILEGED_FLAG=( --privileged )
    RUNTIME_ENV+=(
        --env "NVIDIA_VISIBLE_DEVICES=${GPU_INDEX}"
        --env NVIDIA_DRIVER_CAPABILITIES=all
    )
    RUNTIME_ENV+=( "${VK_ICD_ENV[@]}" )
    echo "docker_start.sh: GPU via --privileged (use sudo)." >&2
fi

"${RUNTIME}" rm -f "${CONTAINER_NAME}" 2>/dev/null || true

"${RUNTIME}" run -d --name "${CONTAINER_NAME}" \
    "${PRIVILEGED_FLAG[@]}" \
    "${SECURITY_OPTS[@]}" \
    "${GPU_DEVICE_ARGS[@]}" \
    --volume "${DATA_VOLUME}:/mnt/data_volume" \
    "${ICD_MOUNTS[@]}" \
    "${LIB_MOUNTS[@]}" \
    "${RUNTIME_ENV[@]}" \
    "${IMAGE}" \
    "$@"

if ! "${RUNTIME}" exec "${CONTAINER_NAME}" test -e "/dev/nvidia${GPU_INDEX}"; then
    echo "docker_start.sh: ERROR — /dev/nvidia${GPU_INDEX} missing inside container." >&2
    "${RUNTIME}" rm -f "${CONTAINER_NAME}" 2>/dev/null || true
    exit 1
fi

echo "docker_start.sh: started ${CONTAINER_NAME} (/dev/nvidia${GPU_INDEX} OK)." >&2
