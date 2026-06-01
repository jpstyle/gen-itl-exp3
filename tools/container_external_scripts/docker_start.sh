#!/bin/bash
# Start semantic-assembler with GPU + Vulkan support.
#
# Usage: docker_start.sh <container_name> <gpu_index> <host_mount> [command...]
#
# Environment:
#   CONTAINER_RUNTIME  docker | podman (default: podman if installed, else docker)
#   DOCKER_IMAGE       image ref (default: jpstyle92/semantic-assembler)
#   GPU_MODE           auto | cdi | gpus | privileged
#                      auto: CDI probe, then docker --gpus, then privileged (root only)
#
# Podman < 5.1 + nvidia-ctk >= 1.18: regenerate CDI without additionalGids once:
#   sudo nvidia-ctk cdi generate \
#     --feature-flag=no-additional-gids-for-device-nodes \
#     --output=/var/run/cdi/nvidia.yaml

set -euo pipefail

readonly SCRIPT_NAME="${0##*/}"
readonly CDI_DEVICE='nvidia.com/gpu=all'
readonly CDI_PROBE_IMAGE="${CDI_PROBE_IMAGE:-docker.io/nvidia/cuda:12.0.0-base-ubuntu22.04}"
readonly CDI_SPEC_PATHS=( /var/run/cdi/nvidia.yaml /etc/cdi/nvidia.yaml )
readonly CDI_COMPAT_FLAG=no-additional-gids-for-device-nodes
readonly ICD_SEARCH_DIRS=(
    /usr/local/etc/vulkan/icd.d
    /usr/local/share/vulkan/icd.d
    /etc/vulkan/icd.d
    /usr/share/vulkan/icd.d
)
readonly LIB_SEARCH_DIRS=( /lib/x86_64-linux-gnu /usr/lib/x86_64-linux-gnu )

CONTAINER_NAME="${1:?container name}"
GPU_INDEX="${2:?gpu index}"
DATA_VOLUME="${3:?host directory to mount at /mnt/data_volume}"
shift 3

IMAGE="${DOCKER_IMAGE:-jpstyle92/semantic-assembler}"
GPU_MODE="${GPU_MODE:-auto}"
EUID_VAL="${EUID:-$(id -u)}"

log() { echo "${SCRIPT_NAME}: $*" >&2; }
die() { log "error: $*"; exit 1; }

resolve_runtime() {
    if [[ -n "${CONTAINER_RUNTIME:-}" ]]; then
        command -v "${CONTAINER_RUNTIME}" >/dev/null 2>&1 \
            || die "CONTAINER_RUNTIME=${CONTAINER_RUNTIME} not found in PATH"
        echo "${CONTAINER_RUNTIME}"
        return
    fi
    if command -v podman >/dev/null 2>&1; then
        echo podman
    elif command -v docker >/dev/null 2>&1; then
        echo docker
    else
        die 'install docker or podman, or set CONTAINER_RUNTIME'
    fi
}

runtime_is() {
    [[ "$(basename "$(command -v "${RUNTIME}")")" == "$1" ]] \
        || [[ "${RUNTIME}" == "$1" ]]
}

cdi_spec_has_additional_gids() {
    local f
    for f in "${CDI_SPEC_PATHS[@]}"; do
        [[ -r "$f" ]] || continue
        grep -q 'additionalGids' "$f" && return 0
    done
    return 1
}

regenerate_cdi_spec() {
    local out=/var/run/cdi/nvidia.yaml
    command -v nvidia-ctk >/dev/null 2>&1 || die 'nvidia-ctk not found'
    mkdir -p "$(dirname "${out}")"
    nvidia-ctk cdi generate --feature-flag="${CDI_COMPAT_FLAG}" --output="${out}"
}

cdi_incompatible_hint() {
    log "${RUNTIME} cannot parse NVIDIA CDI spec (additionalGids field)."
    log "One-time host fix:"
    log "  sudo nvidia-ctk cdi generate --feature-flag=${CDI_COMPAT_FLAG} --output=/var/run/cdi/nvidia.yaml"
}

ensure_cdi_spec_compatible() {
    cdi_spec_has_additional_gids || return 0
    if [[ "${EUID_VAL}" -eq 0 ]]; then
        log "regenerating Podman-compatible CDI spec at /var/run/cdi/nvidia.yaml"
        regenerate_cdi_spec
        return 0
    fi
    cdi_incompatible_hint
    exit 1
}

cdi_probe() {
  "${RUNTIME}" run --rm \
    --device "${CDI_DEVICE}" \
    --security-opt=label=disable \
    "${CDI_PROBE_IMAGE}" \
    true &>/dev/null
}

collect_driver_mounts() {
    ICD_MOUNTS=()
    LIB_MOUNTS=()
    VK_ICD_ENV=()
    local icds=() filename f dir

    while IFS= read -r filename; do
        [[ -n "$filename" ]] || continue
        ICD_MOUNTS+=( --volume "${filename}:${filename}:ro" )
        icds+=( "$filename" )
    done < <(find "${ICD_SEARCH_DIRS[@]}" -name '*nvidia*.json' 2>/dev/null | sort -u)

    if ((${#icds[@]} > 0)); then
        VK_ICD_ENV=( --env "VK_ICD_FILENAMES=$(IFS=:; echo "${icds[*]}")" )
    fi

    for dir in "${LIB_SEARCH_DIRS[@]}"; do
        [[ -d "$dir" ]] || continue
        for f in "$dir"/libnvidia*.so* "$dir"/libGLX_nvidia*.so* "$dir"/libEGL_nvidia*.so*; do
            [[ -e "$f" ]] || continue
            LIB_MOUNTS+=( --volume "${f}:${f}:ro" )
        done
    done
}

# Sets: GPU_BACKEND (cdi|gpus|privileged), PRIVILEGED_FLAG, SECURITY_OPTS,
#       GPU_DEVICE_ARGS, RUNTIME_ENV, ICD_MOUNTS, LIB_MOUNTS
select_gpu_backend() {
    GPU_BACKEND=
    PRIVILEGED_FLAG=()
    SECURITY_OPTS=()
    GPU_DEVICE_ARGS=()
    RUNTIME_ENV=( --env XDG_RUNTIME_DIR=/tmp/runtime-nonroot )
    ICD_MOUNTS=()
    LIB_MOUNTS=()

    case "${GPU_MODE}" in
        cdi|auto)
            ensure_cdi_spec_compatible
            if cdi_probe; then
                GPU_BACKEND=cdi
                return
            fi
            [[ "${GPU_MODE}" == cdi ]] && die "CDI device ${CDI_DEVICE} not available"
            ;;
        privileged)
            collect_driver_mounts
            GPU_BACKEND=privileged
            return
            ;;
        gpus)
            if runtime_is docker; then
                GPU_BACKEND=gpus
                return
            fi
            die 'GPU_MODE=gpus is for docker only; use auto, cdi, or privileged with podman'
            ;;
        *)
            die "unknown GPU_MODE=${GPU_MODE} (use auto, cdi, gpus, privileged)"
            ;;
    esac

    # auto: fallbacks after CDI
    if runtime_is docker; then
        GPU_BACKEND=gpus
        return
    fi
    if [[ "${EUID_VAL}" -eq 0 ]]; then
        log 'CDI probe failed; falling back to privileged + driver bind-mounts'
        collect_driver_mounts
        GPU_BACKEND=privileged
        return
    fi

    log 'CDI probe failed.'
    cdi_incompatible_hint
    log "Or: GPU_MODE=privileged sudo bash ${SCRIPT_NAME} ${CONTAINER_NAME} ${GPU_INDEX} ${DATA_VOLUME} ..."
    exit 1
}

apply_gpu_backend() {
    case "${GPU_BACKEND}" in
        cdi)
            SECURITY_OPTS=( --security-opt=label=disable )
            GPU_DEVICE_ARGS=( --device "${CDI_DEVICE}" )
            log "GPU via CDI (${CDI_DEVICE})."
            ;;
        gpus)
            GPU_DEVICE_ARGS=( --gpus "device=${GPU_INDEX}" )
            log "GPU via docker --gpus device=${GPU_INDEX}."
            ;;
        privileged)
            PRIVILEGED_FLAG=( --privileged )
            SECURITY_OPTS=( --security-opt=label=disable )
            RUNTIME_ENV+=(
                --env "NVIDIA_VISIBLE_DEVICES=${GPU_INDEX}"
                --env NVIDIA_DRIVER_CAPABILITIES=all
                "${VK_ICD_ENV[@]}"
            )
            log 'GPU via --privileged + driver bind-mounts (use sudo for podman rootless).'
            ;;
        *) die "internal error: unknown GPU_BACKEND=${GPU_BACKEND}" ;;
    esac
}

[[ -e "/dev/nvidia${GPU_INDEX}" ]] || die "host has no /dev/nvidia${GPU_INDEX}"

RUNTIME="$(resolve_runtime)"
select_gpu_backend
apply_gpu_backend

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
    log "ERROR — /dev/nvidia${GPU_INDEX} missing inside container."
    "${RUNTIME}" rm -f "${CONTAINER_NAME}" 2>/dev/null || true
    exit 1
fi

log "started ${CONTAINER_NAME} with ${RUNTIME} (/dev/nvidia${GPU_INDEX} OK)."
