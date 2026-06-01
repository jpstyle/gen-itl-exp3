#!/bin/bash
# Vulkan/GPU environment for Unity HDRP (and PyTorch) inside the container.
# Sourced by eval_with_xvfb.sh; docker_start.sh passes matching --env values.

ROOT="${SEMANTIC_ASSEMBLER_ROOT:-/home/nonroot/semantic-assembler}"

if [[ -z "${XDG_RUNTIME_DIR:-}" ]]; then
    export XDG_RUNTIME_DIR=/tmp/runtime-nonroot
fi
mkdir -p "${XDG_RUNTIME_DIR}"
chmod 700 "${XDG_RUNTIME_DIR}" 2>/dev/null || true

export LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

_has_gpu_devices=false
[[ -e /dev/nvidia0 ]] && _has_gpu_devices=true

# Only pin NVIDIA ICD when GPU devices are present; otherwise Vulkan init fails with zero GPUs.
if [[ "${_has_gpu_devices}" == true ]]; then
    if [[ -z "${VK_ICD_FILENAMES:-}" ]]; then
        icds=()
        while IFS= read -r -d '' f; do
            icds+=("$f")
        done < <(find /etc/vulkan/icd.d /usr/share/vulkan/icd.d \
            -name '*nvidia*.json' -print0 2>/dev/null | sort -z)
        if ((${#icds[@]} > 0)); then
            VK_ICD_FILENAMES=$(IFS=:; echo "${icds[*]}")
            export VK_ICD_FILENAMES
        fi
    fi
    lvp=/usr/share/vulkan/icd.d/lvp_icd.x86_64.json
    if [[ -n "${VK_ICD_FILENAMES:-}" && -f "${lvp}" && ! -f "${lvp}.disabled" ]]; then
        mv "${lvp}" "${lvp}.disabled" 2>/dev/null || true
    fi
else
    echo "setup_gpu_vulkan_env: /dev/nvidia0 missing — GPU was not passed into this container." >&2
    echo "  Stop and recreate with tools/container_external_scripts/docker_start.sh on the host." >&2
    unset VK_ICD_FILENAMES
fi

_in_container=false
[[ -f /.dockerenv ]] && _in_container=true
[[ "$(id -un 2>/dev/null)" == "nonroot" ]] && _in_container=true
if [[ "${_in_container}" == true && "${_has_gpu_devices}" == true && -n "${VK_ICD_FILENAMES:-}" ]]; then
    while IFS= read -r -d '' bundled; do
        if [[ -f "${bundled}" && ! -f "${bundled}.disabled-by-gpu-setup" ]]; then
            mv "${bundled}" "${bundled}.disabled-by-gpu-setup"
        fi
    done < <(find "${ROOT}/unity/Builds" -name 'libvulkan.so*' ! -name '*.disabled-by-gpu-setup' -print0 2>/dev/null)
fi
unset _in_container _has_gpu_devices lvp icds f
