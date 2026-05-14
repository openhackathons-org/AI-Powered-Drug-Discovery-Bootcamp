#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  scripts/run_nim_apptainer.sh molmim [port] [gpu_id]
  scripts/run_nim_apptainer.sh boltz2 [port] [gpu_id]

Environment:
  NGC_API_KEY              Required for model download and private NGC pulls.
  LOCAL_NIM_CACHE          Host cache directory. Default: ~/.cache/nim
  SIF_DIR                  Directory for pulled .sif images. Default: ./.sif
  MOLMIM_IMAGE             Default: nvcr.io/nim/nvidia/molmim:1.0.0
  BOLTZ2_IMAGE             Default: nvcr.io/nim/mit/boltz2:1.6.0
  APPTAINER_BIN            Optional explicit runtime binary.
  APPTAINER_PULL_ARGS      Optional extra args for apptainer/singularity pull.
  APPTAINER_RUN_ARGS       Optional extra args for apptainer/singularity run.

Examples:
  export NGC_API_KEY=<your-ngc-key>
  scripts/run_nim_apptainer.sh molmim 8001 0
  scripts/run_nim_apptainer.sh boltz2 8000 1
EOF
}

if [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ] || [ "$#" -lt 1 ]; then
    usage
    exit 0
fi

if [ -z "${NGC_API_KEY:-}" ]; then
    echo "Error: NGC_API_KEY is required." >&2
    exit 1
fi

if [ -n "${APPTAINER_BIN:-}" ]; then
    runtime="$APPTAINER_BIN"
elif command -v apptainer >/dev/null 2>&1; then
    runtime="apptainer"
elif command -v singularity >/dev/null 2>&1; then
    runtime="singularity"
else
    echo "Error: neither apptainer nor singularity was found in PATH." >&2
    exit 1
fi

service="$1"
port="${2:-}"
gpu_id="${3:-0}"

case "$service" in
    molmim)
        image="${MOLMIM_IMAGE:-nvcr.io/nim/nvidia/molmim:1.0.0}"
        port="${port:-8001}"
        sif_name="molmim_${image##*:}.sif"
        ;;
    boltz2)
        image="${BOLTZ2_IMAGE:-nvcr.io/nim/mit/boltz2:1.6.0}"
        port="${port:-8000}"
        sif_name="boltz2_${image##*:}.sif"
        ;;
    *)
        echo "Error: service must be 'molmim' or 'boltz2'." >&2
        usage >&2
        exit 1
        ;;
esac

sif_dir="${SIF_DIR:-$PWD/.sif}"
cache_dir="${LOCAL_NIM_CACHE:-$HOME/.cache/nim}"
mkdir -p "$sif_dir" "$cache_dir"

sif_path="$sif_dir/$sif_name"

if [ ! -f "$sif_path" ]; then
    echo "Pulling $image into $sif_path"
    export APPTAINER_DOCKER_USERNAME="${APPTAINER_DOCKER_USERNAME:-\$oauthtoken}"
    export APPTAINER_DOCKER_PASSWORD="${APPTAINER_DOCKER_PASSWORD:-$NGC_API_KEY}"
    # shellcheck disable=SC2086
    "$runtime" pull ${APPTAINER_PULL_ARGS:-} "$sif_path" "docker://$image"
fi

echo "Starting $service NIM on port $port with GPU $gpu_id"
echo "Cache: $cache_dir"
echo "Image: $sif_path"

# Apptainer/Singularity share the host network namespace by default. Setting
# NIM_HTTP_API_PORT changes the port that the service binds inside that shared
# namespace, so Docker-style -p port mapping is not needed for the common HPC case.
# shellcheck disable=SC2086
exec "$runtime" run \
    --nv \
    --cleanenv \
    --env "NGC_API_KEY=$NGC_API_KEY" \
    --env "NGC_CLI_API_KEY=${NGC_CLI_API_KEY:-$NGC_API_KEY}" \
    --env "NVIDIA_VISIBLE_DEVICES=$gpu_id" \
    --env "CUDA_VISIBLE_DEVICES=0" \
    --env "NIM_CACHE_PATH=/opt/nim/.cache" \
    --env "NIM_HTTP_API_PORT=$port" \
    --bind "$cache_dir:/opt/nim/.cache" \
    ${APPTAINER_RUN_ARGS:-} \
    "$sif_path"
