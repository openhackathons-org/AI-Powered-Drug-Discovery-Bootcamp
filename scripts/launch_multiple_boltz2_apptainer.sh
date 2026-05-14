#!/usr/bin/env bash
set -euo pipefail

num_instances="${1:-1}"
start_port="${2:-8000}"
log_dir="${BOLTZ2_LOG_DIR:-$PWD/logs/boltz2-apptainer}"

mkdir -p "$log_dir"

gpu_count=1
if command -v nvidia-smi >/dev/null 2>&1; then
    gpu_count="$(nvidia-smi -L | wc -l)"
    if [ "$gpu_count" -lt 1 ]; then
        gpu_count=1
    fi
fi

echo "Launching $num_instances Boltz-2 NIM instance(s)"
echo "Start port: $start_port"
echo "Detected GPUs: $gpu_count"
echo "Logs: $log_dir"

for i in $(seq 0 $((num_instances - 1))); do
    port=$((start_port + i))
    gpu_id=$((i % gpu_count))
    log_file="$log_dir/boltz2-${port}.log"

    echo "Starting Boltz-2 on port $port using GPU $gpu_id"
    nohup "$(dirname "$0")/run_nim_apptainer.sh" boltz2 "$port" "$gpu_id" \
        >"$log_file" 2>&1 &
    echo "$!" >"$log_dir/boltz2-${port}.pid"
    sleep 2
done

echo ""
echo "Started instances. Check readiness with:"
echo "  scripts/check_nim_health.sh boltz2 $start_port $num_instances"
echo ""
echo "Stop them with:"
echo "  kill \$(cat $log_dir/*.pid)"
