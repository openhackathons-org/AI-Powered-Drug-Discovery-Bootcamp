#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
log_dir="${OPENHACKATHON_LOG_DIR:-$repo_root/logs/nims}"
env_file="${OPENHACKATHON_ENV_FILE:-$repo_root/.openhackathon-nims.env}"

molmim_port="${MOLMIM_PORT:-8001}"
boltz2_port="${BOLTZ2_PORT:-8000}"
extra_boltz2_start_port="${EXTRA_BOLTZ2_START_PORT:-8010}"
boltz2_count=1
start_molmim=1

usage() {
    cat <<'EOF'
Usage:
  scripts/openhackathon_services.sh start [--boltz2 N] [--no-molmim]
  scripts/openhackathon_services.sh stop
  scripts/openhackathon_services.sh status
  scripts/openhackathon_services.sh env

Common workflow:
  export NGC_API_KEY=<your-ngc-key>
  scripts/openhackathon_services.sh start --boltz2 2
  source .openhackathon-nims.env
  jupyter-lab

Defaults:
  MolMIM:        http://localhost:8001
  Boltz-2 first: http://localhost:8000
  Boltz-2 extra: starts at http://localhost:8010

Environment overrides:
  MOLMIM_PORT, BOLTZ2_PORT, EXTRA_BOLTZ2_START_PORT
  OPENHACKATHON_LOG_DIR, OPENHACKATHON_ENV_FILE
  NGC_API_KEY, LOCAL_NIM_CACHE, SIF_DIR
EOF
}

pid_file() {
    echo "$log_dir/$1.pid"
}

port_for_boltz2() {
    local idx="$1"
    if [ "$idx" -eq 0 ]; then
        echo "$boltz2_port"
    else
        echo $((extra_boltz2_start_port + idx - 1))
    fi
}

write_env_file() {
    local endpoints=()
    for i in $(seq 0 $((boltz2_count - 1))); do
        endpoints+=("http://localhost:$(port_for_boltz2 "$i")")
    done

    {
        echo "export MOLMIM_URL=\"http://localhost:$molmim_port\""
        echo "export BOLTZ2_URL=\"${endpoints[0]}\""
        local joined
        joined="$(IFS=,; echo "${endpoints[*]}")"
        echo "export BOLTZ2_ENDPOINTS=\"$joined\""
    } > "$env_file"
}

start_service() {
    local name="$1"
    local service="$2"
    local port="$3"
    local gpu="$4"
    local log_file="$log_dir/$name.log"
    local pid_path
    pid_path="$(pid_file "$name")"

    if [ -f "$pid_path" ] && kill -0 "$(cat "$pid_path")" >/dev/null 2>&1; then
        echo "$name already appears to be running (pid $(cat "$pid_path"))."
        return 0
    fi

    echo "Starting $name on port $port using GPU $gpu"
    nohup "$repo_root/scripts/run_nim_apptainer.sh" "$service" "$port" "$gpu" \
        >"$log_file" 2>&1 &
    echo "$!" > "$pid_path"
}

cmd_start() {
    mkdir -p "$log_dir"

    if [ -z "${NGC_API_KEY:-}" ]; then
        echo "Error: NGC_API_KEY is required." >&2
        exit 1
    fi

    local gpu_count=1
    if command -v nvidia-smi >/dev/null 2>&1; then
        gpu_count="$(nvidia-smi -L | wc -l)"
        if [ "$gpu_count" -lt 1 ]; then
            gpu_count=1
        fi
    fi

    if [ "$start_molmim" -eq 1 ]; then
        start_service "molmim" "molmim" "$molmim_port" 0
    fi

    for i in $(seq 0 $((boltz2_count - 1))); do
        start_service "boltz2-$i" "boltz2" "$(port_for_boltz2 "$i")" "$((i % gpu_count))"
        sleep 2
    done

    write_env_file

    echo ""
    echo "Environment written to: $env_file"
    echo "Use:"
    echo "  source $env_file"
    echo "  scripts/openhackathon_services.sh status"
}

cmd_stop() {
    if [ ! -d "$log_dir" ]; then
        echo "No log directory found: $log_dir"
        return 0
    fi

    local found=0
    for pid_path in "$log_dir"/*.pid; do
        [ -e "$pid_path" ] || continue
        found=1
        local pid
        pid="$(cat "$pid_path")"
        if kill -0 "$pid" >/dev/null 2>&1; then
            echo "Stopping $(basename "$pid_path" .pid) (pid $pid)"
            kill "$pid"
        else
            echo "$(basename "$pid_path" .pid) is not running"
        fi
        rm -f "$pid_path"
    done

    if [ "$found" -eq 0 ]; then
        echo "No service pid files found in $log_dir"
    fi
}

cmd_status() {
    if [ -f "$env_file" ]; then
        # shellcheck disable=SC1090
        source "$env_file"
    fi

    echo "Logs: $log_dir"
    "$repo_root/scripts/check_nim_health.sh" molmim "$molmim_port" 1 || true

    local endpoints="${BOLTZ2_ENDPOINTS:-http://localhost:$boltz2_port}"
    IFS=',' read -r -a urls <<< "$endpoints"
    for url in "${urls[@]}"; do
        local port
        port="${url##*:}"
        "$repo_root/scripts/check_nim_health.sh" boltz2 "$port" 1 || true
    done
}

cmd_env() {
    if [ ! -f "$env_file" ]; then
        write_env_file
    fi
    cat "$env_file"
}

cmd="${1:-}"
shift || true

while [ "$#" -gt 0 ]; do
    case "$1" in
        --boltz2)
            boltz2_count="$2"
            shift 2
            ;;
        --no-molmim)
            start_molmim=0
            shift
            ;;
        --help|-h)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            usage >&2
            exit 1
            ;;
    esac
done

case "$cmd" in
    start) cmd_start ;;
    stop) cmd_stop ;;
    status) cmd_status ;;
    env) cmd_env ;;
    help|--help|-h|"") usage ;;
    *)
        echo "Unknown command: $cmd" >&2
        usage >&2
        exit 1
        ;;
esac
