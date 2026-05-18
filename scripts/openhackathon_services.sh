#!/usr/bin/env bash
# Copyright (c) 2026, NVIDIA CORPORATION. Licensed under the Apache License, Version 2.0 (the "License") you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0 Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
log_dir="${OPENHACKATHON_LOG_DIR:-$repo_root/logs/nims}"
env_file="${OPENHACKATHON_ENV_FILE:-$repo_root/.openhackathon-nims.env}"

molmim_port="${MOLMIM_PORT:-8001}"
boltz2_port="${BOLTZ2_PORT:-8000}"
extra_boltz2_start_port="${EXTRA_BOLTZ2_START_PORT:-8010}"
boltz2_count=1
start_molmim=1
molmim_url_override=""
external_molmim_url=""
auto_ports="${OPENHACKATHON_AUTO_PORTS:-1}"
detected_arch="${OPENHACKATHON_ARCH:-$(uname -m)}"
hosted_molmim_url="${OPENHACKATHON_HOSTED_MOLMIM_URL:-https://health.api.nvidia.com/v1/biology/nvidia/molmim}"
molmim_mode="${OPENHACKATHON_MOLMIM_MODE:-auto}"
wait_for_ready="${OPENHACKATHON_WAIT_FOR_READY:-1}"
ready_poll_seconds="${OPENHACKATHON_READY_POLL_SECONDS:-10}"
molmim_ready_timeout="${OPENHACKATHON_MOLMIM_READY_TIMEOUT:-900}"
boltz2_ready_timeout="${OPENHACKATHON_BOLTZ2_READY_TIMEOUT:-1800}"
resolved_boltz2_ports=()
reserved_ports=()

usage() {
    cat <<'EOF'
Usage:
  scripts/openhackathon_services.sh start [--boltz2 N] [--molmim auto|local|hosted|none] [--molmim-url URL]
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

MolMIM selection:
  auto    x86_64/amd64 tries local MolMIM first, then hosted fallback.
          aarch64/arm64 uses hosted MolMIM by default.
  local   require local MolMIM NIM.
  hosted  use hosted MolMIM and do not launch local MolMIM.
  none    do not configure or launch MolMIM.

Environment overrides:
  MOLMIM_PORT, BOLTZ2_PORT, EXTRA_BOLTZ2_START_PORT
  OPENHACKATHON_LOG_DIR, OPENHACKATHON_ENV_FILE
  OPENHACKATHON_AUTO_PORTS=0 to fail instead of picking a free port
  OPENHACKATHON_CONTAINER_RUNTIME=auto|apptainer|singularity|docker
  OPENHACKATHON_DOCKER_GPU_MODE=device|all
  OPENHACKATHON_DOCKER_USE_NVIDIA_RUNTIME=0|1
  OPENHACKATHON_MOLMIM_MODE=auto|local|hosted|none
  OPENHACKATHON_HOSTED_MOLMIM_URL=https://health.api.nvidia.com/v1/biology/nvidia/molmim
  OPENHACKATHON_WAIT_FOR_READY=0 to return immediately after launching
  OPENHACKATHON_MOLMIM_READY_TIMEOUT, OPENHACKATHON_BOLTZ2_READY_TIMEOUT
  NGC_API_KEY, NVIDIA_API_KEY, MOLMIM_API_KEY, LOCAL_NIM_CACHE, LOCAL_NIM_WORKSPACE, SIF_DIR
  MOLMIM_IMAGE, BOLTZ2_IMAGE
EOF
}

pid_file() {
    echo "$log_dir/$1.pid"
}

port_for_boltz2() {
    local idx="$1"
    if [ "${#resolved_boltz2_ports[@]}" -gt "$idx" ]; then
        echo "${resolved_boltz2_ports[$idx]}"
        return 0
    fi

    if [ "$idx" -eq 0 ]; then
        echo "$boltz2_port"
    else
        echo $((extra_boltz2_start_port + idx - 1))
    fi
}

port_is_free() {
    local port="$1"
    local reserved
    for reserved in "${reserved_ports[@]}"; do
        if [ "$reserved" = "$port" ]; then
            return 1
        fi
    done

    if command -v ss >/dev/null 2>&1; then
        ! ss -ltn | awk '{print $4}' | grep -Eq "(^|:)$port$"
    else
        ! (echo >/dev/tcp/127.0.0.1/"$port") >/dev/null 2>&1
    fi
}

find_free_port() {
    local port="$1"
    local limit=$((port + 200))
    while [ "$port" -le "$limit" ]; do
        if port_is_free "$port"; then
            echo "$port"
            return 0
        fi
        port=$((port + 1))
    done
    return 1
}

resolve_port() {
    local name="$1"
    local desired="$2"
    local actual

    if port_is_free "$desired"; then
        echo "$desired"
        return 0
    fi

    if [ "$auto_ports" != "1" ]; then
        echo "Error: port $desired for $name is already in use. Choose another port or enable OPENHACKATHON_AUTO_PORTS=1." >&2
        exit 1
    fi

    actual="$(find_free_port "$((desired + 1))")" || {
        echo "Error: could not find a free port for $name after $desired." >&2
        exit 1
    }

    echo "Port $desired for $name is in use; using $actual instead." >&2
    echo "$actual"
}

service_is_running() {
    local name="$1"
    local pid_path
    pid_path="$(pid_file "$name")"

    [ -f "$pid_path" ] && kill -0 "$(cat "$pid_path")" >/dev/null 2>&1
}

stop_service_by_name() {
    local name="$1"
    local pid_path
    pid_path="$(pid_file "$name")"

    [ -f "$pid_path" ] || return 0
    local pid
    pid="$(cat "$pid_path")"
    if kill -0 "$pid" >/dev/null 2>&1; then
        echo "Stopping $name (pid $pid)"
        kill -- "-$pid" >/dev/null 2>&1 || kill "$pid" >/dev/null 2>&1 || true
        sleep 1
        if kill -0 "$pid" >/dev/null 2>&1; then
            kill -KILL -- "-$pid" >/dev/null 2>&1 || kill -KILL "$pid" >/dev/null 2>&1 || true
        fi
    fi
    rm -f "$pid_path"
}

env_url_port() {
    local url="$1"
    echo "${url##*:}"
}

is_arm_arch() {
    case "$detected_arch" in
        aarch64|arm64)
            return 0
            ;;
        *)
            return 1
            ;;
    esac
}

is_external_url() {
    local url="$1"
    [ -n "$url" ] || return 1
    case "$url" in
        http://localhost:*|http://127.0.0.1:*|"")
            return 1
            ;;
        http://*|https://*)
            return 0
            ;;
        *)
            return 1
            ;;
    esac
}

show_service_log_tail() {
    local name="$1"
    local log_file="$log_dir/$name.log"
    if [ -f "$log_file" ]; then
        echo "Last lines from $log_file:" >&2
        tail -40 "$log_file" >&2 || true
    fi
}

wait_until_ready() {
    local service="$1"
    local endpoint="$2"
    local timeout="$3"
    local name="${4:-}"
    local deadline=$((SECONDS + timeout))

    if [ "$wait_for_ready" != "1" ]; then
        return 0
    fi

    echo "Waiting up to ${timeout}s for $service at $endpoint"
    while [ "$SECONDS" -le "$deadline" ]; do
        if "$repo_root/scripts/check_nim_health.sh" "$service" "$endpoint" 1 >/dev/null 2>&1; then
            echo "$service is ready: $endpoint"
            return 0
        fi

        if [ -n "$name" ] && ! service_is_running "$name"; then
            echo "$service process $name exited before becoming ready." >&2
            show_service_log_tail "$name"
            return 1
        fi

        sleep "$ready_poll_seconds"
    done

    echo "$service did not become ready within ${timeout}s: $endpoint" >&2
    if [ -n "$name" ]; then
        show_service_log_tail "$name"
    fi
    return 1
}

validate_molmim_mode() {
    case "$molmim_mode" in
        auto|local|hosted|none)
            ;;
        *)
            echo "Error: MolMIM mode must be auto, local, hosted, or none." >&2
            exit 1
            ;;
    esac
}

write_env_file() {
    local endpoints=()
    for i in $(seq 0 $((boltz2_count - 1))); do
        endpoints+=("http://localhost:$(port_for_boltz2 "$i")")
    done

    local output_molmim_url="$external_molmim_url"
    if [ -z "$output_molmim_url" ]; then
        output_molmim_url="${molmim_url_override:-${MOLMIM_URL:-}}"
    fi

    {
        if [ "$start_molmim" -eq 1 ]; then
            echo "export MOLMIM_URL=\"http://localhost:$molmim_port\""
        elif [ -n "$output_molmim_url" ]; then
            echo "export MOLMIM_URL=\"$output_molmim_url\""
        else
            echo "# MOLMIM_URL is not set. Export MOLMIM_URL to a hosted MolMIM endpoint before running MolMIM notebooks."
        fi
        if [ "$start_molmim" -eq 1 ]; then
            echo "export OPENHACKATHON_ACTIVE_MOLMIM_MODE=\"local\""
            echo "export OPENHACKATHON_USE_CMA=\"1\""
        elif [ -n "$output_molmim_url" ]; then
            echo "export OPENHACKATHON_ACTIVE_MOLMIM_MODE=\"hosted\""
            echo "export OPENHACKATHON_USE_CMA=\"0\""
        else
            echo "export OPENHACKATHON_ACTIVE_MOLMIM_MODE=\"none\""
            echo "export OPENHACKATHON_USE_CMA=\"0\""
        fi
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
    CUDA_VISIBLE_DEVICES="$gpu" \
    NVIDIA_VISIBLE_DEVICES="$gpu" \
    APPTAINERENV_CUDA_VISIBLE_DEVICES="$gpu" \
    APPTAINERENV_NVIDIA_VISIBLE_DEVICES="$gpu" \
    SINGULARITYENV_CUDA_VISIBLE_DEVICES="$gpu" \
    SINGULARITYENV_NVIDIA_VISIBLE_DEVICES="$gpu" \
    nohup setsid "$repo_root/scripts/run_nim_container.sh" "$service" "$port" "$gpu" \
        >"$log_file" 2>&1 &
    echo "$!" > "$pid_path"
}

cmd_start() {
    mkdir -p "$log_dir"
    local requested_molmim_url="${MOLMIM_URL:-}"
    validate_molmim_mode

    if [ -z "${NGC_API_KEY:-}" ]; then
        echo "Error: NGC_API_KEY is required." >&2
        exit 1
    fi

    if [ -f "$env_file" ]; then
        # shellcheck disable=SC1090
        source "$env_file"
    fi

    local env_molmim_url="${MOLMIM_URL:-}"
    case "$molmim_mode" in
        auto)
            if [ -n "$molmim_url_override" ]; then
                start_molmim=0
                external_molmim_url="$molmim_url_override"
            elif is_external_url "$requested_molmim_url"; then
                start_molmim=0
                external_molmim_url="$requested_molmim_url"
            elif is_arm_arch; then
                start_molmim=0
                if is_external_url "$env_molmim_url"; then
                    external_molmim_url="$env_molmim_url"
                else
                    external_molmim_url="$hosted_molmim_url"
                fi
            else
                start_molmim=1
                if is_external_url "$env_molmim_url"; then
                    external_molmim_url="$env_molmim_url"
                else
                    external_molmim_url="${molmim_url_override:-${requested_molmim_url:-$hosted_molmim_url}}"
                fi
            fi
            ;;
        local)
            start_molmim=1
            external_molmim_url="${molmim_url_override:-${requested_molmim_url:-$hosted_molmim_url}}"
            ;;
        hosted)
            start_molmim=0
            external_molmim_url="${molmim_url_override:-${requested_molmim_url:-${env_molmim_url:-$hosted_molmim_url}}}"
            ;;
        none)
            start_molmim=0
            external_molmim_url="${molmim_url_override:-${requested_molmim_url:-}}"
            ;;
    esac

    if [ "$start_molmim" -eq 0 ] && [ -z "$external_molmim_url" ]; then
        echo "Warning: MolMIM will not be launched locally and no hosted MOLMIM_URL was provided." >&2
        echo "         Set MOLMIM_URL, pass --molmim-url, or use --molmim auto/local/hosted." >&2
    fi

    echo "Detected architecture: $detected_arch"
    if [ "$start_molmim" -eq 1 ]; then
        echo "MolMIM mode: local with hosted fallback ($external_molmim_url)"
    elif [ -n "$external_molmim_url" ]; then
        echo "MolMIM mode: hosted/external ($external_molmim_url)"
    else
        echo "MolMIM mode: none"
    fi

    local gpu_count=1
    if command -v nvidia-smi >/dev/null 2>&1; then
        gpu_count="$(nvidia-smi -L | wc -l)"
        if [ "$gpu_count" -lt 1 ]; then
            gpu_count=1
        fi
    fi

    if [ "$start_molmim" -eq 1 ]; then
        if service_is_running "molmim" && [ -n "${MOLMIM_URL:-}" ]; then
            molmim_port="$(env_url_port "$MOLMIM_URL")"
        else
            molmim_port="$(resolve_port "molmim" "$molmim_port")"
        fi
        reserved_ports+=("$molmim_port")
        start_service "molmim" "molmim" "$molmim_port" 0
        if ! wait_until_ready "molmim" "http://localhost:$molmim_port" "$molmim_ready_timeout" "molmim"; then
            if [ "$molmim_mode" = "auto" ] && [ -n "$external_molmim_url" ]; then
                echo "Local MolMIM did not become healthy; falling back to hosted/external MolMIM."
                stop_service_by_name "molmim"
                start_molmim=0
            else
                echo "Error: local MolMIM did not become ready." >&2
                exit 1
            fi
        fi
    fi

    local existing_boltz2_endpoints="${BOLTZ2_ENDPOINTS:-}"
    local existing_boltz2_urls=()
    if [ -n "$existing_boltz2_endpoints" ]; then
        IFS=',' read -r -a existing_boltz2_urls <<< "$existing_boltz2_endpoints"
    fi

    for i in $(seq 0 $((boltz2_count - 1))); do
        local desired_port
        local actual_port
        if service_is_running "boltz2-$i" && [ "${#existing_boltz2_urls[@]}" -gt "$i" ]; then
            actual_port="$(env_url_port "${existing_boltz2_urls[$i]}")"
        else
            desired_port="$(port_for_boltz2 "$i")"
            actual_port="$(resolve_port "boltz2-$i" "$desired_port")"
        fi
        resolved_boltz2_ports+=("$actual_port")
        reserved_ports+=("$actual_port")
        start_service "boltz2-$i" "boltz2" "$actual_port" "$((i % gpu_count))"
        sleep 2
    done

    write_env_file

    if [ "$start_molmim" -eq 0 ] && [ -n "$external_molmim_url" ]; then
        wait_until_ready "molmim" "$external_molmim_url" "$molmim_ready_timeout" || {
            echo "Error: hosted/external MolMIM is not ready." >&2
            exit 1
        }
    fi

    for i in $(seq 0 $((boltz2_count - 1))); do
        wait_until_ready "boltz2" "http://localhost:$(port_for_boltz2 "$i")" "$boltz2_ready_timeout" "boltz2-$i" || {
            echo "Error: Boltz-2 endpoint $i is not ready." >&2
            exit 1
        }
    done

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
            kill -- "-$pid" >/dev/null 2>&1 || kill "$pid" >/dev/null 2>&1 || true
            sleep 1
            if kill -0 "$pid" >/dev/null 2>&1; then
                kill -KILL -- "-$pid" >/dev/null 2>&1 || kill -KILL "$pid" >/dev/null 2>&1 || true
            fi
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

    echo "Architecture: $detected_arch"
    echo "MolMIM mode: ${OPENHACKATHON_ACTIVE_MOLMIM_MODE:-$molmim_mode}"
    echo "Logs: $log_dir"
    if [ -n "${MOLMIM_URL:-}" ]; then
        "$repo_root/scripts/check_nim_health.sh" molmim "$MOLMIM_URL" 1 || true
    else
        echo "MolMIM URL is not configured. Set MOLMIM_URL to a hosted endpoint or start local MolMIM."
    fi

    local endpoints="${BOLTZ2_ENDPOINTS:-http://localhost:$boltz2_port}"
    IFS=',' read -r -a urls <<< "$endpoints"
    for url in "${urls[@]}"; do
        "$repo_root/scripts/check_nim_health.sh" boltz2 "$url" 1 || true
    done
}

cmd_env() {
    if [ ! -f "$env_file" ] || [ "$start_molmim" -eq 0 ] || [ -n "$molmim_url_override" ]; then
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
        --molmim)
            molmim_mode="$2"
            shift 2
            ;;
        --no-molmim)
            molmim_mode="none"
            start_molmim=0
            shift
            ;;
        --molmim-url|--hosted-molmim-url)
            molmim_url_override="$2"
            external_molmim_url="$2"
            molmim_mode="hosted"
            start_molmim=0
            shift 2
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
