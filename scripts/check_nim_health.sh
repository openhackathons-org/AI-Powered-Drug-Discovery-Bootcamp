#!/usr/bin/env bash
set -euo pipefail

service="${1:-boltz2}"
start_port="${2:-8000}"
num_instances="${3:-1}"

case "$service" in
    molmim)
        path="/v1/health/ready"
        ;;
    boltz2)
        path="/v1/health/ready"
        ;;
    *)
        echo "Error: service must be 'molmim' or 'boltz2'." >&2
        exit 1
        ;;
esac

for i in $(seq 0 $((num_instances - 1))); do
    port=$((start_port + i))
    url="http://localhost:${port}${path}"
    printf "%s " "$url"
    if curl -fsS --max-time 5 "$url"; then
        printf "\n"
    else
        printf "not ready\n"
    fi
done
