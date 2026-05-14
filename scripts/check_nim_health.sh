#!/usr/bin/env bash
# Copyright (c) 2026, NVIDIA CORPORATION. Licensed under the Apache License, Version 2.0 (the "License") you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0 Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.

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
