#!/bin/bash

# Script to launch multiple Boltz2 NIM instances on different ports
# Usage: ./launch_multiple_boltz2.sh [number_of_instances] [starting_port] [memory_per_instance]

# Default values
NUM_INSTANCES=${1:-3}
START_PORT=${2:-8000}
MEMORY_PER_INSTANCE=${3:-16g}

echo "Launching $NUM_INSTANCES Boltz2 NIM instances starting from port $START_PORT"
echo "Memory per instance: $MEMORY_PER_INSTANCE"

# Check if required environment variables are set
if [ -z "$NGC_API_KEY" ]; then
    echo "Error: NGC_API_KEY environment variable is not set"
    echo "Please set it with: export NGC_API_KEY=your_api_key"
    exit 1
fi

if [ -z "$LOCAL_NIM_CACHE" ]; then
    echo "Warning: LOCAL_NIM_CACHE not set, using default: ~/.cache/nim"
    export LOCAL_NIM_CACHE="$HOME/.cache/nim"
fi

# Create cache directory if it doesn't exist
mkdir -p "$LOCAL_NIM_CACHE"

# Function to launch a single instance
launch_instance() {
    local port=$1
    local instance_id=$2
    local container_name="boltz2-nim-$instance_id"
    
    echo "Launching Boltz2 NIM instance $instance_id on port $port..."
    
    docker run -d \
        --name "$container_name" \
        --runtime=nvidia \
        --gpus='"device='$((instance_id % $(nvidia-smi -L | wc -l)))'"' \
        -p "$port:8000" \
        -e NGC_API_KEY \
        -v "$LOCAL_NIM_CACHE:/opt/nim/.cache" \
        --memory="$MEMORY_PER_INSTANCE" \
        --shm-size=2g \
        nvcr.io/nim/mit/boltz2:1.1.0
    
    if [ $? -eq 0 ]; then
        echo "✓ Successfully launched instance $instance_id on port $port (container: $container_name)"
    else
        echo "✗ Failed to launch instance $instance_id on port $port"
    fi
}

# Launch multiple instances
for i in $(seq 0 $((NUM_INSTANCES-1))); do
    port=$((START_PORT + i))
    launch_instance $port $i
    sleep 2  # Small delay between launches
done

echo ""
echo "Launch completed. Checking status of all instances..."
sleep 5

# Check status
echo ""
echo "Container Status:"
docker ps --filter "name=boltz2-nim-" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"

echo ""
echo "To stop all instances, run:"
echo "docker stop \$(docker ps -q --filter 'name=boltz2-nim-')"
echo ""
echo "To remove all instances, run:"
echo "docker rm \$(docker ps -aq --filter 'name=boltz2-nim-')"
echo ""
echo "To check logs for a specific instance:"
echo "docker logs boltz2-nim-0  # for instance 0"
echo "docker logs boltz2-nim-1  # for instance 1"
echo "etc."
