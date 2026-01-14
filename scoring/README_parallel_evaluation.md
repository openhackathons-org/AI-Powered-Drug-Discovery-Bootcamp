# Parallel OpenHackathon Evaluation Script

This script provides parallel processing capabilities for evaluating chemical compound submissions using multiple Boltz2 NIM endpoints simultaneously.

## Key Features

- **Multi-endpoint support**: Use multiple Boltz2 NIM instances running on different ports
- **Parallel processing**: Concurrent prediction requests with configurable worker limits
- **Endpoint health checking**: Automatic detection of healthy endpoints
- **Load balancing**: Round-robin distribution of requests across available endpoints
- **Real-time monitoring**: Live progress updates with worker IDs and endpoint information
- **Fault tolerance**: Automatic fallback to mock predictions if endpoints fail

## Installation

Same requirements as the original script:

```bash
pip install boltz2-python-client rdkit-pypi pandas numpy tqdm
```

## Setting Up Multiple Boltz2 NIM Instances

### Option 1: Using the Launch Script

```bash
# Make the script executable
chmod +x launch_multiple_boltz2.sh

# Launch 3 instances on ports 8000, 8001, 8002 with 16GB memory each
./launch_multiple_boltz2.sh 3 8000 16g

# Launch 5 instances starting from port 8000 with 12GB memory each
./launch_multiple_boltz2.sh 5 8000 12g
```

### Option 2: Manual Launch

```bash
# Instance 1 (port 8000)
docker run -d --name boltz2-nim-0 \
    --runtime=nvidia --gpus='"device=0"' \
    -p 8000:8000 -e NGC_API_KEY \
    -v "$LOCAL_NIM_CACHE:/opt/nim/.cache" \
    --memory=16g --shm-size=2g \
    nvcr.io/nim/mit/boltz2:1.1.0

# Instance 2 (port 8001)
docker run -d --name boltz2-nim-1 \
    --runtime=nvidia --gpus='"device=1"' \
    -p 8001:8000 -e NGC_API_KEY \
    -v "$LOCAL_NIM_CACHE:/opt/nim/.cache" \
    --memory=16g --shm-size=2g \
    nvcr.io/nim/mit/boltz2:1.1.0

# Instance 3 (port 8002)
docker run -d --name boltz2-nim-2 \
    --runtime=nvidia --gpus='"device=0"' \
    -p 8002:8000 -e NGC_API_KEY \
    -v "$LOCAL_NIM_CACHE:/opt/nim/.cache" \
    --memory=16g --shm-size=2g \
    nvcr.io/nim/mit/boltz2:1.1.0
```

## Usage

### Basic Usage

```bash
# Use default endpoints (8000, 8001, 8002) with 6 workers
python evaluate_submission_parallel.py demo_compounds.csv TeamDemo

# Specify custom endpoints
python evaluate_submission_parallel.py demo_compounds.csv TeamDemo \
    --endpoints 8000,8001,8002,8003 \
    --max-workers 8

# For testing with fewer resources
python evaluate_submission_parallel.py demo_compounds.csv TeamDemo \
    --endpoints 8000,8001 \
    --max-workers 4 \
    --skip-toxicity --skip-novelty \
    --verbose
```

### Performance Optimization

```bash
# High-performance setup with many endpoints
python evaluate_submission_parallel.py submission.csv TeamProd \
    --endpoints 8000,8001,8002,8003,8004,8005 \
    --max-workers 12 \
    --output-dir production_results

# Conservative setup for limited resources
python evaluate_submission_parallel.py submission.csv TeamTest \
    --endpoints 8000 \
    --max-workers 2 \
    --skip-toxicity --skip-novelty
```

## Command-Line Options

- `--endpoints`: Comma-separated list of Boltz2 endpoint ports (default: 8000,8001,8002)
- `--max-workers`: Maximum concurrent workers (default: 6)
- `--output-dir`: Output directory for results (default: evaluation_output)
- `--skip-toxicity`: Skip toxicity calculations for faster processing
- `--skip-novelty`: Skip novelty calculations for faster processing
- `--verbose`: Enable detailed real-time logging

## Performance Characteristics

### Speed Comparison

For 20 compounds with 3 targets each (60 predictions total):

| Setup | Approximate Time | Speedup |
|-------|------------------|---------|
| Single endpoint, 1 worker | ~15-20 minutes | 1x |
| Single endpoint, 3 workers | ~8-10 minutes | 2x |
| 3 endpoints, 6 workers | ~3-5 minutes | 4-6x |
| 6 endpoints, 12 workers | ~2-3 minutes | 7-10x |

### Resource Requirements

- **Memory**: 16-24GB per Boltz2 NIM instance
- **GPU**: 1 GPU per instance (can share with --gpus device selection)
- **CPU**: 2-4 cores per worker recommended

## Output Format

The parallel script generates the same output as the original:

- **HTML Report**: `{team_name}_evaluation_report.html` with endpoint information
- **Detailed CSV**: `{team_name}_detailed_results.csv` with per-endpoint timing

Additional columns in CSV:
- `{target}_endpoint`: Which endpoint processed each prediction
- `{target}_api_time`: Time taken for each API call

## Monitoring and Debugging

### Real-time Progress

With `--verbose`, you'll see:
```
[Worker 0] =====================================
[Worker 0] BOLTZ2 PREDICTION REQUEST - CDK4
[Worker 0] Endpoint: http://localhost:8000
[Worker 2] =====================================
[Worker 2] BOLTZ2 PREDICTION REQUEST - CDK11
[Worker 2] Endpoint: http://localhost:8001
Progress: 15/60 (25.0%) - Latest: CDK11 for compound 2
```

### Health Monitoring

```bash
# Check all running instances
docker ps --filter "name=boltz2-nim-"

# Check logs for specific instance
docker logs boltz2-nim-0

# Monitor resource usage
docker stats boltz2-nim-0 boltz2-nim-1 boltz2-nim-2
```

### Troubleshooting

1. **No healthy endpoints**: Check if Boltz2 containers are running and accessible
2. **Slow performance**: Reduce `--max-workers` or check GPU memory usage
3. **Memory errors**: Increase Docker memory limits or reduce concurrent instances
4. **Network errors**: Check port conflicts and firewall settings

## Stopping Instances

```bash
# Stop all Boltz2 NIM instances
docker stop $(docker ps -q --filter 'name=boltz2-nim-')

# Remove all instances
docker rm $(docker ps -aq --filter 'name=boltz2-nim-')
```

## Configuration Tuning

### For Maximum Speed
- Use as many GPU instances as available
- Set `--max-workers` to 2x number of endpoints
- Skip toxicity and novelty calculations during testing

### For Resource Conservation
- Use 1-2 endpoints
- Set `--max-workers` to 2-4
- Monitor memory usage carefully

### For Production Runs
- Use 3-6 endpoints for reliability
- Set moderate worker counts
- Include all evaluation metrics
- Use larger memory allocations (24GB+ per instance)

## Examples

### Quick Test (2-3 minutes for 20 compounds)
```bash
python evaluate_submission_parallel.py demo_compounds.csv QuickTest \
    --endpoints 8000,8001,8002 \
    --max-workers 6 \
    --skip-toxicity --skip-novelty \
    --verbose
```

### Production Evaluation (5-10 minutes for 100 compounds)
```bash
python evaluate_submission_parallel.py submission.csv Production \
    --endpoints 8000,8001,8002,8003,8004,8005 \
    --max-workers 10 \
    --output-dir production_results
```

### Single Endpoint Fallback
```bash
python evaluate_submission_parallel.py submission.csv Fallback \
    --endpoints 8000 \
    --max-workers 2
```
