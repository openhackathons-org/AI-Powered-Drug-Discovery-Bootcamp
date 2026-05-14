# BioNeMo NIM Deployment Guide

This guide provides instructions for deploying the NVIDIA MolMIM and Boltz-2
NIM services used by the BioNeMo bootcamp tutorials and challenge.

## Prerequisites

Before starting, ensure you have:
- NVIDIA GPU allocation on a compute node
- NGC API key from [NVIDIA NGC](https://org.ngc.nvidia.com/setup/api-key)
- Apptainer/Singularity for HPC clusters, or Docker for local workstations

## Getting Started

1. Start the MolMIM NIM for molecule generation
2. Start one or more Boltz-2 NIM endpoints for affinity prediction
3. Verify service health before launching notebooks
4. Work through the tutorials and challenge


## Setup Instructions

Please refer to the [NVIDIA MolMIM NIM docs](https://docs.nvidia.com/nim/bionemo/molmim/latest/index.html) and [QuickStart guide](https://docs.nvidia.com/nim/bionemo/molmim/latest/quickstart-guide.html) for comprehensive information. Additional examples showcasing MolMIM capabilities like clustering molecules and interpolating between molecular structures are available in the [endpoint documentation](https://docs.nvidia.com/nim/bionemo/molmim/latest/endpoints.html#notebooks).

### Option A: Apptainer/Singularity on HPC

Use this path on clusters where Docker and the NVIDIA Container Runtime are not
available. The scripts pull the NIM images into local `.sif` files and run them
with `--nv`.

```bash
export NGC_API_KEY=<PASTE_API_KEY_HERE>
export LOCAL_NIM_CACHE=${LOCAL_NIM_CACHE:-$HOME/.cache/nim}

# Start MolMIM plus two Boltz-2 endpoints
scripts/openhackathon_services.sh start --boltz2 2
source .openhackathon-nims.env
scripts/openhackathon_services.sh status
```

See [`singularity.md`](singularity.md) for the full cluster workflow, cache
notes, image overrides, and multi-endpoint evaluation commands.

### Option B: Docker on Local GPU Workstations

First login to the `nvcr.io` Docker registry with your NGC API key:

```bash
export NGC_API_KEY=<PASTE_API_KEY_HERE>
export NGC_CLI_API_KEY=$NGC_API_KEY
docker login nvcr.io
```

Start MolMIM:

```bash
docker run --rm -it --name molmim --runtime=nvidia \
     -e CUDA_VISIBLE_DEVICES=0 \
     -e NGC_CLI_API_KEY \
     -p 8001:8000 \
     nvcr.io/nim/nvidia/molmim:1.0.0
```

Start Boltz-2:

```bash
export LOCAL_NIM_CACHE=${LOCAL_NIM_CACHE:-$HOME/.cache/nim}
mkdir -p "$LOCAL_NIM_CACHE"

docker run --rm -it --name boltz2 --runtime=nvidia \
     --shm-size=16G \
     -e NGC_API_KEY \
     -e NIM_LOG=INFO \
     -e NIM_LOG_LEVEL=INFO \
     -e TLLM_LOG_LEVEL=INFO \
     -v "$LOCAL_NIM_CACHE:/opt/nim/.cache" \
     -p 8000:8000 \
     nvcr.io/nim/mit/boltz2:1.6.0
```

### Python Environment

Clone this repository; optionally set up a python virtual environment, and install dependencies:

```bash
git clone https://github.com/openhackathons-org/AI-Powered-Drug-Discovery-Bootcamp.git
cd AI-Powered-Drug-Discovery-Bootcamp
python3 -m venv venv
source venv/bin/activate
pip install -r deployment-requirements.txt
```

### Launch Tutorials

After the NIM services are running and dependencies are installed, start Jupyter
Lab to explore the tutorials and challenge:

```bash
jupyter-lab
```

## Structure

- **Tutorials - Container Setup**: [`tutorials/00_Container_Setup.ipynb`](tutorials/00_Container_Setup.ipynb) - Detailed container deployment guide
- **Tutorials - Lab 1**: Basic MolMIM operations (clustering, generation, interpolation)
- **Tutorials - Lab 2**: Advanced techniques with custom oracles and optimization
- **Challenge**: Apply your knowledge to solve drug discovery problems


## Support

For technical issues or questions:
- Check the [NVIDIA MolMIM documentation](https://docs.nvidia.com/nim/bionemo/molmim/latest/index.html)
- Review the tutorials for step-by-step guidance
- Consult the challenge folder for specific hackathon requirements

## License

Please see [LICENSE.txt](LICENSE.txt) for licensing information.
