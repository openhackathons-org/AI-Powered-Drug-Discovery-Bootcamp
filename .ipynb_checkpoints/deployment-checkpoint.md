# BioNemo MolMIM Deployment Guide

This guide provides instructions for deploying and setting up the NVIDIA MolMIM NIM for the BioNemo Bootcamp tutorials and challenge.

## Prerequisites

Before starting, ensure you have:
- NVIDIA GPU with Docker support
- NGC API key from [NVIDIA NGC](https://org.ngc.nvidia.com/setup/api-key)
- Docker installed and running

## Setup Instructions

Please refer to the [NVIDIA MolMIM NIM docs](https://docs.nvidia.com/nim/bionemo/molmim/latest/index.html) and [QuickStart guide](https://docs.nvidia.com/nim/bionemo/molmim/latest/quickstart-guide.html) for comprehensive information. Additional examples showcasing MolMIM capabilities like clustering molecules and interpolating between molecular structures are available in the [endpoint documentation](https://docs.nvidia.com/nim/bionemo/molmim/latest/endpoints.html#notebooks).

### Step 1: Login to NGC Registry

First login to the nvcr.io docker registry with your API key:

```bash
export NGC_CLI_API_KEY=<PASTE_API_KEY_HERE>
docker login nvcr.io
```

### Step 2: Start MolMIM NIM Container

Run the following command to download and start the MolMIM server. It will pull the docker container and the required model weights from NGC:

```bash
docker run --rm -it --name molmim --runtime=nvidia \
     -e CUDA_VISIBLE_DEVICES=0 \
     -e NGC_CLI_API_KEY \
     -p 8000:8000 \
     nvcr.io/nim/nvidia/molmim:1.0.0
```

### Step 3: Setup Python Environment

Clone this repository, optionally set up a python virtual environment, and install dependencies:

```bash
git clone <repository-url>
cd Bootcamp-BioNemo
python3 -m venv venv
source venv/bin/activate
pip install -r tutorials/requirements.txt
```

### Step 4: Launch Tutorials

After the MolMIM container is running and dependencies are installed, you can start with the tutorials:

```bash
jupyter-lab tutorials/
```

## Tutorial Structure

- **Container Setup**: `tutorials/container_setup.ipynb` - Detailed container deployment guide
- **Lab 1**: Basic MolMIM operations (clustering, generation, interpolation)
- **Lab 2**: Advanced techniques with custom oracles and optimization
- **Challenge**: Apply your knowledge to solve drug discovery problems

## Getting Started

1. Complete the container setup tutorial for detailed deployment instructions
2. Work through Lab 1 to understand MolMIM fundamentals
3. Advance to Lab 2 for guided generation with custom scoring oracles
4. Take on the challenge to apply your skills

## Support

For technical issues or questions:
- Check the [NVIDIA MolMIM documentation](https://docs.nvidia.com/nim/bionemo/molmim/latest/index.html)
- Review the tutorials for step-by-step guidance
- Consult the challenge folder for specific hackathon requirements

## License

Please see [LICENSE.txt](LICENSE.txt) for licensing information.

