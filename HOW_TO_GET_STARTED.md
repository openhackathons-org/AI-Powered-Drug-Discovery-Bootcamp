# How To Get Started

Copyright (c) 2026, NVIDIA CORPORATION. Licensed under the Apache License, Version 2.0 (the "License") you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0 Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.

This guide summarizes the dependencies and first-run workflow for implementing the AI-Powered-Drug-Discovery-Bootcamp from this branch. The recommended deployment path for clusters is Apptainer/Singularity because many shared GPU systems do not provide Docker runtime support.

## What You Need

### System Runtime

- A Linux GPU workstation or GPU compute-node allocation.
- NVIDIA driver access on the node where the NIM services run.
- `apptainer` or `singularity` available in `PATH` for the cluster workflow.
- An NGC API key with access to the BioNeMo NIM images.
- Enough local or shared storage for NIM image files and model caches.

Docker with NVIDIA Container Runtime is supported as an alternate workstation path. See [`deployment.md`](deployment.md) for Docker details.

### NIM Services

The bootcamp notebooks expect local BioNeMo NIM endpoints:

- MolMIM NIM for molecular generation, hidden-state encoding, and decoding.
- Boltz-2 NIM for protein-ligand affinity prediction.

The service wrapper starts these services and writes endpoint values to `.openhackathon-nims.env`:

- `MOLMIM_URL`
- `BOLTZ2_URL`
- `BOLTZ2_ENDPOINTS`

Always source `.openhackathon-nims.env` before running notebooks or scoring scripts.

### Python Packages

Install the Python dependencies from [`deployment-requirements.txt`](deployment-requirements.txt):

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r deployment-requirements.txt
```

The current dependency set includes:

```text
numpy
pandas
matplotlib
seaborn
cma
rdkit
biopython
tqdm
requests
sqlalchemy
boltz2-python-client>=0.5.2.post1
jupyterlab
scipy
```

## Quick Start On A Cluster

From the repository root:

```bash
export NGC_API_KEY=<your-ngc-key>

python -m venv .venv
source .venv/bin/activate
pip install -r deployment-requirements.txt

scripts/openhackathon_services.sh start --boltz2 1
source .openhackathon-nims.env
scripts/openhackathon_services.sh status
python scoring/check_dependencies.py

jupyter-lab Start_Here.ipynb
```

This starts one MolMIM endpoint and one Boltz-2 endpoint. That is the recommended starting point for the bootcamp demo workflow.

## Multi-GPU Notes

If you have multiple GPUs allocated, the wrapper can start more than one Boltz-2 endpoint:

```bash
scripts/openhackathon_services.sh start --boltz2 4
source .openhackathon-nims.env
scripts/openhackathon_services.sh status
```

On some Apptainer/Singularity installations, plain `--nv` exposes all GPUs to every container even when `NVIDIA_VISIBLE_DEVICES` is set. Start with one Boltz-2 endpoint unless you have confirmed that the cluster supports strict GPU isolation. See [`singularity.md`](singularity.md) for details.

## Notebook Paths

Start here:

- [`Start_Here.ipynb`](Start_Here.ipynb)

Full path:

- [`tutorials/00_Container_Setup.ipynb`](tutorials/00_Container_Setup.ipynb)
- [`challenge/02_The_Challenge-Designing_CDK4_Inhibitors.ipynb`](challenge/02_The_Challenge-Designing_CDK4_Inhibitors.ipynb)
- [`challenge/03_Hands-On_CDK_Inhibitor_Design.ipynb`](challenge/03_Hands-On_CDK_Inhibitor_Design.ipynb)

Compact path:

- [`mini-hands-on/00_Introduction.ipynb`](mini-hands-on/00_Introduction.ipynb)

Optional deeper dives:

- [`tutorials/01_MolMIMGeneration.ipynb`](tutorials/01_MolMIMGeneration.ipynb)
- [`tutorials/02_ClusterMolMIMEmbeddings.ipynb`](tutorials/02_ClusterMolMIMEmbeddings.ipynb)
- [`tutorials/03_MolMIMInterpolation.ipynb`](tutorials/03_MolMIMInterpolation.ipynb)
- [`tutorials/04_MolMIMOracleControlledGeneration.ipynb`](tutorials/04_MolMIMOracleControlledGeneration.ipynb)
- [`tutorials/05_Suggested_Tools_for_Scoring_Oracles.ipynb`](tutorials/05_Suggested_Tools_for_Scoring_Oracles.ipynb)
- [`tutorials/06_Boltz2_Validation.ipynb`](tutorials/06_Boltz2_Validation.ipynb)

## Running The CDK Challenge

The hands-on CDK notebook defaults to demo mode so it can finish in a live bootcamp session with one Boltz-2 endpoint. For a larger exploratory run, set:

```bash
export OPENHACKATHON_DEMO_MODE=0
```

You can also tune individual search knobs:

```bash
export OPENHACKATHON_CMA_ITERATIONS=10
export OPENHACKATHON_CMA_POPSIZE=20
export OPENHACKATHON_TOP_K_FOR_BOLTZ2=10
```

Then launch Jupyter from the same shell:

```bash
source .openhackathon-nims.env
jupyter-lab challenge/03_Hands-On_CDK_Inhibitor_Design.ipynb
```

## Optional Components

- **ChEMBL novelty cache**: Full novelty scoring uses `scoring/chembl_data/chembl_fingerprints.pkl`. If that file is missing or only present as a Git LFS pointer, the challenge notebook falls back to seed/reference novelty scoring. For full novelty scoring, run `git lfs pull` or rebuild the cache with `cd scoring && python create_chembl_database.py`.
- **ReaSyn synthesis pathway prediction**: ReaSyn is optional. The notebook cells skip cleanly unless a ReaSyn MCP server is running and `REASYN_URL` is set when needed.
- **Docker workstation path**: Use [`deployment.md`](deployment.md) if running on a workstation with Docker and NVIDIA Container Runtime.

## Stop Services

When finished:

```bash
scripts/openhackathon_services.sh stop
```

Logs are written under `logs/nims/` by default.

## Troubleshooting

Check service status first:

```bash
scripts/openhackathon_services.sh status
```

Then verify Python and endpoint dependencies:

```bash
source .openhackathon-nims.env
python scoring/check_dependencies.py
```

If a port is busy on a shared node, the wrapper selects the next free port and records the actual endpoint in `.openhackathon-nims.env`. Do not hard-code `localhost:8000` in notebooks or scoring scripts.

If Apptainer/Singularity cannot isolate GPUs, run one Boltz-2 endpoint per allocation or ask the cluster administrators about NVIDIA Container CLI support for Apptainer.
