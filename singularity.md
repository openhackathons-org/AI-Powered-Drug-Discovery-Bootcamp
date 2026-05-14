# Singularity/Apptainer Deployment

This bootcamp should run on clusters that do not provide Docker or the NVIDIA
Container Runtime. The recommended HPC path is to run the NIM containers with
Apptainer or Singularity and `--nv`.

The Docker commands in NVIDIA NIM documentation map host ports with `-p`.
Apptainer normally shares the host network namespace, so these scripts set
`NIM_HTTP_API_PORT` instead of using Docker-style port mapping.

## Prerequisites

- A GPU allocation on a compute node.
- `apptainer` or `singularity` available in `PATH`.
- An NGC API key exported as `NGC_API_KEY`.
- Enough local or shared storage for NIM model caches.

```bash
export NGC_API_KEY=<your-ngc-key>
export LOCAL_NIM_CACHE=${LOCAL_NIM_CACHE:-$HOME/.cache/nim}
mkdir -p "$LOCAL_NIM_CACHE"
```

## Quick Start

For most learners, use the service wrapper. It hides image pulls, PIDs, logs,
ports, and endpoint environment variables:

```bash
export NGC_API_KEY=<your-ngc-key>
scripts/openhackathon_services.sh start --boltz2 2
source .openhackathon-nims.env
scripts/openhackathon_services.sh status
jupyter-lab
```

This starts:

- MolMIM at `http://localhost:8001`
- Boltz-2 at `http://localhost:8000`
- Additional Boltz-2 endpoints at `http://localhost:8010`, `8011`, ...

Stop services with:

```bash
scripts/openhackathon_services.sh stop
```

Logs are written to `logs/nims/` by default.

## Manual Start: MolMIM

The notebooks expect MolMIM at `http://localhost:8001` by default.

```bash
scripts/run_nim_apptainer.sh molmim 8001 0
```

In another terminal on the same node:

```bash
scripts/check_nim_health.sh molmim 8001 1
curl -X POST http://localhost:8001/embedding \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{"sequences": ["CC(Cc1ccc(cc1)C(C(=O)O)C)C"]}'
```

## Manual Start: Boltz-2

For a single endpoint:

```bash
scripts/run_nim_apptainer.sh boltz2 8000 0
```

For multiple endpoints on a multi-GPU node, keep the first Boltz-2 endpoint at
`8000` for notebook defaults and use `8010+` for additional replicas:

```bash
scripts/run_nim_apptainer.sh boltz2 8000 0
scripts/launch_multiple_boltz2_apptainer.sh 3 8010
scripts/check_nim_health.sh boltz2 8000 1
scripts/check_nim_health.sh boltz2 8010 3
export BOLTZ2_ENDPOINTS="http://localhost:8000,http://localhost:8010,http://localhost:8011,http://localhost:8012"
```

## Run the Evaluation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r deployment-requirements.txt

cd scoring
python evaluate_submission_parallel_no_mock.py cdk_test_compounds.csv CDK_Validation \
  --endpoints 8000,8010,8011,8012 \
  --max-workers 8 \
  --skip-toxicity --skip-novelty \
  --verbose
```

## Operational Notes

- Use `module load python312` or the site-provided Python module before creating
  a virtual environment if system Python lacks `ensurepip`.
- Keep NIM caches on fast storage when possible. Model download and package
  installation can be slow or appear stuck on heavily shared filesystems.
- If several users share a node, choose non-conflicting ports and set
  `MOLMIM_URL`, `BOLTZ2_URL`, or `BOLTZ2_ENDPOINTS` accordingly.
- To override image tags, set `MOLMIM_IMAGE` or `BOLTZ2_IMAGE`.
