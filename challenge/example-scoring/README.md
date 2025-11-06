# Example Scoring Workflow

This folder contains a standalone copy of the `evaluate_submission.py` scoring script used for the OpenHackathon CDK4/6 inhibitor challenge. Use it as a quick-start reference for running the official scoring pipeline on your own list of compounds.

## Prerequisites

- Python 3.8+
- RDKit installed in the active Python environment (e.g. via `conda install -c conda-forge rdkit`).
- Optional: `boltz2-python-client` to receive real affinity predictions from a running BioNeMo Boltz v2 endpoint.
- ChEMBL fingerprint cache (optional, improves novelty scoring) located at `scoring/chembl_data/chembl_fingerprints.pkl`.
  - You can override this location with the `--chembl-path` flag (see "Novelty Scoring" below).

## Files

- `evaluate_submission.py`: Full scoring pipeline script (copied from `scoring/evaluate_submission.py`).

## Running the Scoring Script

From the repository root:

```bash
python challenge/example-scoring/evaluate_submission.py scoring/demo_compounds.csv DemoTeam --output-dir challenge/example-scoring/results
```

Replace the arguments as needed:

- **First argument**: path to a CSV/TXT/SMI file with a `smiles` column (or any column containing SMILES strings).
- **Second argument**: team name or identifier used to prefix the generated outputs.
- **`--output-dir`** (optional): destination folder for dashboards, CSVs, and summaries. Defaults to `evaluation_results` in the current working directory.

### Useful Optional Flags

- `--skip-boltz2`: skip live Boltz affinity predictions and use placeholder IC50 values (useful for environments without API access).
- `--confidence-threshold <float>`: minimum confidence required to accept a Boltz2 prediction (default 0.7 on this branch).
- `--skip-pains`: disable PAINS filtering.
- `--skip-novelty`: skip ChEMBL similarity checks (sets a neutral novelty score).
- `--chembl-path <dir>`: folder containing `chembl_fingerprints.pkl` used for novelty scoring (defaults to `scoring/chembl_data`).

For full flag documentation run:

```bash
python challenge/example-scoring/evaluate_submission.py --help
```

## Output

When the script finishes it writes:

- `<team>_full_results.csv`: Complete per-compound metrics.
- `<team>_top_compounds.csv`: Top-ranked compounds subset.
- `<team>_summary.txt`: Quick textual summary.
- `<team>_dashboard.png`: Visual dashboard with potency, selectivity, and property plots.

These files appear in the directory supplied via `--output-dir`.

## Novelty Scoring

The script scores novelty by comparing each compound's Morgan fingerprint against the ChEMBL fingerprints cache. The default cache path is `scoring/chembl_data/chembl_fingerprints.pkl` relative to the repo root.

- To use an alternative cache location, point `--chembl-path` to a directory that contains a `chembl_fingerprints.pkl` file:

  ```bash
  python challenge/example-scoring/evaluate_submission.py my_compounds.csv MyTeam \
    --output-dir challenge/example-scoring/results \
    --chembl-path /path/to/custom_chembl_cache
  ```
- If the pickle file is absent, the script falls back to treating every compound as novel (novelty score = 1.0).

To generate a cache, run the helper script in the main scoring folder (consult `scoring/README_evaluation_script.md` for instructions), then copy the resulting `chembl_fingerprints.pkl` into your desired directory.



