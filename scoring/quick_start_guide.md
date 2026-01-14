# Quick Start Guide - OpenHackathon CDK Evaluation

## Overview

This folder contains a complete evaluation system for CDK inhibitor drug discovery submissions. The system evaluates compounds for:
- High affinity to CDK4 (therapeutic target)
- Minimal binding to CDK11 (avoid off-target effects)
- Drug-like properties and synthetic accessibility
- Safety and novelty

## Prerequisites

```bash
# Install required packages
pip install rdkit-pypi pandas numpy matplotlib seaborn tqdm
```

## Files in This Directory

1. **`evaluate_submission.py`** - Main evaluation script
2. **`create_chembl_database.py`** - Creates/updates ChEMBL reference database
3. **`demo_compounds.csv`** - Sample SMILES for testing
4. **`chembl_data/`** - ChEMBL database (2.5M compounds)
   - `chembl_compounds.db` - SQLite database
   - `chembl_fingerprints.pkl` - Pre-computed fingerprints

## Quick Test

```bash
# Run evaluation on demo compounds
python evaluate_submission.py demo_compounds.csv TestRun

# Check results
ls evaluation_results/
cat evaluation_results/TestRun_summary.txt
```

## Evaluate Your Compounds

1. **Prepare your SMILES file** (CSV format):
```csv
compound_id,smiles
COMP_001,CC1=C(C(=O)N(C2=CC=CC=C2)N1C)C3=CC=C(C=C3)Cl
COMP_002,CC1=CC=C(C=C1)C2=CC(=NN2C3=CC=C(C=C3)S(=O)(=O)N)C(F)(F)F
```

2. **Run evaluation**:
```bash
python evaluate_submission.py your_compounds.csv YourTeamName
```

3. **Find results** in `evaluation_results/`:
   - `YourTeamName_dashboard.png` - Visual summary
   - `YourTeamName_top_compounds.csv` - Best 25 compounds
   - `YourTeamName_summary.txt` - Quick statistics
   - `YourTeamName_full_results.csv` - All data

## Understanding the Scores

Each compound receives scores for:
- **IC50 Predictions**: Lower values (higher pIC50) for CDK4, higher for CDK11
- **Selectivity Ratio**: CDK11_IC50 / OnTarget_IC50 (>10 is good)
- **CDK11 Avoidance**: 1.0 = no binding, 0.0 = strong binding
- **QED**: Drug-likeness (0-1, higher is better)
- **SA**: Synthetic accessibility (0-1, higher is easier)
- **Toxicity**: Safety score (0-1, higher is safer)
- **Novelty**: 1.0 if unlike ChEMBL compounds

**Composite Score** = Weighted sum (0-1, higher is better)

## Advanced Options

```bash
# Specify output directory
python evaluate_submission.py compounds.csv TeamA --output-dir results/teamA/

# Skip some evaluations for speed
python evaluate_submission.py compounds.csv TeamB --skip-toxicity --skip-novelty

# Adjust confidence threshold
python evaluate_submission.py compounds.csv TeamC --confidence-threshold 0.8
```

## Tips

1. **File Formats**: Supports CSV, TXT, and SMI files
2. **Compound IDs**: Auto-generated if not provided
3. **Invalid SMILES**: Automatically filtered out
4. **Large Files**: Process in batches if >10,000 compounds
5. **Dashboard**: Best viewed at full resolution

## Troubleshooting

- **"ChEMBL database not found"**: Database is pre-built in `chembl_data/`
- **Memory errors**: Use `--skip-novelty` for large submissions
- **No valid SMILES**: Check your SMILES format with RDKit

## Example Output

```
Evaluation Complete!
Total valid compounds: 1000
Top compound score: 0.825
Median score: 0.612
Compounds with >10-fold selectivity: 247

Results saved to evaluation_results/
```

## Need Help?

Check the detailed documentation:
- `README_evaluation_script.md` - Full script documentation
- `README_openhackathon_evaluation.md` - Notebook documentation

## License

This evaluation framework is provided for the OpenHackathon CDK inhibitor challenge.
