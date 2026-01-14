# OpenHackathon CDK Inhibitor Evaluation Script

This script evaluates chemical compound submissions for CDK4 selectivity with minimal CDK11 binding. It processes SMILES files and generates comprehensive evaluation reports.

## Features

- **IC50 Prediction**: Predicts binding affinity for CDK4 and CDK11
- **Drug-likeness Assessment**: Calculates QED scores
- **Synthetic Accessibility**: Evaluates ease of synthesis
- **Toxicity Prediction**: Estimates safety profile
- **Novelty Assessment**: Compares against 2.5M ChEMBL compounds
- **Composite Scoring**: Weighted scoring system optimized for selectivity
- **Comprehensive Reports**: Generates dashboards and CSV outputs

## Installation

### 1. Install Dependencies

```bash
pip install rdkit-pypi pandas numpy matplotlib seaborn tqdm
```

For actual Boltz2 predictions (recommended):
```bash
pip install boltz2-python-client
```

### 2. Boltz2 Setup

For local Boltz2 service:
```bash
# The script expects Boltz2 service at http://localhost:8000
# Or set BOLTZ2_URL environment variable
export BOLTZ2_URL=http://localhost:8000
```

For cloud API:
```bash
# Set your API key
export BOLTZ2_API_KEY=your-api-key-here
```

### 2. Ensure ChEMBL Database Exists

The script expects the ChEMBL database in `./chembl_data/`. If not present:
```bash
python create_chembl_database.py
```

## Usage

### Basic Usage

```bash
python evaluate_submission.py <smiles_file> <team_name>
```

### Examples

1. **Evaluate a CSV file**:
```bash
python evaluate_submission.py demo_compounds.csv TeamAlpha
```

2. **Specify output directory**:
```bash
python evaluate_submission.py compounds.csv TeamBeta --output-dir results/beta/
```

3. **Skip toxicity/novelty for faster evaluation**:
```bash
python evaluate_submission.py test.smi TeamGamma --skip-toxicity --skip-novelty
```

4. **Custom ChEMBL database path**:
```bash
python evaluate_submission.py compounds.csv TeamDelta --chembl-path /path/to/chembl_data
```

### Command-line Options

- `smiles_file`: Path to SMILES file (CSV, TXT, or SMI format)
- `team_name`: Team identifier for output files
- `--output-dir`: Directory for results (default: evaluation_results)
- `--chembl-path`: Path to ChEMBL database (default: ./chembl_data)
- `--skip-toxicity`: Skip toxicity predictions
- `--skip-novelty`: Skip novelty assessment
- `--confidence-threshold`: Minimum confidence for IC50 predictions (default: 0.7)

## Input File Format

### CSV Format
```csv
compound_id,smiles
COMP_001,CC1=C(C(=O)N(C2=CC=CC=C2)N1C)C3=CC=C(C=C3)Cl
COMP_002,CC1=CC=C(C=C1)C2=CC(=NN2C3=CC=C(C=C3)S(=O)(=O)N)C(F)(F)F
```

### TXT/SMI Format (tab-separated)
```
CC1=C(C(=O)N(C2=CC=CC=C2)N1C)C3=CC=C(C=C3)Cl
CC1=CC=C(C=C1)C2=CC(=NN2C3=CC=C(C=C3)S(=O)(=O)N)C(F)(F)F
```

## Output Files

The script generates the following files in the output directory:

1. **`<team_name>_dashboard.png`**: Visual evaluation dashboard with 8 panels
2. **`<team_name>_top_compounds.csv`**: Top 25 compounds with detailed scores
3. **`<team_name>_full_results.csv`**: Complete evaluation data for all compounds
4. **`<team_name>_summary.txt`**: Summary statistics and top performers

## Scoring System

The composite score uses weighted components:

- **Binding Affinity (25%)**: On-target potency (CDK4)
- **CDK11 Avoidance (20%)**: Minimal off-target binding
- **Selectivity (15%)**: IC50 ratio between CDK11 and on-targets
- **Drug-likeness (15%)**: QED score
- **Synthetic Accessibility (10%)**: Ease of synthesis
- **Toxicity (10%)**: Safety profile
- **Novelty (5%)**: Uniqueness compared to ChEMBL

## Dashboard Visualization

The evaluation dashboard includes:

1. **Score Distributions**: Box plots of all scoring components
2. **IC50 Comparison**: Median IC50 values for each target
3. **Top Compounds**: Scatter plot of QED vs potency
4. **Novelty Distribution**: Histogram of ChEMBL similarities
5. **Composite Scores**: Distribution of final scores
6. **Selectivity Analysis**: Selectivity ratio vs CDK11 avoidance
7. **Drug Properties**: Molecular weight vs LogP
8. **Score Breakdown**: Component contributions for top 10 compounds

## Environment Variables

- `BOLTZ2_URL`: Boltz2 API endpoint (default: http://localhost:8000)
- `BOLTZ2_API_KEY`: API key for Boltz2 predictions

## Demo Run

Test the script with provided demo compounds:

```bash
# Run evaluation on demo compounds
python evaluate_submission.py demo_compounds.csv DemoTeam

# Check results
ls evaluation_results/
cat evaluation_results/DemoTeam_summary.txt
```

## Troubleshooting

1. **"ChEMBL fingerprints not found"**: Run `create_chembl_database.py` first
2. **"No valid SMILES found"**: Check input file format and SMILES validity
3. **Memory issues**: Process smaller batches or use `--skip-novelty`
4. **Missing dependencies**: Install all required packages

## Notes

- The script uses mock predictions for IC50 and toxicity in demo mode
- For production use, configure Boltz2 API credentials
- ChEMBL database requires ~10GB disk space
- Processing time depends on number of compounds and enabled features

## License

This evaluation framework is provided for the OpenHackathon challenge.
