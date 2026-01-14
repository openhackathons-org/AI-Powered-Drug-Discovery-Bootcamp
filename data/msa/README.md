# Multiple Sequence Alignments (MSA)

This directory contains Multiple Sequence Alignments (MSA) for CDK4 and CDK11 proteins in A3M format.

## File Format

A3M is a FASTA-like format where:
- First sequence is the query (target protein)
- Lower-case letters indicate insertions relative to the query
- `-` indicates gaps/deletions
- Each sequence has a header line starting with `>`

## Files

| File | Protein | Target Type | Sequences | Size |
|------|---------|-------------|-----------|------|
| `CDK4.a3m` | Human CDK4 | ON-TARGET | 200 | ~74 KB |
| `CDK11.a3m` | Human CDK11A | ANTI-TARGET | 200 | ~76 KB |

These MSAs were generated using the NVIDIA hosted MSA-Search API with the Uniref30_2302 database.

## Generating Full MSAs with NVIDIA Hosted API

The current MSA files contain placeholder sequences. To generate full MSAs using NVIDIA's hosted MSA-Search API:

### Step 1: Get API Access

1. Visit [NVIDIA Build - MSA-Search](https://build.nvidia.com/colabfold/msa-search)
2. Sign in or create an account
3. Click "Get API Key" 
4. Ensure your API key has access to `colabfold/msa-search`

### Step 2: Set Environment Variable

```bash
export NVIDIA_API_KEY=nvapi-xxxxx
```

### Step 3: Run MSA Generation

```bash
cd /path/to/Bootcamp-BioNemo
python data/generate_msa.py
```

### Expected Output

```
============================================================
MSA Generation Script for CDK4 and CDK11
Using NVIDIA Hosted MSA-Search API
============================================================

✓ API key found (starts with: nvapi-xxxxx...)

Generating MSA for CDK4...
  Sequence length: 303 aa
  ✓ MSA generated with 512 sequences
  ✓ Saved to data/msa/CDK4.a3m

Generating MSA for CDK11...
  Sequence length: 315 aa
  ✓ MSA generated with 512 sequences
  ✓ Saved to data/msa/CDK11.a3m

============================================================
Completed: 2/2 MSAs generated
============================================================
```

## Using MSAs with Boltz-2

Boltz-2 can use MSA information for improved structure and affinity predictions:

```python
from boltz2_client import Boltz2Client, Polymer, Ligand, PredictionRequest

# Load MSA from A3M file
def load_a3m(filepath: str) -> str:
    """Load A3M file content."""
    with open(filepath, "r") as f:
        return f.read()

# Create polymer with MSA for improved predictions
msa_content = load_a3m("data/msa/CDK4.a3m")

polymer = Polymer(
    id="A",
    molecule_type="protein",
    sequence=cdk4_sequence,
    msa=msa_content  # Include MSA for improved predictions
)
```

## API Reference

**Endpoint:** `https://health.api.nvidia.com/v1/biology/colabfold/msa-search/predict`

**Request:**
```json
{
    "sequence": "MATSRYEPVAEIGVGAYGTVYKARDPH...",
    "databases": ["uniref30_2302", "colabfold_envdb_202108", "pdb70_220313"],
    "search_type": "alphafold2",
    "output_alignment_formats": ["a3m"],
    "max_msa_sequences": 512,
    "e_value": 0.0001,
    "iterations": 1
}
```

**Response:**
```json
{
    "alignments": {
        "uniref30_2302": {
            "a3m": {
                "alignment": ">query\nMATSRY...\n>hit1\nMATSRY...",
                "format": "a3m"
            }
        }
    }
}
```

## Fallback: Single-Sequence Mode

If MSA is not available, Boltz-2 can run in single-sequence mode:

```python
polymer = Polymer(
    id="A",
    molecule_type="protein",
    sequence=cdk4_sequence,
    # No msa parameter - runs without MSA
)
```

**Note:** Single-sequence mode may have reduced accuracy for structure/affinity predictions.

## Quality Considerations

For best Boltz-2 affinity predictions:
- Use diverse MSAs with 100-1000 sequences
- Include sequences with 30-90% identity to query
- MSA depth correlates with prediction confidence
