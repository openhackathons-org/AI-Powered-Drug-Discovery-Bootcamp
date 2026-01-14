# Hackathon Data Directory

This directory contains pre-compiled protein sequences, structural data, and known ligand information for CDK4 (on-target) and CDK11 (anti-target) proteins.

## Directory Structure

```
data/
├── fasta/              # Protein sequences in FASTA format
│   ├── CDK4.fasta      # Human CDK4 (UniProt: P11802)
│   └── CDK11.fasta     # Human CDK11A kinase domain (UniProt: Q9UQ88)
├── msa/                # Multiple Sequence Alignments (A3M format)
│   ├── CDK4.a3m        # MSA for CDK4 (pre-computed via MSA-Search NIM)
│   └── CDK11.a3m       # MSA for CDK11 (pre-computed via MSA-Search NIM)
├── pdb/                # PDB reference structures
│   └── README.md       # Links to relevant PDB entries
├── known_ligands.csv   # Known CDK inhibitors with binding data
├── load_data.py        # Utility functions for loading data
└── README.md           # This file
```

## Protein Information

### CDK4 (Cyclin-dependent kinase 4) - ON-TARGET
- **UniProt ID:** P11802
- **Sequence Length:** 303 amino acids
- **Key Binding Site Residues:** Lys35, Glu71, Val96, Lys112, Asp158, Phe164, Leu196
- **Role:** Primary on-target for CDK4/6 inhibitors; regulates G1→S cell cycle transition
- **Representative PDB Structures:**
  - `5L2I` - CDK4-Cyclin D1 with Palbociclib
  - `5L2S` - CDK4-Cyclin D1 with Ribociclib
  - `5L2T` - CDK4-Cyclin D1 with Abemaciclib
  - `2W96` - CDK4 with PD0332991 (Palbociclib)

### CDK11 (Cyclin-dependent kinase 11A) - ANTI-TARGET
- **UniProt ID:** Q9UQ88
- **Sequence Length:** 315 amino acids (kinase domain)
- **Key Binding Site Residues:** Lys41, Glu87, Val113, Lys128, Asp175, Phe182, Asp206
- **Role:** Anti-target; CDK11 inhibition can cause toxicity
- **Note:** CDK11A and CDK11B are highly homologous (>95% identity in kinase domain)
- **Representative PDB Structures:**
  - `6GU6` - CDK11B kinase with ATP analog
  - `6GU7` - CDK11B kinase with small molecule

## Known Ligands Dataset

The `known_ligands.csv` file contains curated binding data for CDK inhibitors:

| Column | Description |
|--------|-------------|
| `compound_id` | Unique identifier |
| `compound_name` | Common name |
| `smiles` | Canonical SMILES structure |
| `CDK4_IC50_nM` | IC50 against CDK4 (nM) |
| `CDK11_IC50_nM` | IC50 against CDK11 (nM) |
| `selectivity_CDK4_over_CDK11` | Selectivity ratio |
| `pdb_id` | PDB structure ID (if available) |
| `literature_reference` | PMID or DOI |
| `notes` | Additional information |

### Key Compounds

**FDA-Approved CDK4/6 Inhibitors (Selective - GOOD):**
- **Palbociclib** (Ibrance) - IC50: CDK4=11nM, CDK11=>10,000nM ✓
- **Ribociclib** (Kisqali) - IC50: CDK4=10nM, CDK11=>10,000nM ✓
- **Abemaciclib** (Verzenio) - IC50: CDK4=2nM, CDK11=>10,000nM ✓
- **Trilaciclib** (Cosela) - IC50: CDK4=1nM, CDK11=>10,000nM ✓

**CDK11-Selective Inhibitors (for comparison - BAD for hackathon goal):**
- **ZNL-05-044** - IC50: CDK4=>10,000nM, CDK11=38nM ✗
- **OTS964** - IC50: CDK4=>5,000nM, CDK11=10nM ✗
- **SR-4835** - IC50: CDK4=>5,000nM, CDK11=98nM ✗

## Hackathon Goal

Design molecules that:
1. **Potently inhibit CDK4** (low IC50, ideally <100 nM)
2. **Avoid CDK11** (high IC50, ideally >1,000 nM)
3. **Maximize selectivity ratio** (CDK11_IC50 / CDK4_IC50)

## MSA Files

The MSA files in `msa/` directory were generated using the **MSA-Search NIM** and are formatted in A3M for direct use with Boltz-2.

## Using This Data

```python
# Load protein sequences and ligands
from data.load_data import load_protein, load_ligands, get_binding_site_positions

# Get CDK4 sequence
cdk4_seq = load_protein("CDK4")
print(f"CDK4 sequence length: {len(cdk4_seq)}")

# Get binding site positions for Boltz-2
cdk4_sites = get_binding_site_positions("CDK4")
print(f"CDK4 binding sites: {cdk4_sites}")

# Load known ligands
ligands = load_ligands()
print(f"Loaded {len(ligands)} known ligands")

# Get FDA-approved inhibitors
from data.load_data import get_fda_approved_inhibitors
fda = get_fda_approved_inhibitors()
```

## Data Sources

- **Protein Sequences:** UniProt (https://www.uniprot.org/)
- **Binding Data:** ChEMBL, literature curation
- **PDB Structures:** RCSB PDB (https://www.rcsb.org/)
- **MSA Alignments:** Generated using MSA-Search NIM

## References

1. Finn RS, et al. (2015) "Palbociclib and Letrozole in Advanced Breast Cancer." NEJM. PMID: 25589493
2. Hortobagyi GN, et al. (2016) "Ribociclib as First-Line Therapy for HR-Positive, Advanced Breast Cancer." NEJM. PMID: 26461310
3. Liang J, et al. (2022) "Discovery of selective CDK11 inhibitors." J Med Chem. PMID: 35597007
