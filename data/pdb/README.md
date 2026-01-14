# PDB Structure References

This directory contains links and instructions for obtaining crystal structures of CDK4 (on-target) and CDK11 (anti-target) proteins with bound inhibitors.

## CDK4 Structures (ON-TARGET)

| PDB ID | Description | Ligand | Resolution | Link |
|--------|-------------|--------|------------|------|
| **5L2I** | CDK4-CycD1-Palbociclib | Palbociclib | 2.38 Å | [RCSB](https://www.rcsb.org/structure/5L2I) |
| **5L2S** | CDK4-CycD1-Ribociclib | Ribociclib | 2.61 Å | [RCSB](https://www.rcsb.org/structure/5L2S) |
| **5L2T** | CDK4-CycD1-Abemaciclib | Abemaciclib | 2.30 Å | [RCSB](https://www.rcsb.org/structure/5L2T) |
| **2W96** | CDK4-CycD3-PD0332991 | PD0332991 | 2.30 Å | [RCSB](https://www.rcsb.org/structure/2W96) |
| **2W99** | CDK4-CycD1-Fascaplysin | Fascaplysin | 2.10 Å | [RCSB](https://www.rcsb.org/structure/2W99) |
| **6GUB** | CDK4-CycD1 apo | None | 2.47 Å | [RCSB](https://www.rcsb.org/structure/6GUB) |

## CDK11 Structures (ANTI-TARGET)

| PDB ID | Description | Ligand | Resolution | Link |
|--------|-------------|--------|------------|------|
| **6GU6** | CDK11B kinase | ATP analog | 1.95 Å | [RCSB](https://www.rcsb.org/structure/6GU6) |
| **6GU7** | CDK11B kinase | Small molecule | 2.20 Å | [RCSB](https://www.rcsb.org/structure/6GU7) |

**Note:** CDK11 structures are more limited compared to CDK4. The kinase domain structures available are suitable for binding predictions.

## Downloading Structures

### Command Line (wget/curl)
```bash
# Download PDB files for CDK4
wget https://files.rcsb.org/download/5L2I.pdb
wget https://files.rcsb.org/download/5L2T.pdb
wget https://files.rcsb.org/download/2W96.pdb

# Download PDB file for CDK11
wget https://files.rcsb.org/download/6GU6.pdb

# Download mmCIF format (preferred for Boltz-2)
wget https://files.rcsb.org/download/5L2I.cif
wget https://files.rcsb.org/download/6GU6.cif
```

### Python Script
```python
import requests
from pathlib import Path

def download_pdb(pdb_id: str, output_dir: str = ".", format: str = "pdb"):
    """Download PDB structure from RCSB."""
    url = f"https://files.rcsb.org/download/{pdb_id}.{format}"
    response = requests.get(url)
    if response.status_code == 200:
        output_path = Path(output_dir) / f"{pdb_id}.{format}"
        output_path.write_text(response.text)
        print(f"Downloaded {pdb_id}.{format}")
    else:
        print(f"Failed to download {pdb_id}: {response.status_code}")

# Download key structures
for pdb_id in ["5L2I", "5L2T", "6GU6"]:
    download_pdb(pdb_id, format="cif")
```

## Recommended Structures for Hackathon

For the hackathon, we recommend using:

1. **CDK4 reference:** `5L2I` (Palbociclib complex) - highest quality, FDA-approved drug
2. **CDK11 reference:** `6GU6` (apo kinase domain) - best available CDK11 structure

These structures can be used to:
- Understand binding site geometry
- Validate Boltz-2 predictions
- Guide ligand design modifications
