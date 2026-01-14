"""
Utility functions for loading hackathon protein and ligand data.

Usage:
    from data.load_data import load_protein, load_ligands, get_binding_site_residues
    
    cdk4_seq = load_protein("CDK4")
    ligands_df = load_ligands()
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd


# Path to data directory
DATA_DIR = Path(__file__).parent.resolve()
FASTA_DIR = DATA_DIR / "fasta"
MSA_DIR = DATA_DIR / "msa"


# Valid proteins for this hackathon (CDK4 = on-target, CDK11 = anti-target)
VALID_PROTEINS = ["CDK4", "CDK11"]


# Binding site residue information (aligned with evaluate_submission.py)
BINDING_SITE_RESIDUES = {
    "CDK4": [
        {"residue": "Lys35", "position": 35},
        {"residue": "Glu71", "position": 71},
        {"residue": "Val96", "position": 96},
        {"residue": "Lys112", "position": 112},
        {"residue": "Asp158", "position": 158},
        {"residue": "Phe164", "position": 164},
        {"residue": "Leu196", "position": 196},
    ],
    "CDK11": [
        {"residue": "Lys41", "position": 41},
        {"residue": "Glu87", "position": 87},
        {"residue": "Val113", "position": 113},
        {"residue": "Lys128", "position": 128},
        {"residue": "Asp175", "position": 175},
        {"residue": "Phe182", "position": 182},
        {"residue": "Asp206", "position": 206},
    ],
}


def load_fasta(fasta_path: Path) -> Tuple[str, str]:
    """
    Load a FASTA file and return (header, sequence).
    
    Args:
        fasta_path: Path to FASTA file
        
    Returns:
        Tuple of (header_line, sequence)
    """
    with open(fasta_path) as f:
        lines = f.readlines()
    
    header = lines[0].strip().lstrip(">")
    sequence = "".join(line.strip() for line in lines[1:] if not line.startswith(">"))
    
    return header, sequence


def load_protein(protein_name: str) -> str:
    """
    Load protein sequence by name.
    
    Args:
        protein_name: One of "CDK4" or "CDK11"
        
    Returns:
        Protein sequence string
    """
    if protein_name not in VALID_PROTEINS:
        raise ValueError(f"Unknown protein: {protein_name}. Valid options: {VALID_PROTEINS}")
    
    fasta_path = FASTA_DIR / f"{protein_name}.fasta"
    if not fasta_path.exists():
        raise FileNotFoundError(f"FASTA file not found: {fasta_path}")
    
    _, sequence = load_fasta(fasta_path)
    return sequence


def load_msa(protein_name: str) -> str:
    """
    Load MSA (A3M format) for a protein.
    
    Args:
        protein_name: One of "CDK4" or "CDK11"
        
    Returns:
        A3M content as string
    """
    if protein_name not in VALID_PROTEINS:
        raise ValueError(f"Unknown protein: {protein_name}. Valid options: {VALID_PROTEINS}")
    
    a3m_path = MSA_DIR / f"{protein_name}.a3m"
    if not a3m_path.exists():
        raise FileNotFoundError(f"A3M file not found: {a3m_path}")
    
    with open(a3m_path) as f:
        return f.read()


def get_binding_site_residues(protein_name: str) -> List[Dict]:
    """
    Get binding site residue information for a protein.
    
    Args:
        protein_name: One of "CDK4" or "CDK11"
        
    Returns:
        List of dicts with 'residue' and 'position' keys
    """
    if protein_name not in BINDING_SITE_RESIDUES:
        raise ValueError(f"Unknown protein: {protein_name}")
    
    return BINDING_SITE_RESIDUES[protein_name]


def get_binding_site_positions(protein_name: str) -> List[int]:
    """
    Get binding site residue positions (indices) for a protein.
    
    Args:
        protein_name: One of "CDK4" or "CDK11"
        
    Returns:
        List of residue position integers
    """
    residues = get_binding_site_residues(protein_name)
    return [r["position"] for r in residues]


def load_ligands(filter_selective: bool = False) -> pd.DataFrame:
    """
    Load known ligands dataset.
    
    Args:
        filter_selective: If True, only return CDK4-selective compounds
        
    Returns:
        DataFrame with ligand information
    """
    csv_path = DATA_DIR / "known_ligands.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Ligands file not found: {csv_path}")
    
    df = pd.read_csv(csv_path)
    
    if filter_selective:
        # Filter for compounds with CDK4 IC50 < 100 nM and selectivity > 10
        def is_selective(row):
            try:
                cdk4 = float(str(row["CDK4_IC50_nM"]).replace(">", "").replace("<", ""))
                selectivity = str(row["selectivity_CDK4_over_CDK11"])
                if selectivity.startswith(">"):
                    sel_val = float(selectivity.replace(">", ""))
                elif selectivity.startswith("<"):
                    sel_val = 0
                else:
                    sel_val = float(selectivity)
                return cdk4 < 100 and sel_val > 10
            except:
                return False
        
        df = df[df.apply(is_selective, axis=1)]
    
    return df


def get_fda_approved_inhibitors() -> pd.DataFrame:
    """
    Get only FDA-approved CDK4/6 inhibitors.
    
    Returns:
        DataFrame with Palbociclib, Ribociclib, Abemaciclib, Trilaciclib
    """
    df = load_ligands()
    fda_compounds = ["PALBOCICLIB", "RIBOCICLIB", "ABEMACICLIB", "TRILACICLIB"]
    return df[df["compound_id"].isin(fda_compounds)]


def get_cdk11_selective_inhibitors() -> pd.DataFrame:
    """
    Get CDK11-selective inhibitors (for comparison - these are BAD for the hackathon goal).
    
    Returns:
        DataFrame with CDK11-selective compounds
    """
    df = load_ligands()
    cdk11_compounds = ["ZNL_05_044", "OTS964", "SR_4835", "COMPOUND_3i"]
    return df[df["compound_id"].isin(cdk11_compounds)]


def print_protein_info(protein_name: str) -> None:
    """Print summary information about a protein."""
    seq = load_protein(protein_name)
    binding_sites = get_binding_site_residues(protein_name)
    
    target_type = "ON-TARGET" if protein_name == "CDK4" else "ANTI-TARGET"
    
    print(f"\n{'='*60}")
    print(f"Protein: {protein_name} ({target_type})")
    print(f"{'='*60}")
    print(f"Sequence length: {len(seq)} amino acids")
    print(f"First 50 residues: {seq[:50]}...")
    print(f"\nBinding site residues:")
    for site in binding_sites:
        print(f"  - {site['residue']} (position {site['position']})")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    # Demo usage
    print("Loading hackathon data...\n")
    print("="*60)
    print("HACKATHON OBJECTIVE:")
    print("  - Design molecules that INHIBIT CDK4 (on-target)")
    print("  - While AVOIDING CDK11 (anti-target)")
    print("="*60)
    
    # Load and print protein info
    for protein in VALID_PROTEINS:
        print_protein_info(protein)
    
    # Load ligands
    ligands = load_ligands()
    print(f"\nLoaded {len(ligands)} known ligands")
    
    print("\n" + "="*60)
    print("FDA-approved CDK4/6 inhibitors (GOOD - selective for CDK4):")
    print("="*60)
    fda = get_fda_approved_inhibitors()
    print(fda[["compound_name", "CDK4_IC50_nM", "CDK11_IC50_nM", "selectivity_CDK4_over_CDK11"]].to_string(index=False))
    
    print("\n" + "="*60)
    print("CDK11-selective inhibitors (BAD - opposite of hackathon goal):")
    print("="*60)
    cdk11 = get_cdk11_selective_inhibitors()
    print(cdk11[["compound_name", "CDK4_IC50_nM", "CDK11_IC50_nM", "selectivity_CDK4_over_CDK11"]].to_string(index=False))
