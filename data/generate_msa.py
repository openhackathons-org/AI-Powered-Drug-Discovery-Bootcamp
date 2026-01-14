#!/usr/bin/env python3
"""
Generate MSA files for CDK4 and CDK11 using NVIDIA hosted MSA-Search API.

Prerequisites:
    - NGC API key set as NVIDIA_API_KEY or NGC_API_KEY environment variable
    
Usage:
    # Set your API key
    export NVIDIA_API_KEY=nvapi-xxxxx
    
    # Run MSA generation
    python generate_msa.py
"""

import json
import os
import sys
import time
import requests
from pathlib import Path
from typing import Optional

# Configuration - Use NVIDIA hosted endpoint
MSA_ENDPOINT = "https://health.api.nvidia.com/v1/biology/colabfold/msa-search/predict"

# Get API key from environment
NVIDIA_API_KEY = os.environ.get("NVIDIA_API_KEY") or os.environ.get("NGC_API_KEY", "")

# Paths
SCRIPT_DIR = Path(__file__).parent.resolve()
FASTA_DIR = SCRIPT_DIR / "fasta"
MSA_DIR = SCRIPT_DIR / "msa"


def load_fasta(fasta_path: Path) -> str:
    """Load sequence from FASTA file."""
    with open(fasta_path) as f:
        lines = f.readlines()
    sequence = "".join(line.strip() for line in lines[1:] if not line.startswith(">"))
    return sequence


def generate_msa(
    sequence: str,
    protein_name: str,
    max_sequences: int = 512,
    e_value: float = 0.0001,
) -> Optional[str]:
    """
    Generate MSA using NVIDIA hosted MSA-Search API.
    
    Args:
        sequence: Protein sequence
        protein_name: Name for logging
        max_sequences: Maximum MSA sequences
        e_value: E-value threshold
        
    Returns:
        A3M formatted MSA string, or None on failure
    """
    print(f"\nGenerating MSA for {protein_name}...")
    print(f"  Sequence length: {len(sequence)} aa")
    print(f"  Endpoint: {MSA_ENDPOINT}")
    
    payload = {
        "sequence": sequence,
        "e_value": e_value,
        "iterations": 1,
        "databases": ["Uniref30_2302", "colabfold_envdb_202108", "PDB70_220313"],
        "search_type": "alphafold2",
        "output_alignment_formats": ["a3m"],
        "max_msa_sequences": max_sequences,
    }
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {NVIDIA_API_KEY}",
    }
    
    try:
        print(f"  Sending request to NVIDIA MSA-Search API...")
        response = requests.post(
            MSA_ENDPOINT,
            headers=headers,
            json=payload,
            timeout=600,  # 10 minute timeout for MSA generation
        )
        
        if response.status_code == 401:
            print(f"  ✗ Authentication failed (401). Check your NVIDIA_API_KEY.")
            print(f"    Set it with: export NVIDIA_API_KEY=nvapi-xxxxx")
            return None
        
        if response.status_code == 403:
            print(f"  ✗ Access forbidden (403). Your API key may not have MSA-Search access.")
            print(f"    To get access:")
            print(f"    1. Visit https://build.nvidia.com/colabfold/msa-search")
            print(f"    2. Click 'Get API Key' or enable the MSA-Search API for your key")
            print(f"    3. Ensure your API key has access to 'colabfold/msa-search'")
            return None
        
        if response.status_code == 429:
            print(f"  ✗ Rate limited (429). Wait a moment and try again.")
            print(f"    The API limits requests. Wait 30-60 seconds between calls.")
            return None
        
        if response.status_code != 200:
            print(f"  ✗ MSA-Search returned status {response.status_code}")
            print(f"    Response: {response.text[:500]}")
            return None
        
        result = response.json()
        
        # Extract A3M alignment from response
        a3m_content = None
        
        # Try different response structures
        if "alignments" in result:
            for db_name, db_data in result["alignments"].items():
                if isinstance(db_data, dict) and "a3m" in db_data:
                    if isinstance(db_data["a3m"], dict) and "alignment" in db_data["a3m"]:
                        a3m_content = db_data["a3m"]["alignment"]
                        print(f"  ✓ Got A3M from {db_name}")
                        break
                    elif isinstance(db_data["a3m"], str):
                        a3m_content = db_data["a3m"]
                        print(f"  ✓ Got A3M from {db_name}")
                        break
        
        # Try direct a3m field
        if a3m_content is None and "a3m" in result:
            if isinstance(result["a3m"], dict) and "alignment" in result["a3m"]:
                a3m_content = result["a3m"]["alignment"]
            elif isinstance(result["a3m"], str):
                a3m_content = result["a3m"]
            print(f"  ✓ Got A3M from direct field")
        
        # Try combined_a3m field
        if a3m_content is None and "combined_a3m" in result:
            a3m_content = result["combined_a3m"]
            print(f"  ✓ Got A3M from combined_a3m field")
        
        if a3m_content is None:
            print(f"  ✗ Could not find A3M in response")
            print(f"    Response keys: {list(result.keys())}")
            # Save raw response for debugging
            debug_path = MSA_DIR / f"{protein_name}_response.json"
            with open(debug_path, "w") as f:
                json.dump(result, f, indent=2)
            print(f"    Saved response to {debug_path} for debugging")
            return None
        
        # Count sequences in MSA
        seq_count = a3m_content.count(">")
        print(f"  ✓ MSA generated with {seq_count} sequences")
        
        return a3m_content
        
    except requests.exceptions.Timeout:
        print(f"  ✗ Request timed out (>600s)")
        return None
    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return None


def save_msa(a3m_content: str, output_path: Path) -> None:
    """Save A3M content to file."""
    with open(output_path, "w") as f:
        f.write(a3m_content)
    print(f"  ✓ Saved to {output_path}")


def main():
    print("="*60)
    print("MSA Generation Script for CDK4 and CDK11")
    print("Using NVIDIA Hosted MSA-Search API")
    print("="*60)
    
    # Check for API key
    if not NVIDIA_API_KEY:
        print("\n✗ NVIDIA_API_KEY not set!")
        print("Please set your API key:")
        print("  export NVIDIA_API_KEY=nvapi-xxxxx")
        print("\nYou can get an API key from: https://build.nvidia.com/")
        sys.exit(1)
    
    print(f"\n✓ API key found (starts with: {NVIDIA_API_KEY[:10]}...)")
    
    # Proteins to process
    proteins = ["CDK4", "CDK11"]
    
    success_count = 0
    for i, protein_name in enumerate(proteins):
        fasta_path = FASTA_DIR / f"{protein_name}.fasta"
        output_path = MSA_DIR / f"{protein_name}.a3m"
        
        if not fasta_path.exists():
            print(f"\n✗ FASTA file not found: {fasta_path}")
            continue
        
        # Load sequence
        sequence = load_fasta(fasta_path)
        
        # Generate MSA
        a3m_content = generate_msa(sequence, protein_name)
        
        if a3m_content:
            save_msa(a3m_content, output_path)
            success_count += 1
        else:
            print(f"  ✗ Failed to generate MSA for {protein_name}")
        
        # Add delay between requests to avoid rate limiting
        if i < len(proteins) - 1:
            print(f"\n  Waiting 35 seconds to avoid rate limiting...")
            time.sleep(35)
    
    print("\n" + "="*60)
    print(f"Completed: {success_count}/{len(proteins)} MSAs generated")
    print("="*60)
    
    return 0 if success_count == len(proteins) else 1


if __name__ == "__main__":
    sys.exit(main())
