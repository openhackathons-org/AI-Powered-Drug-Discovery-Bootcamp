#!/usr/bin/env python3
"""
OpenHackathon CDK Inhibitor Evaluation Script

This script evaluates chemical compound submissions for CDK4/6 selectivity
with minimal CDK11 binding. It processes SMILES files and generates
comprehensive evaluation reports.

Usage:
    python evaluate_submission.py <smiles_file> <team_name> [options]

Example:
    python evaluate_submission.py compounds.csv TeamAlpha --output-dir results/
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import time
import asyncio
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')

# RDKit imports
from rdkit import Chem
from rdkit.Chem import Descriptors, QED, AllChem, DataStructs, FilterCatalog
from rdkit.Chem.Scaffolds import MurckoScaffold
import sqlite3
import pickle
from tqdm import tqdm
import json

# Try to import boltz2 client (optional)
BOLTZ2_AVAILABLE = False
BOLTZ2_IMPORT_ERROR = None

try:
    from boltz2_client import Boltz2Client, Polymer, Ligand, PredictionRequest
    BOLTZ2_AVAILABLE = True
    print("✓ Successfully loaded boltz2-python-client")
except ImportError as e:
    BOLTZ2_AVAILABLE = False
    BOLTZ2_IMPORT_ERROR = str(e)
    print("\n" + "="*80)
    print("NOTICE: boltz2-python-client is not installed or cannot be imported.")
    print("The script will use mock predictions for IC50 values.")
    print("\nTo enable actual Boltz2 predictions, install it with:")
    print("  pip install boltz2-python-client")
    print("="*80 + "\n")

# Configuration
BASE_DIR = Path(__file__).resolve().parent

CONFIG = {
    "boltz2_url": os.environ.get("BOLTZ2_URL", "http://localhost:8000"),
    "boltz2_api_key": os.environ.get("BOLTZ2_API_KEY", ""),
    "chembl_data_path": str(BASE_DIR / "chembl_data"),
    "novelty_cutoff": 0.85,
    "top_n_compounds": 25,
    "confidence_threshold": 0.2,
    "weights": {
        "binding_affinity": 0.25,
        "selectivity": 0.15,
        "cdk11_avoidance": 0.20,
        "qed": 0.15,
        "sa": 0.10,
        "pains": 0.10,
        "novelty": 0.05
    }
}

_CHEMBL_FP_CACHE = None


def get_novelty_cutoff() -> float:
    """Return the configured novelty similarity cutoff."""
    return CONFIG.get("novelty_cutoff", 0.85)


print(CONFIG)

# CDK Protein Information
CDK_PROTEIN_INFO = {
    "CDK4": {
        "sequence": "MATSRYEPVAEIGVGAYGTVYKARDPHSGHFVALKSVRVPNGGGGGGGLPISTVREVALLRRLEAF"
                    "EHPNVVRLMDVCATSRTDREIKVTLVFEHVDQDLRTYLDKAPPPGLPAETIKDLMRQFLRGLDFLH"
                    "ANCIVHRDLKPENILVTSGGTVKLADFGLARIYSYQMALTPVVVTLWYRAPEVLLQSTYATPVDMW"
                    "SVGCIFAEMFRRKPLFCGNSEADQLGKIFDLIGLPPEDDWPRDVSLPRGAFPPRGPRPVQSVVPEM"
                    "EESGAQLLLEMLTFNPHKRISAFRALQHSYLHKDEGNPE",
        "binding_site_residues": [
            {"residue": "Lys35", "position": 35},
            {"residue": "Glu71", "position": 71},
            {"residue": "Val96", "position": 96},
            {"residue": "Lys112", "position": 112},
            {"residue": "Asp158", "position": 158},
            {"residue": "Phe164", "position": 164},
            {"residue": "Leu196", "position": 196}
        ],
    },
    "CDK6": {
        "sequence": "MEKDGLCRADQQYECVAEIGEGAYGKVFKARDLKNGGRFVALKRVRVQTGEEGMPLSTIREVAVLR"
                    "HLETFEHPNVVRLFDVCTVSRTDRETKLTLVFEHVDQDLTTYLDKVPEPGVPTETIKDMMFQLLRG"
                    "LDFLHSHRVVHRDLKPQNILVTSSGQIKLADFGLARIYSFQMALTSVVVTLWYRAPEVLLQSSYAT"
                    "PVDLWSVGCIFAEMFRRKPLFRGSSDVDQLGKILDVIGLPGEEDWPRDVALPRQAFHSKSAQPIEK"
                    "FVTDIDELGKDLLLKCLTFNPAKRISAYSALSHPYFQDLERCKENLDSHLPPSQNTSELNTA",
        "binding_site_residues": [
            {"residue": "Lys43", "position": 43},
            {"residue": "Glu81", "position": 81},
            {"residue": "Val101", "position": 101},
            {"residue": "Lys116", "position": 116},
            {"residue": "Asp163", "position": 163},
            {"residue": "Phe170", "position": 170},
            {"residue": "Leu196", "position": 196}
        ],
    },
    "CDK11": {
        "sequence": "ALQGCRSVEEFQCLNRIEEGTYGVVYRAKDKKTDEIVALKRLKMEKEKEGFPITSLREINTILKAQ"
                    "HPNIVTVREIVVGSNMDKIYIVMNYVEHDLKSLMETMKQPFLPGEVKTLMIQLLRGVKHLHDNWIL"
                    "HRDLKTSNLLLSHAGILKVGDFGLAREYGSPLKAYTPVVVTLWYRAPELLLGAKEYSTAVDMWSVG"
                    "CIFGELLTQKPLFPGKSEIDQINKVFKDLGTPSEKIWPGYSELPAVKKMTFSEHPYNNLRKRFGAL"
                    "LSDQGFDLMNKFLTYFPGRRISAEDGLKHEYFRETPLPIDPSMFPKLVEKY",
        "binding_site_residues": [
            {"residue": "Lys41", "position": 41},
            {"residue": "Glu87", "position": 87},
            {"residue": "Val113", "position": 113},
            {"residue": "Lys128", "position": 128},
            {"residue": "Asp175", "position": 175},
            {"residue": "Phe182", "position": 182},
            {"residue": "Asp206", "position": 206}
        ],
    }
}


def load_smiles_file(file_path: str) -> pd.DataFrame:
    """Load SMILES from various file formats"""
    file_path = Path(file_path)
    
    if file_path.suffix == '.csv':
        df = pd.read_csv(file_path)
        # Find SMILES column
        smiles_cols = [col for col in df.columns if 'smiles' in col.lower()]
        if smiles_cols:
            df['smiles'] = df[smiles_cols[0]]
        elif 'SMILES' in df.columns:
            df['smiles'] = df['SMILES']
    elif file_path.suffix in ['.txt', '.smi']:
        df = pd.read_csv(file_path, sep='\t', names=['smiles'])
    else:
        raise ValueError(f"Unsupported file format: {file_path.suffix}")
    
    # Add compound IDs if not present
    if 'compound_id' not in df.columns:
        df['compound_id'] = [f"COMP_{i+1:04d}" for i in range(len(df))]
    
    return df


def validate_smiles(df: pd.DataFrame) -> pd.DataFrame:
    """Validate SMILES and add molecular objects"""
    print("Validating SMILES...")
    
    valid_mols = []
    canonical_smiles = []
    is_valid = []
    
    for smiles in tqdm(df['smiles'], desc="Validating"):
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                valid_mols.append(mol)
                canonical_smiles.append(Chem.MolToSmiles(mol, canonical=True))
                is_valid.append(True)
            else:
                valid_mols.append(None)
                canonical_smiles.append(None)
                is_valid.append(False)
        except:
            valid_mols.append(None)
            canonical_smiles.append(None)
            is_valid.append(False)
    
    df['mol'] = valid_mols
    df['canonical_smiles'] = canonical_smiles
    df['is_valid'] = is_valid
    
    print(f"Valid SMILES: {df['is_valid'].sum()}/{len(df)}")
    return df


def calculate_qed_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate QED drug-likeness scores"""
    print("Calculating QED scores...")
    
    qed_scores = []
    for mol in tqdm(df['mol'], desc="Calculating QED"):
        if mol is not None:
            qed_scores.append(QED.qed(mol))
        else:
            qed_scores.append(np.nan)
    
    df['qed_score'] = qed_scores
    df['qed_category'] = pd.cut(df['qed_score'], 
                                 bins=[0, 0.25, 0.5, 0.75, 1.0],
                                 labels=['Poor', 'Fair', 'Good', 'Excellent'])
    
    return df


def calculate_sa_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate synthetic accessibility scores"""
    print("Calculating SA scores...")
    
    # Simple SA score approximation
    sa_scores = []
    for mol in tqdm(df['mol'], desc="Calculating SA"):
        if mol is not None:
            # Simplified SA score based on molecular complexity
            score = 0
            score += min(mol.GetNumHeavyAtoms() / 50, 1) * 2
            score += min(len(Chem.GetSymmSSSR(mol)) / 6, 1) * 2
            score += min(Descriptors.BertzCT(mol) / 1000, 1) * 3
            # Calculate heteroatoms (N, O, S, halogens, etc.)
            num_heteroatoms = Descriptors.NumHeteroatoms(mol)
            score += min(num_heteroatoms / 10, 1) * 1
            score += min(Descriptors.NumRotatableBonds(mol) / 10, 1) * 2
            sa_scores.append(score)
        else:
            sa_scores.append(np.nan)
    
    df['sa_score'] = sa_scores
    df['sa_score_normalized'] = 1 - (df['sa_score'] / 10)  # Normalize and invert
    
    return df


def print_rt(message: str):
    """Print message in real-time with immediate flush"""
    print(message, flush=True)
    

def predict_binding_affinity_boltz2(smiles: str, protein_target: str, 
                                   confidence_threshold: float = CONFIG["confidence_threshold"],
                                   verbose: bool = True) -> Dict:
    """Predict IC50 values using Boltz2 Python client"""
    
    start_time = time.time()
    
    print_rt(f"\n{'='*80}")
    print_rt(f"BOLTZ2 PREDICTION REQUEST - {protein_target}")
    print_rt(f"{'='*80}")
    print_rt(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}")
    print_rt(f"SMILES: {smiles}")
    print_rt(f"Protein: {protein_target}")
    print_rt(f"Sequence length: {len(CDK_PROTEIN_INFO[protein_target]['sequence'])} aa")
    binding_sites = CDK_PROTEIN_INFO[protein_target].get("binding_site_residues", [])
    if binding_sites:
        print_rt("Binding site residues (≤9 residues):")
        for site in binding_sites:
            print_rt(f"  - {site['residue']} (pos {site['position']})")
    binding_site_summary = ", ".join(
        f"{site['residue']} (pos {site['position']})" for site in binding_sites
    ) if binding_sites else ""
    
    try:
        if BOLTZ2_AVAILABLE:
            print_rt(f"Mode: ACTUAL Boltz2 Python Client")
            print_rt(f"Endpoint: {CONFIG['boltz2_url']}")
            
            # Initialize Boltz2 client
            client = Boltz2Client(
                base_url=CONFIG['boltz2_url'],
                api_key=CONFIG.get("boltz2_api_key") if CONFIG.get("boltz2_api_key") else None
            )
            
            # Create prediction input
            protein_sequence = CDK_PROTEIN_INFO[protein_target]["sequence"]
            
            # Print input information
            print_rt(f"\nINPUT DETAILS:")
            print_rt(f"  SMILES: {smiles}")
            print_rt(f"  Target: {protein_target}")
            print_rt(f"  Sequence length: {len(protein_sequence)} residues")
            
            # Create protein polymer with full sequence
            protein = Polymer(
                id="A",
                molecule_type="protein",
                sequence=protein_sequence
            )
            
            # Create ligand with affinity prediction enabled
            ligand = Ligand(
                id="LIG",
                smiles=smiles,
                predict_affinity=True
            )
            
            # Create prediction request with minimal parameters for speed
            request = PredictionRequest(
                polymers=[protein],
                ligands=[ligand],
                # Minimal structure parameters for faster execution
                recycling_steps=1,
                sampling_steps=10,  # Minimum allowed value
                diffusion_samples=1,
                # Affinity prediction parameters
                sampling_steps_affinity=50,
                diffusion_samples_affinity=2,
                affinity_mw_correction=True
            )
            
            # Print detailed request information
            print_rt(f"\nBOLTZ2 REQUEST PARAMETERS:")
            print(f"  Protein ID: {protein.id}")
            print(f"  Protein sequence: {protein_sequence[:50]}...{protein_sequence[-50:]}")
            print(f"  Full sequence length: {len(protein_sequence)} residues")
            print(f"  Ligand ID: {ligand.id}")
            print(f"  Ligand SMILES: {smiles}")
            print(f"  Affinity prediction: {ligand.predict_affinity}")
            print(f"\n  Structure parameters:")
            print(f"    - recycling_steps: {request.recycling_steps}")
            print(f"    - sampling_steps: {request.sampling_steps}")
            print(f"    - diffusion_samples: {request.diffusion_samples}")
            print(f"\n  Affinity parameters:")
            print(f"    - sampling_steps_affinity: {request.sampling_steps_affinity}")
            print(f"    - diffusion_samples_affinity: {request.diffusion_samples_affinity}")
            print(f"    - affinity_mw_correction: {request.affinity_mw_correction}")
            
            # Print full sequence if verbose
            if verbose:
                print(f"\n  Full protein sequence:")
                print(f"    {protein_sequence}")
            
            # Run prediction
            print_rt(f"\nSending request to Boltz2...")
            api_start = time.time()
            
            # Define async function to run prediction
            async def run_prediction():
                return await client.predict(request)
            
            # Run async function in sync context
            prediction = asyncio.run(run_prediction())
            
            api_time = time.time() - api_start
            
            # Extract results
            print_rt(f"\nBOLTZ2 PREDICTION RESPONSE")
            print_rt(f"{'-'*80}")
            print_rt(f"Status: SUCCESS")
            
            # Print raw response information
            print_rt(f"Response received in {api_time:.2f} seconds")
            print(f"\nRAW BOLTZ2 RESPONSE:")
            print(f"  Response type: {type(prediction)}")
            
            # Print all attributes of the response
            attrs = [attr for attr in dir(prediction) if not attr.startswith('_') and not callable(getattr(prediction, attr))]
            print(f"  Response attributes: {attrs}")
            
            # Try to dump the full response
            if hasattr(prediction, 'model_dump'):
                try:
                    response_dict = prediction.model_dump()
                    print(f"\n  Full response data:")
                    # Print key response fields
                    for key, value in response_dict.items():
                        if key == 'structures':
                            print(f"    - {key}: {len(value) if isinstance(value, list) else type(value)} structure(s)")
                        elif key == 'affinities':
                            print(f"    - {key}: {value}")
                        elif key == 'confidence_scores':
                            print(f"    - {key}: {value}")
                        else:
                            print(f"    - {key}: {str(value)[:100]}...")
                except Exception as e:
                    print(f"  Could not parse response: {e}")
            
            # Print affinity details if available
            if hasattr(prediction, 'affinities') and prediction.affinities:
                print(f"\n  Affinity predictions found for: {list(prediction.affinities.keys())}")
                for ligand_id, affinity in prediction.affinities.items():
                    print(f"\n  Detailed affinity data for {ligand_id}:")
                    if hasattr(affinity, 'model_dump'):
                        affinity_dict = affinity.model_dump()
                        for key, value in affinity_dict.items():
                            print(f"    - {key}: {value}")
                    else:
                        if hasattr(affinity, 'affinity_pic50'):
                            print(f"    - affinity_pic50: {affinity.affinity_pic50}")
                        if hasattr(affinity, 'affinity_probability_binary'):
                            print(f"    - affinity_probability_binary: {affinity.affinity_probability_binary}")
                        if hasattr(affinity, 'affinity_model_predictions'):
                            print(f"    - affinity_model_predictions: {affinity.affinity_model_predictions}")
            
            # Extract affinity data
            ic50_nm = None
            pic50 = None
            confidence = 0.8  # Default confidence
            
            if hasattr(prediction, 'affinities') and prediction.affinities and "LIG" in prediction.affinities:
                affinity = prediction.affinities["LIG"]
                
                if hasattr(affinity, 'affinity_pic50') and affinity.affinity_pic50:
                    pic50 = affinity.affinity_pic50[0]
                    # Convert pIC50 to IC50 in nM: IC50 (M) = 10^(-pIC50), then convert to nM
                    ic50_nm = 10 ** (-pic50) * 1e9
                    
                    # Use binding probability as confidence
                    if hasattr(affinity, 'affinity_probability_binary') and affinity.affinity_probability_binary:
                        confidence = affinity.affinity_probability_binary[0]
                    
                    # Print affinity results
                    print_rt(f"\nAFFINITY PREDICTION RESULTS:")
                    print_rt(f"  pIC50: {pic50:.2f}")
                    print_rt(f"  IC50: {ic50_nm:.2f} nM")
                    print_rt(f"  Binding probability: {confidence:.1%}")
                    
                    # Interpret the results
                    if pic50 > 7.0:
                        binding_strength = "STRONG"
                    elif pic50 > 5.0:
                        binding_strength = "MODERATE"
                    else:
                        binding_strength = "WEAK"
                    print_rt(f"  Binding strength: {binding_strength}")
                else:
                    print_rt(f"Warning: No pIC50 data in affinity response")
            else:
                print_rt(f"Warning: No affinity data in response, using estimated values")
                # Use mock values based on target
                if protein_target in ["CDK4", "CDK6"]:
                    ic50_nm = np.random.lognormal(np.log(50), 1.5)
                else:
                    ic50_nm = np.random.lognormal(np.log(5000), 1.5)
                pic50 = -np.log10(ic50_nm * 1e-9)
                
                print_rt(f"\nESTIMATED AFFINITY (MOCK):")
                print_rt(f"  pIC50: {pic50:.2f}")
                print_rt(f"  IC50: {ic50_nm:.2f} nM")
            
            print_rt(f"\nConfidence: {confidence:.3f}")
            print_rt(f"Accepted: {confidence >= 0.2}")
            
        else:
            # Fallback to mock if client not available
            print_rt(f"Mode: MOCK predictions (boltz2-python-client not available)")
            
            # Print input information
            print_rt(f"\nINPUT DETAILS:")
            print_rt(f"  SMILES: {smiles}")
            print_rt(f"  Target: {protein_target}")
            print_rt(f"  Sequence length: {len(CDK_PROTEIN_INFO[protein_target]['sequence'])} residues")
            
            if protein_target in ["CDK4", "CDK6"]:
                ic50_nm = np.random.lognormal(np.log(50), 1.5)
            else:
                ic50_nm = np.random.lognormal(np.log(5000), 1.5)
            pic50 = -np.log10(ic50_nm * 1e-9)
            confidence = np.random.uniform(0.65, 0.95)
            api_time = np.random.uniform(0.1, 0.5)
            time.sleep(api_time)
            
            print_rt(f"\nBOLTZ2 PREDICTION RESPONSE")
            print_rt(f"{'-'*80}")
            print_rt(f"Status: MOCK")
            
            print_rt(f"\nAFFINITY PREDICTION RESULTS (MOCK):")
            print_rt(f"  pIC50: {pic50:.2f}")
            print_rt(f"  IC50: {ic50_nm:.2f} nM")
            print_rt(f"  Binding probability: {confidence:.1%}")
            
            # Interpret the results
            if pic50 > 7.0:
                binding_strength = "STRONG"
            elif pic50 > 5.0:
                binding_strength = "MODERATE"
            else:
                binding_strength = "WEAK"
            print_rt(f"  Binding strength: {binding_strength}")
            
            print_rt(f"\nConfidence: {confidence:.3f}")
            print_rt(f"Accepted: {confidence >= confidence_threshold}")
            
    except (ImportError, Exception) as e:
        print(f"\nBOLTZ2 PREDICTION RESPONSE")
        print(f"{'-'*80}")
        print(f"ERROR: {type(e).__name__}: {str(e)}")
        
        # Print request details that caused the error
        print(f"\nRequest that caused error:")
        print(f"  SMILES: {smiles}")
        print(f"  Target: {protein_target}")
        print(f"  Sequence length: {len(CDK_PROTEIN_INFO[protein_target]['sequence'])} residues")
        
        # Try to get more details about the error
        if hasattr(e, '__dict__'):
            print(f"\nError details: {e.__dict__}")
        if hasattr(e, 'response'):
            print(f"Response status: {getattr(e.response, 'status_code', 'N/A')}")
            print(f"Response text: {getattr(e.response, 'text', 'N/A')}")
            
        print(f"\nFalling back to mock predictions")
        
        # Fallback to mock predictions
        if protein_target in ["CDK4", "CDK6"]:
            ic50_nm = np.random.lognormal(np.log(50), 1.5)
        else:
            ic50_nm = np.random.lognormal(np.log(5000), 1.5)
        pic50 = -np.log10(ic50_nm * 1e-9)
        confidence = 0.75
        api_time = 0.1
        
        print(f"\nESTIMATED AFFINITY (MOCK DUE TO ERROR):")
        print(f"  pIC50: {pic50:.2f}")
        print(f"  IC50: {ic50_nm:.2f} nM")
    
    elapsed_time = time.time() - start_time
    print(f"\nTiming:")
    print(f"  API/Model time: {api_time*1000:.1f} ms" if 'api_time' in locals() else "  API/Model time: N/A")
    print(f"  Total time: {elapsed_time*1000:.1f} ms")
    print(f"{'='*80}\n")
    
    if confidence < confidence_threshold:
        return {
            "ic50_nm": np.nan,
            "pic50": np.nan,
            "confidence": confidence,
            "prediction_accepted": False,
            "reason": f"Low confidence ({confidence:.3f} < {confidence_threshold})",
            "binding_site_residues": binding_site_summary
        }
    
    # Ensure pic50 is calculated if only ic50_nm is available
    if pic50 is None and ic50_nm is not None:
        pic50 = -np.log10(ic50_nm * 1e-9)
    
    return {
        "ic50_nm": ic50_nm,
        "pic50": pic50,
        "confidence": confidence,
        "prediction_accepted": True,
        "binding_site_residues": binding_site_summary
    }


def calculate_all_binding_affinities(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """Calculate IC50 values for all compounds"""
    print("Calculating IC50 values...")
    
    # Initialize statistics
    prediction_stats = {
        "total_predictions": 0,
        "accepted_predictions": 0,
        "rejected_predictions": 0,
        "total_time": 0,
        "by_target": {target: {"accepted": 0, "rejected": 0} for target in ["CDK4", "CDK6", "CDK11"]}
    }
    
    for target in ["CDK4", "CDK6", "CDK11"]:
        df[f"{target}_ic50_nm"] = np.nan
        df[f"{target}_pic50"] = np.nan
        df[f"{target}_confidence"] = np.nan
        df[f"{target}_prediction_accepted"] = False
        df[f"{target}_binding_site_residues"] = ""
    
    # Temporarily disable verbose output if not requested
    overall_start = time.time()
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Predicting IC50", disable=verbose):
        if row['mol'] is not None:
            for target in ["CDK4", "CDK6", "CDK11"]:
                # Always show Boltz2 predictions for transparency
                try:
                    result = predict_binding_affinity_boltz2(
                        row['canonical_smiles'], 
                        target,
                        CONFIG["confidence_threshold"],
                        verbose=verbose
                    )
                    
                    prediction_stats["total_predictions"] += 1
                    df.at[idx, f"{target}_binding_site_residues"] = result.get("binding_site_residues", "")
                    
                    if result["prediction_accepted"]:
                        df.at[idx, f"{target}_ic50_nm"] = result["ic50_nm"]
                        df.at[idx, f"{target}_pic50"] = result["pic50"]
                        df.at[idx, f"{target}_confidence"] = result["confidence"]
                        df.at[idx, f"{target}_prediction_accepted"] = True
                        prediction_stats["accepted_predictions"] += 1
                        prediction_stats["by_target"][target]["accepted"] += 1
                    else:
                        prediction_stats["rejected_predictions"] += 1
                        prediction_stats["by_target"][target]["rejected"] += 1
                except Exception as e:
                    print(f"Error during prediction: {e}")
    
    prediction_stats["total_time"] = time.time() - overall_start
    
    # Print summary statistics
    print(f"\n{'='*80}")
    print(f"BOLTZ2 PREDICTION SUMMARY")
    print(f"{'='*80}")
    print(f"Total predictions: {prediction_stats['total_predictions']}")
    print(f"Accepted: {prediction_stats['accepted_predictions']} ({prediction_stats['accepted_predictions']/prediction_stats['total_predictions']*100:.1f}%)")
    print(f"Rejected: {prediction_stats['rejected_predictions']} ({prediction_stats['rejected_predictions']/prediction_stats['total_predictions']*100:.1f}%)")
    print(f"\nBy target:")
    for target in ["CDK4", "CDK6", "CDK11"]:
        stats = prediction_stats["by_target"][target]
        total = stats["accepted"] + stats["rejected"]
        percentage = stats['accepted']/total*100 if total > 0 else 0
        print(f"  {target}: {stats['accepted']}/{total} accepted ({percentage:.1f}%)")
    print(f"\nTotal time: {prediction_stats['total_time']:.2f} seconds")
    print(f"Average time per prediction: {prediction_stats['total_time']/prediction_stats['total_predictions']*1000:.1f} ms")
    print(f"{'='*80}\n")
    
    # Print detailed affinity results for each compound
    print(f"\n{'='*80}")
    print(f"AFFINITY PREDICTION RESULTS PER COMPOUND")
    print(f"{'='*80}")
    for idx, row in df.iterrows():
        if pd.notna(row.get('canonical_smiles')):
            print(f"\nCompound {idx + 1}: {row['canonical_smiles'][:50]}...")
            print(f"  CDK4:  IC50 = {row.get('CDK4_ic50_nm', np.nan):.1f} nM, pIC50 = {row.get('CDK4_pic50', np.nan):.2f}")
            print(f"  CDK6:  IC50 = {row.get('CDK6_ic50_nm', np.nan):.1f} nM, pIC50 = {row.get('CDK6_pic50', np.nan):.2f}")
            print(f"  CDK11: IC50 = {row.get('CDK11_ic50_nm', np.nan):.1f} nM, pIC50 = {row.get('CDK11_pic50', np.nan):.2f}")
            if pd.notna(row.get('selectivity_ratio')):
                print(f"  Selectivity (CDK11/CDK4,6): {row['selectivity_ratio']:.1f}x")
    print(f"{'='*80}\n")
    
    # Calculate selectivity metrics
    df['on_target_pic50'] = df[['CDK4_pic50', 'CDK6_pic50']].mean(axis=1)
    df['on_target_ic50_nm'] = df[['CDK4_ic50_nm', 'CDK6_ic50_nm']].mean(axis=1)
    df['selectivity_ratio'] = df['CDK11_ic50_nm'] / df['on_target_ic50_nm']
    df['selectivity_ratio'] = df['selectivity_ratio'].replace([np.inf, -np.inf], np.nan)
    
    # CDK11 avoidance score
    def calculate_cdk11_avoidance(ic50):
        if pd.isna(ic50):
            return np.nan
        elif ic50 >= 10000:
            return 1.0
        elif ic50 >= 1000:
            return 0.5 + 0.5 * (ic50 - 1000) / 9000
        else:
            return 0.5 * ic50 / 1000
    
    df['cdk11_avoidance'] = df['CDK11_ic50_nm'].apply(calculate_cdk11_avoidance)
    
    return df


def apply_pains_filter(df: pd.DataFrame) -> pd.DataFrame:
    """Apply PAINS filters and annotate compounds."""
    print("Applying PAINS filters...")

    params = FilterCatalog.FilterCatalogParams()
    params.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.PAINS_A)
    params.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.PAINS_B)
    params.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.PAINS_C)
    catalog = FilterCatalog.FilterCatalog(params)

    pains_flags = []
    pains_matches = []

    for mol in tqdm(df['mol'], desc="PAINS"):
        if mol is None:
            pains_flags.append(np.nan)
            pains_matches.append([])
            continue

        if not catalog.HasMatch(mol):
            pains_flags.append(False)
            pains_matches.append([])
        else:
            matches = [entry.GetDescription() for entry in catalog.GetMatches(mol)]
            pains_flags.append(True)
            pains_matches.append(matches)

    df['is_pains'] = pains_flags
    df['pains_alerts'] = [list(filter(None, alerts)) for alerts in pains_matches]
    df['pains_score'] = np.where(df['is_pains'] == True, 0.0, 1.0)

    n_pains = (df['is_pains'] == True).sum()
    print(f"PAINS-positive compounds: {n_pains}/{len(df)} ({n_pains/len(df)*100:.1f}%)")
    if n_pains > 0:
        print("Top PAINS alerts:")
        alert_counts: Dict[str, int] = {}
        for alerts in df.loc[df['is_pains'] == True, 'pains_alerts']:
            for alert in alerts:
                alert_counts[alert] = alert_counts.get(alert, 0) + 1
        for alert, count in sorted(alert_counts.items(), key=lambda item: item[1], reverse=True)[:5]:
            print(f"  {alert}: {count}")

    return df


def load_chembl_fingerprints(chembl_path: str) -> Dict[str, object]:
    """Load ChEMBL fingerprints from pickle file with caching."""
    global _CHEMBL_FP_CACHE

    if _CHEMBL_FP_CACHE is not None:
        return _CHEMBL_FP_CACHE

    fp_path = os.path.join(chembl_path, "chembl_fingerprints.pkl")

    if os.path.exists(fp_path):
        print(f"Loading ChEMBL fingerprints from {fp_path}...")
        with open(fp_path, 'rb') as f:
            _CHEMBL_FP_CACHE = pickle.load(f)
    else:
        print("Warning: ChEMBL fingerprints not found. Using empty reference set.")
        _CHEMBL_FP_CACHE = {}

    return _CHEMBL_FP_CACHE


def calculate_novelty_scores(
    df: pd.DataFrame,
    chembl_path: str = CONFIG["chembl_data_path"],
    similarity_cutoff: Optional[float] = None,
) -> pd.DataFrame:
    """Calculate novelty scores by comparing to ChEMBL database."""

    print("Calculating novelty scores...")

    cutoff = similarity_cutoff if similarity_cutoff is not None else get_novelty_cutoff()

    reference_fps = load_chembl_fingerprints(chembl_path)

    if not reference_fps:
        print("No reference compounds available. Setting all compounds as novel.")
        df['max_chembl_similarity'] = 0.0
        df['is_novel'] = True
        df['novelty_score'] = 1.0
        return df

    query_fps: List[Optional[DataStructs.ExplicitBitVect]] = []
    for mol in df['mol']:
        if mol is not None:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
            query_fps.append(fp)
        else:
            query_fps.append(None)

    max_similarities: List[float] = []
    reference_fp_values = list(reference_fps.values())

    for query_fp in tqdm(query_fps, desc="Computing novelty"):
        if query_fp is None:
            max_similarities.append(np.nan)
            continue

        max_sim = 0.0
        for ref_fp in reference_fp_values:
            sim = DataStructs.TanimotoSimilarity(query_fp, ref_fp)
            if sim > max_sim:
                max_sim = sim
                if max_sim >= cutoff:
                    break

        max_similarities.append(max_sim)

    df['max_chembl_similarity'] = max_similarities
    df['is_novel'] = df['max_chembl_similarity'] < cutoff
    df['novelty_score'] = 1.0 - df['max_chembl_similarity'].fillna(cutoff).clip(upper=cutoff) / cutoff
    df.loc[~df['is_novel'], 'novelty_score'] = 0.0

    n_novel = df['is_novel'].sum()
    print(f"Novel compounds: {n_novel}/{len(df)} ({n_novel/len(df)*100:.1f}%) using cutoff {cutoff:.2f}")

    return df


def normalize_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize all scores to 0-1 range"""
    score_mappings = {
        'on_target_pic50': lambda x: (x - x.min()) / (x.max() - x.min()) if x.max() > x.min() else x.fillna(0.5),
        'selectivity_ratio': lambda x: np.minimum(x / 50, 1.0),
        'cdk11_avoidance': lambda x: x,
        'qed_score': lambda x: x,
        'sa_score_normalized': lambda x: x,
        'pains_score': lambda x: x,
        'novelty_score': lambda x: x
    }
    
    for col, norm_func in score_mappings.items():
        if col in df.columns:
            df[f'{col}_norm'] = norm_func(df[col].fillna(df[col].median()))
    
    return df


def calculate_composite_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate weighted composite scores"""
    weights = CONFIG["weights"]
    
    weight_mapping = {
        'binding_affinity': 'on_target_pic50_norm',
        'selectivity': 'selectivity_ratio_norm',
        'cdk11_avoidance': 'cdk11_avoidance_norm',
        'qed': 'qed_score_norm',
        'sa': 'sa_score_normalized_norm',
        'pains': 'pains_score_norm',
        'novelty': 'novelty_score_norm'
    }
    
    df['composite_score'] = 0
    for weight_key, col_name in weight_mapping.items():
        if col_name in df.columns and weight_key in weights:
            df['composite_score'] += weights[weight_key] * df[col_name]
    
    df['rank'] = df['composite_score'].rank(ascending=False, method='min')
    
    return df


def create_evaluation_dashboard(df: pd.DataFrame, team_name: str, output_dir: Path):
    """Generate evaluation dashboard"""
    print("Creating evaluation dashboard...")
    
    fig = plt.figure(figsize=(20, 24))
    
    # 1. Score distributions
    ax1 = plt.subplot(4, 2, 1)
    score_cols = ['qed_score', 'sa_score_normalized', 'pains_score', 'novelty_score']
    valid_cols = [col for col in score_cols if col in df.columns]
    if valid_cols:
        df[valid_cols].boxplot(ax=ax1)
        ax1.set_title('Score Distributions', fontsize=14)
        ax1.set_ylabel('Score (0-1)')
        plt.xticks(rotation=45)
    
    # 2. IC50 comparison
    ax2 = plt.subplot(4, 2, 2)
    ic50_cols = ['CDK4_ic50_nm', 'CDK6_ic50_nm', 'CDK11_ic50_nm']
    valid_ic50_cols = [col for col in ic50_cols if col in df.columns]
    if valid_ic50_cols:
        ic50_data = df[valid_ic50_cols].median()
        ic50_data.plot(kind='bar', ax=ax2, color=['green', 'blue', 'red'], logy=True)
        ax2.set_title('Median IC50 Values', fontsize=14)
        ax2.set_ylabel('IC50 (nM, log scale)')
        ax2.set_xlabel('Target')
        plt.xticks(rotation=45)
    
    # 3. Top compounds scatter
    ax3 = plt.subplot(4, 2, 3)
    top_25 = df.nsmallest(25, 'rank')
    if 'qed_score' in df.columns and 'on_target_pic50' in df.columns:
        valid = top_25[['qed_score', 'on_target_pic50', 'composite_score']].dropna()
        if not valid.empty:
            scatter = ax3.scatter(valid['qed_score'], valid['on_target_pic50'], 
                                 c=valid['composite_score'], s=100, cmap='viridis')
            ax3.set_xlabel('QED Score')
            ax3.set_ylabel('On-target pIC50')
            ax3.set_title('Top 25 Compounds: QED vs Potency', fontsize=14)
            plt.colorbar(scatter, ax=ax3, label='Composite Score')
        else:
            ax3.set_visible(False)
    
    # 4. Novelty distribution
    ax4 = plt.subplot(4, 2, 4)
    if 'max_chembl_similarity' in df.columns:
        valid_novelty = df['max_chembl_similarity'].dropna()
        if not valid_novelty.empty:
            valid_novelty.hist(bins=30, ax=ax4, alpha=0.7)
            novelty_cutoff = get_novelty_cutoff()
            ax4.axvline(novelty_cutoff, color='red', linestyle='--', 
                        label=f'Cutoff ({novelty_cutoff:.2f})')
            ax4.set_xlabel('Max ChEMBL Similarity')
            ax4.set_ylabel('Count')
            ax4.set_title('Novelty Distribution', fontsize=14)
            ax4.legend()
        else:
            ax4.set_visible(False)
    
    # 5. Composite score distribution
    ax5 = plt.subplot(4, 2, 5)
    if 'composite_score' in df.columns:
        df['composite_score'].hist(bins=50, ax=ax5, alpha=0.7, color='purple')
        ax5.axvline(df['composite_score'].quantile(0.975), color='green', linestyle='--', 
                    label='Top 2.5%')
        ax5.set_xlabel('Composite Score')
        ax5.set_ylabel('Count')
        ax5.set_title('Composite Score Distribution', fontsize=14)
        ax5.legend()
    
    # 6. Selectivity vs CDK11 avoidance
    ax6 = plt.subplot(4, 2, 6)
    if 'selectivity_ratio' in df.columns and 'cdk11_avoidance' in df.columns:
        ax6.scatter(df['selectivity_ratio'], df['cdk11_avoidance'], alpha=0.5)
        ax6.set_xlabel('Selectivity Ratio')
        ax6.set_ylabel('CDK11 Avoidance Score')
        ax6.set_title('Selectivity vs CDK11 Avoidance', fontsize=14)
        ax6.set_xlim(0, 100)
    
    # 7. Drug properties
    ax7 = plt.subplot(4, 2, 7)
    if 'mol' in df.columns:
        mw = [Descriptors.MolWt(mol) if mol else np.nan for mol in df['mol']]
        logp = [Descriptors.MolLogP(mol) if mol else np.nan for mol in df['mol']]
        valid_idx = ~(pd.isna(mw) | pd.isna(logp))
        if valid_idx.sum() > 0:
            ax7.scatter(np.array(mw)[valid_idx], np.array(logp)[valid_idx], alpha=0.5)
            ax7.set_xlabel('Molecular Weight')
            ax7.set_ylabel('LogP')
            ax7.set_title('Drug-like Properties', fontsize=14)
            ax7.axvline(500, color='red', linestyle='--', alpha=0.5)
            ax7.axhline(5, color='red', linestyle='--', alpha=0.5)
    
    # 8. Score breakdown for top compounds
    ax8 = plt.subplot(4, 2, 8)
    top_10 = df.nsmallest(10, 'rank')
    if len(top_10) > 0:
        score_components = []
        for weight_key in CONFIG["weights"]:
            col_map = {
                'binding_affinity': 'on_target_pic50_norm',
                'selectivity': 'selectivity_ratio_norm',
                'cdk11_avoidance': 'cdk11_avoidance_norm',
                'qed': 'qed_score_norm',
                'sa': 'sa_score_normalized_norm',
                'pains': 'pains_score_norm',
                'novelty': 'novelty_score_norm'
            }
            if col_map.get(weight_key) in df.columns:
                score_components.append(
                    top_10[col_map[weight_key]] * CONFIG["weights"][weight_key]
                )
        
        if score_components:
            score_df = pd.DataFrame(score_components).T
            score_df.columns = list(CONFIG["weights"].keys())[:len(score_components)]
            score_df.plot(kind='bar', stacked=True, ax=ax8)
            ax8.set_title('Score Breakdown - Top 10 Compounds', fontsize=14)
            ax8.set_xlabel('Compound Rank')
            ax8.set_ylabel('Score Contribution')
            ax8.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    dashboard_path = output_dir / f"{team_name}_dashboard.png"
    plt.savefig(dashboard_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Dashboard saved to {dashboard_path}")


def generate_report(df: pd.DataFrame, team_name: str, output_dir: Path):
    """Generate evaluation report files"""
    print("Generating evaluation report...")
    
    # Save top compounds
    top_25 = df.nsmallest(CONFIG['top_n_compounds'], 'rank')
    summary_cols = [
        'rank', 'compound_id', 'canonical_smiles', 'composite_score',
        'on_target_pic50', 'on_target_ic50_nm', 'selectivity_ratio', 
        'cdk11_avoidance', 'qed_score', 'sa_score', 'pains_score', 'novelty_score',
        'pains_alerts'
    ]
    valid_cols = [col for col in summary_cols if col in top_25.columns]
    
    top_compounds_path = output_dir / f"{team_name}_top_compounds.csv"
    top_25[valid_cols].to_csv(top_compounds_path, index=False)
    
    # Save full results
    full_results_path = output_dir / f"{team_name}_full_results.csv"
    df.to_csv(full_results_path, index=False)
    
    # Generate summary statistics
    summary_path = output_dir / f"{team_name}_summary.txt"
    with open(summary_path, 'w') as f:
        f.write(f"Evaluation Summary for {team_name}\n")
        f.write(f"{'='*50}\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write(f"Total compounds submitted: {len(df)}\n")
        f.write(f"Valid compounds: {df['is_valid'].sum()}\n")
        f.write(f"Novel compounds: {df['is_novel'].sum() if 'is_novel' in df.columns else 'N/A'}\n\n")
        
        f.write("Top 5 Compounds:\n")
        for idx, row in top_25.head(5).iterrows():
            rank_value = row['rank']
            rank_display = int(rank_value) if pd.notna(rank_value) else 'N/A'
            f.write(f"{rank_display}. {row['compound_id']} - Score: {row['composite_score']:.3f}\n")
            if 'on_target_ic50_nm' in row:
                f.write(f"   IC50: CDK4/6={row['on_target_ic50_nm']:.1f}nM, ")
                f.write(f"CDK11={df.loc[idx, 'CDK11_ic50_nm']:.1f}nM\n")
        
        f.write(f"\nMedian Scores:\n")
        for score_name in ['qed_score', 'sa_score_normalized', 'pains_score', 
                          'novelty_score', 'composite_score']:
            if score_name in df.columns:
                f.write(f"  {score_name}: {df[score_name].median():.3f}\n")
    
    print(f"Report files saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate OpenHackathon CDK inhibitor submissions"
    )
    parser.add_argument("smiles_file", help="Path to SMILES file (CSV, TXT, or SMI)")
    parser.add_argument("team_name", help="Team name for identification")
    parser.add_argument("--output-dir", default="evaluation_results", 
                        help="Output directory for results")
    parser.add_argument("--chembl-path", default="./chembl_data",
                        help="Path to ChEMBL database")
    parser.add_argument("--skip-pains", action="store_true",
                        help="Skip PAINS filtering")
    parser.add_argument("--skip-novelty", action="store_true",
                        help="Skip novelty assessment")
    parser.add_argument("--confidence-threshold", type=float, default=0.7,
                        help="Minimum confidence for IC50 predictions")
    parser.add_argument("--boltz2-url", default=CONFIG["boltz2_url"],
                        help="Base URL for the Boltz2 NIM endpoint")
    parser.add_argument("--novelty-cutoff", type=float, default=CONFIG["novelty_cutoff"],
                        help="Maximum Tanimoto similarity to ChEMBL allowed before a compound is considered non-novel")
    parser.add_argument("--skip-boltz2", action="store_true",
                        help="Skip Boltz2 affinity predictions and use placeholder values")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Show detailed Boltz2 prediction requests and responses")
    
    args = parser.parse_args()
    
    # Update config with command-line arguments
    CONFIG["chembl_data_path"] = args.chembl_path
    CONFIG["confidence_threshold"] = args.confidence_threshold
    CONFIG["novelty_cutoff"] = args.novelty_cutoff
    CONFIG["boltz2_url"] = args.boltz2_url
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Evaluating submission for {args.team_name}")
    print(f"Input file: {args.smiles_file}")
    print(f"Output directory: {output_dir}")
    
    # Load and validate SMILES
    df = load_smiles_file(args.smiles_file)
    df = validate_smiles(df)
    
    if df['is_valid'].sum() == 0:
        print("Error: No valid SMILES found in submission!")
        return 1
    
    # Filter to valid compounds only
    df = df[df['is_valid']].copy()
    
    # Calculate properties
    df = calculate_qed_scores(df)
    df = calculate_sa_scores(df)
    
    # Calculate binding affinities
    if args.skip_boltz2:
        for target in ["CDK4", "CDK6", "CDK11"]:
            df[f"{target}_ic50_nm"] = np.nan
            df[f"{target}_pic50"] = np.nan
            df[f"{target}_confidence"] = np.nan
            df[f"{target}_prediction_accepted"] = False
            binding_sites = CDK_PROTEIN_INFO[target].get("binding_site_residues", [])
            binding_site_summary = ", ".join(
                f"{site['residue']} (pos {site['position']})" for site in binding_sites
            ) if binding_sites else ""
            df[f"{target}_binding_site_residues"] = binding_site_summary
        df['on_target_pic50'] = np.nan
        df['on_target_ic50_nm'] = np.nan
        df['selectivity_ratio'] = np.nan
        df['cdk11_avoidance'] = np.nan
    else:
        df = calculate_all_binding_affinities(df, verbose=args.verbose)
    
    # Apply PAINS filtering
    if not args.skip_pains:
        df = apply_pains_filter(df)
    else:
        df['is_pains'] = False
        df['pains_score'] = 1.0
    
    # Keep only non-PAINS compounds for scoring
    df = df[df['is_pains'] != True].copy()
    if len(df) == 0:
        print("Error: All compounds flagged as PAINS. No compounds to score.")
        return 1

    # Calculate novelty
    if not args.skip_novelty:
        df = calculate_novelty_scores(df, similarity_cutoff=CONFIG["novelty_cutoff"])
    else:
        df['novelty_score'] = 0.5  # Neutral score
    
    # Normalize and calculate composite scores
    df = normalize_scores(df)
    df = calculate_composite_scores(df)
    
    # Generate outputs
    create_evaluation_dashboard(df, args.team_name, output_dir)
    generate_report(df, args.team_name, output_dir)
    
    # Print summary
    print("\nEvaluation Complete!")
    print(f"Total valid compounds: {len(df)}")
    print(f"Top compound score: {df['composite_score'].max():.3f}")
    print(f"Median score: {df['composite_score'].median():.3f}")
    
    if 'selectivity_ratio' in df.columns:
        good_selectivity = (df['selectivity_ratio'] > 10).sum()
        print(f"Compounds with >10-fold selectivity: {good_selectivity}")
    
    print(f"\nResults saved to {output_dir}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
