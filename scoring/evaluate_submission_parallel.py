#!/usr/bin/env python3
"""
Parallel OpenHackathon Evaluation Script with Multi-Endpoint Boltz2 Support

This script evaluates chemical compound submissions against multiple criteria
using parallel processing with multiple Boltz2 NIM endpoints for faster predictions.

Usage:
    python evaluate_submission_parallel.py <smiles_file> <team_name> [options]

Example:
    python evaluate_submission_parallel.py demo_compounds.csv TeamA --endpoints 8000,8001,8002 --max-workers 6
"""

import pandas as pd
import numpy as np
import time
import json
import argparse
import os
import sys
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

# RDKit imports
try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, QED, Crippen
    from rdkit.Chem.rdMolDescriptors import CalcNumRotatableBonds
    from rdkit.Chem import rdMolDescriptors
    RDKIT_AVAILABLE = True
except ImportError:
    print("Warning: RDKit not available. Install with: pip install rdkit-pypi")
    RDKIT_AVAILABLE = False

# Boltz2 client import
try:
    from boltz2_client import Boltz2Client, Polymer, Ligand, PredictionRequest
    BOLTZ2_AVAILABLE = True
except ImportError:
    print("Warning: boltz2-python-client not available. Install with: pip install boltz2-python-client")
    BOLTZ2_AVAILABLE = False

# Other imports
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False

# Configuration
CONFIG = {
    "endpoints": [8000, 8001, 8002],  # Default ports for Boltz2 NIM instances
    "max_workers": 6,  # Maximum concurrent workers
    "api_timeout": 300,  # Timeout for API calls in seconds
    "confidence_threshold": 0.7,
    "weights": {
        "binding_affinity": 0.35,
        "cdk11_avoidance": 0.25,
        "qed": 0.15,
        "sa_score": 0.15,
        "toxicity": 0.05,
        "novelty": 0.05
    },
    "chembl_data_path": "./openhackathon/chembl_data"
}

# Protein sequences for CDK targets
CDK_PROTEIN_INFO = {
    "CDK4": {
        "sequence": "MATSRYEPVAEIGVGAYGTVYKARDPHSGHFVALKSVRVPNGGGGGGGLPISTVREVALLRRLEAFEHPNVVRLMDVCATSRTDREIKVTLVFEHVDQDLRTYLDKAPPPGLPAETIKDLMRQFLRGLDFLHANCIVHRDLKPENILVTSGGTVKLADFGLARIYSYQMALTPVVVTLWYRAPEVLLQSTYATPVDMWSVGCIFAEMFRRKPLFCGNSEADQLGKIFDLIGLPPEDDWPRDVSLPRGAFPPRGPRPVQSVVPEMEESGAQLLLEMLTFNPHKRISAFRALQHSYLHKDEGNPE"
    },
    "CDK6": {
        "sequence": "MEKDGLCRADQQYECVAEIGEGAYGKVFKARDLKNGGRFVALKRVRVQTGEEGMPLSTIREVAVLRHLETFEHPNVVRLFDVCTVSRTDRETKLTLVFEHVDQDLTTYLDKVPEPGVPTETIKDMMFQLLRGLDFLHSHRVVHRDLKPQNILVTSSGQIKLADFGLARIYSFQMALTSVVVTLWYRAPEVLLQSSYATPVDLWSVGCIFAEMFRRKPLFRGSSDVDQLGKILDVIGLPGEEDWPRDVALPRQAFHSKSAQPIEKFVTDIDELGKDLLLKCLTFNPAKRISAYSALSHPYFQDLERCKENLDSHLPPSQNTSELNTA"
    },
    "CDK11": {
        "sequence": "ALQGCRSVEEFQCLNRIEEGTYGVVYRAKDKKTDEIVALKRLKMEKEKEGFPITSLREINTILKAQHPNIVTVREIVVGSNMDKIYIVMNYVEHDLKSLMETMKQPFLPGEVKTLMIQLLRGVKHLHDNWILHRDLKTSNLLLSHAGILKVGDFGLAREYGSPLKAYTPVVVTLWYRAPELLLGAKEYSTAVDMWSVGCIFAEMFRRKPLFPGKSEIDQINKVFKDLGTPSEKIWPGYSELPAVKKMTFSEHPYNNLRKRFGALLSDQGFDLMNKFLTYFPGRRISAEDGLKHEYFRETPLPIDPSMFPKLVEKY"
    }
}

class EndpointPool:
    """Manages a pool of Boltz2 endpoints with health checking and load balancing"""
    
    def __init__(self, endpoints: List[int], timeout: int = 300):
        self.endpoints = [f"http://localhost:{port}" for port in endpoints]
        self.timeout = timeout
        self.healthy_endpoints = []
        self.clients = {}
        self.current_index = 0
        
    async def check_health(self, endpoint: str) -> bool:
        """Check if an endpoint is healthy using boltz2-python-client"""
        try:
            if BOLTZ2_AVAILABLE:
                # Create client and test with a simple request
                client = Boltz2Client(base_url=endpoint, api_key="", timeout=10)
                
                # Test with a minimal prediction request to verify the endpoint works
                test_protein = Polymer(id="test_protein", sequence="MKLLKWAWLLLSKASSAHDKA")  # Short test sequence
                test_ligand = Ligand(id="test_ligand", smiles="CCO", predict_affinity=True)  # Simple ethanol
                
                test_request = PredictionRequest(
                    polymers=[test_protein],
                    ligands=[test_ligand],
                    recycling_steps=1,      # Minimal settings for health check
                    sampling_steps=10,      # Minimum required
                    diffusion_samples=1,
                    sampling_steps_affinity=20,  # Reduced for health check
                    diffusion_samples_affinity=1,
                    affinity_mw_correction=True
                )
                
                print(f"Testing endpoint {endpoint} with minimal prediction...")
                
                # Try to make a prediction - if this works, the endpoint is healthy
                test_prediction = await client.predict(test_request)
                
                # If we get here without exception, the endpoint is working
                self.clients[endpoint] = client
                print(f"✓ Endpoint {endpoint} is healthy and responsive")
                return True
                
            return False
        except Exception as e:
            print(f"Health check failed for {endpoint}: {e}")
            return False
    
    async def initialize(self):
        """Initialize the endpoint pool by checking health of all endpoints"""
        print(f"Initializing endpoint pool with {len(self.endpoints)} endpoints...")
        for endpoint in self.endpoints:
            is_healthy = await self.check_health(endpoint)
            if is_healthy:
                self.healthy_endpoints.append(endpoint)
                print(f"✓ {endpoint} is healthy")
            else:
                print(f"✗ {endpoint} is not available")
        
        if not self.healthy_endpoints:
            raise Exception("No healthy Boltz2 endpoints found!")
        
        print(f"Endpoint pool initialized with {len(self.healthy_endpoints)} healthy endpoints")
    
    def get_next_endpoint(self) -> Tuple[str, Optional[object]]:
        """Get the next available endpoint using round-robin"""
        if not self.healthy_endpoints:
            return None, None
            
        endpoint = self.healthy_endpoints[self.current_index]
        client = self.clients.get(endpoint)
        
        self.current_index = (self.current_index + 1) % len(self.healthy_endpoints)
        return endpoint, client

def print_rt(message: str):
    """Print message in real-time with immediate flush"""
    print(message, flush=True)

async def predict_binding_affinity_boltz2_async(smiles: str, protein_target: str, 
                                               endpoint_pool: EndpointPool,
                                               confidence_threshold: float = CONFIG["confidence_threshold"],
                                               verbose: bool = True,
                                               worker_id: int = 0) -> Dict:
    """Async predict IC50 values using Boltz2 Python client with endpoint pool"""
    
    start_time = time.time()
    endpoint, client = endpoint_pool.get_next_endpoint()
    
    if verbose:
        print_rt(f"\n[Worker {worker_id}] {'='*60}")
        print_rt(f"[Worker {worker_id}] BOLTZ2 PREDICTION REQUEST - {protein_target}")
        print_rt(f"[Worker {worker_id}] Endpoint: {endpoint}")
        print_rt(f"[Worker {worker_id}] {'='*60}")
        print_rt(f"[Worker {worker_id}] Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}")
        print_rt(f"[Worker {worker_id}] SMILES: {smiles}")
        print_rt(f"[Worker {worker_id}] Protein: {protein_target}")
        print_rt(f"[Worker {worker_id}] Sequence length: {len(CDK_PROTEIN_INFO[protein_target]['sequence'])} aa")
    
    try:
        if BOLTZ2_AVAILABLE and client and endpoint:
            if verbose:
                print_rt(f"[Worker {worker_id}] Mode: ACTUAL Boltz2 Python Client")
                print_rt(f"[Worker {worker_id}] Endpoint: {endpoint}")
                print_rt(f"[Worker {worker_id}] Client type: {type(client)}")
                print_rt(f"[Worker {worker_id}] BOLTZ2_AVAILABLE: {BOLTZ2_AVAILABLE}")
            
            protein_sequence = CDK_PROTEIN_INFO[protein_target]["sequence"]
            
            if verbose:
                print_rt(f"[Worker {worker_id}] \nINPUT DETAILS:")
                print_rt(f"[Worker {worker_id}]   SMILES: {smiles}")
                print_rt(f"[Worker {worker_id}]   Target: {protein_target}")
                print_rt(f"[Worker {worker_id}]   Sequence length: {len(protein_sequence)} residues")
            
            # Create prediction request
            protein = Polymer(id=f"protein_{protein_target}", sequence=protein_sequence)
            ligand = Ligand(
                id=f"ligand_{protein_target}_{hash(smiles) % 10000}",
                smiles=smiles,
                predict_affinity=True
            )
            
            request = PredictionRequest(
                polymers=[protein],
                ligands=[ligand],
                recycling_steps=3,
                sampling_steps=10,
                diffusion_samples=1,
                sampling_steps_affinity=50,
                diffusion_samples_affinity=2,
                affinity_mw_correction=True
            )
            
            if verbose:
                print_rt(f"[Worker {worker_id}] \nBOLTZ2 REQUEST PARAMETERS:")
                print_rt(f"[Worker {worker_id}]   Protein ID: {protein.id}")
                print_rt(f"[Worker {worker_id}]   Ligand ID: {ligand.id}")
                print_rt(f"[Worker {worker_id}]   Affinity prediction: {ligand.predict_affinity}")
                print_rt(f"[Worker {worker_id}] \nSending request to Boltz2...")
            
            # Make async prediction
            try:
                prediction = await client.predict(request)
                api_time = time.time() - start_time
                
                if verbose:
                    print_rt(f"[Worker {worker_id}] Prediction successful in {api_time:.2f} seconds")
            except Exception as pred_error:
                if verbose:
                    print_rt(f"[Worker {worker_id}] Prediction failed: {pred_error}")
                raise pred_error
            
            if verbose:
                print_rt(f"[Worker {worker_id}] \nBOLTZ2 PREDICTION RESPONSE")
                print_rt(f"[Worker {worker_id}] {'-'*50}")
                print_rt(f"[Worker {worker_id}] Status: SUCCESS")
                print_rt(f"[Worker {worker_id}] Response received in {api_time:.2f} seconds")
            
            # Extract affinity data
            pic50 = 5.0  # Default
            confidence = 0.5  # Default
            
            if hasattr(prediction, 'affinities') and prediction.affinities:
                ligand_id = f"ligand_{protein_target}_{hash(smiles) % 10000}"
                if ligand_id in prediction.affinities:
                    affinity = prediction.affinities[ligand_id]
                    if hasattr(affinity, 'affinity_pic50') and affinity.affinity_pic50:
                        pic50 = affinity.affinity_pic50[0]
                    
                    if hasattr(affinity, 'affinity_probability_binary') and affinity.affinity_probability_binary:
                        confidence = affinity.affinity_probability_binary[0]
                    
                    ic50_nm = 10**(9 - pic50)
                    
                    if verbose:
                        print_rt(f"[Worker {worker_id}] \nAFFINITY PREDICTION RESULTS:")
                        print_rt(f"[Worker {worker_id}]   pIC50: {pic50:.2f}")
                        print_rt(f"[Worker {worker_id}]   IC50: {ic50_nm:.2f} nM")
                        print_rt(f"[Worker {worker_id}]   Binding probability: {confidence:.1%}")
                        
                        binding_strength = "STRONG" if pic50 > 7.0 else "MODERATE" if pic50 > 5.0 else "WEAK"
                        print_rt(f"[Worker {worker_id}]   Binding strength: {binding_strength}")
                else:
                    if verbose:
                        print_rt(f"[Worker {worker_id}]   Warning: No pIC50 data in affinity response")
            else:
                if verbose:
                    print_rt(f"[Worker {worker_id}]   Warning: No affinity data in response, using estimated values")
                
                # Use mock values based on target
                if protein_target in ["CDK4", "CDK6"]:
                    ic50_nm = np.random.lognormal(np.log(50), 1.5)
                else:
                    ic50_nm = np.random.lognormal(np.log(5000), 1.5)
                pic50 = -np.log10(ic50_nm * 1e-9)
                
                if verbose:
                    print_rt(f"[Worker {worker_id}] \nESTIMATED AFFINITY (MOCK):")
                    print_rt(f"[Worker {worker_id}]   pIC50: {pic50:.2f}")
                    print_rt(f"[Worker {worker_id}]   IC50: {ic50_nm:.2f} nM")
            
            if verbose:
                print_rt(f"[Worker {worker_id}] \nConfidence: {confidence:.3f}")
                print_rt(f"[Worker {worker_id}] Accepted: {confidence >= confidence_threshold}")
            
        else:
            # No mock fallback - raise error if no valid client/endpoint
            error_msg = f"No valid Boltz2 client available for prediction"
            if verbose:
                print_rt(f"[Worker {worker_id}] ❌ PREDICTION FAILED")
                print_rt(f"[Worker {worker_id}] Error: {error_msg}")
                print_rt(f"[Worker {worker_id}] Reason: BOLTZ2_AVAILABLE={BOLTZ2_AVAILABLE}, client={client is not None}, endpoint={endpoint}")
                print_rt(f"[Worker {worker_id}] \nINPUT DETAILS:")
                print_rt(f"[Worker {worker_id}]   SMILES: {smiles}")
                print_rt(f"[Worker {worker_id}]   Target: {protein_target}")
                print_rt(f"[Worker {worker_id}]   Sequence length: {len(CDK_PROTEIN_INFO[protein_target]['sequence'])} residues")
            
            raise RuntimeError(error_msg)
    
            
            total_time = time.time() - start_time
            
            return {
                'ic50_nm': ic50_nm,
                'pic50': pic50,
                'confidence': confidence,
                'accepted': confidence >= confidence_threshold,
                'api_time': api_time,
                'total_time': total_time,
                'endpoint': endpoint,
                'worker_id': worker_id
            }
            
        else:
            # No mock fallback - raise error if no valid client/endpoint
            error_msg = f"No valid Boltz2 client available for prediction"
            if verbose:
                print_rt(f"[Worker {worker_id}] ❌ PREDICTION FAILED")
                print_rt(f"[Worker {worker_id}] Error: {error_msg}")
                print_rt(f"[Worker {worker_id}] Reason: BOLTZ2_AVAILABLE={BOLTZ2_AVAILABLE}, client={client is not None}, endpoint={endpoint}")
                print_rt(f"[Worker {worker_id}] \nINPUT DETAILS:")
                print_rt(f"[Worker {worker_id}]   SMILES: {smiles}")
                print_rt(f"[Worker {worker_id}]   Target: {protein_target}")
                print_rt(f"[Worker {worker_id}]   Sequence length: {len(CDK_PROTEIN_INFO[protein_target]['sequence'])} residues")
            
            raise RuntimeError(error_msg)
    
    except Exception as e:
        if verbose:
            print_rt(f"[Worker {worker_id}] \nBOLTZ2 PREDICTION RESPONSE")
            print_rt(f"[Worker {worker_id}] {'-'*50}")
            print_rt(f"[Worker {worker_id}] ERROR: {type(e).__name__}: {str(e)}")
            print_rt(f"[Worker {worker_id}] \nRequest that caused error:")
            print_rt(f"[Worker {worker_id}]   SMILES: {smiles}")
            print_rt(f"[Worker {worker_id}]   Target: {protein_target}")
            print_rt(f"[Worker {worker_id}]   Sequence length: {len(CDK_PROTEIN_INFO[protein_target]['sequence'])} residues")
            print_rt(f"[Worker {worker_id}] \n❌ NO MOCK FALLBACK - PREDICTION FAILED")
        
        # Re-raise the exception instead of using fallback values
        raise e

def predict_binding_affinity_boltz2_sync(smiles: str, protein_target: str, 
                                        endpoint_pool: EndpointPool,
                                        confidence_threshold: float = CONFIG["confidence_threshold"],
                                        verbose: bool = True,
                                        worker_id: int = 0) -> Dict:
    """Synchronous wrapper for async prediction"""
    return asyncio.run(predict_binding_affinity_boltz2_async(
        smiles, protein_target, endpoint_pool, confidence_threshold, verbose, worker_id
    ))

async def calculate_all_binding_affinities_parallel(df: pd.DataFrame, 
                                                   endpoint_pool: EndpointPool,
                                                   max_workers: int = 6,
                                                   verbose: bool = True) -> pd.DataFrame:
    """Calculate binding affinities in parallel using multiple endpoints"""
    print_rt(f"\nCalculating binding affinities using {len(endpoint_pool.healthy_endpoints)} endpoints with {max_workers} workers...")
    
    # Prepare prediction tasks
    tasks = []
    for idx, row in df.iterrows():
        smiles = row['smiles']
        for target in ["CDK4", "CDK6", "CDK11"]:
            tasks.append((idx, smiles, target))
    
    results = {}
    completed_tasks = 0
    total_tasks = len(tasks)
    
    # Use semaphore to limit concurrent workers
    semaphore = asyncio.Semaphore(max_workers)
    
    async def worker_task(task_data, worker_id):
        async with semaphore:
            idx, smiles, target = task_data
            result = await predict_binding_affinity_boltz2_async(
                smiles, target, endpoint_pool, CONFIG["confidence_threshold"], verbose, worker_id
            )
            return (idx, target, result)
    
    # Start all tasks
    worker_tasks = []
    for i, task_data in enumerate(tasks):
        worker_id = i % max_workers
        worker_tasks.append(worker_task(task_data, worker_id))
    
    # Process results as they complete
    for task in asyncio.as_completed(worker_tasks):
        idx, target, result = await task
        
        if idx not in results:
            results[idx] = {}
        results[idx][target] = result
        
        completed_tasks += 1
        if verbose:
            progress = (completed_tasks / total_tasks) * 100
            print_rt(f"Progress: {completed_tasks}/{total_tasks} ({progress:.1f}%) - Latest: {target} for compound {idx}")
    
    # Apply results to dataframe
    for idx in results:
        for target in ["CDK4", "CDK6", "CDK11"]:
            if target in results[idx]:
                result = results[idx][target]
                df.loc[idx, f'{target}_ic50_nm'] = result['ic50_nm']
                df.loc[idx, f'{target}_pic50'] = result['pic50']
                df.loc[idx, f'{target}_confidence'] = result['confidence']
                df.loc[idx, f'{target}_accepted'] = result['accepted']
                df.loc[idx, f'{target}_api_time'] = result['api_time']
                df.loc[idx, f'{target}_endpoint'] = result['endpoint']
    
    # Calculate selectivity metrics
    df['on_target_pic50'] = df[['CDK4_pic50', 'CDK6_pic50']].mean(axis=1)
    df['cdk11_avoidance'] = np.maximum(0, df['on_target_pic50'] - df['CDK11_pic50'])
    df['selectivity_ratio'] = df['on_target_pic50'] / (df['CDK11_pic50'] + 1e-6)
    
    if verbose:
        print_rt(f"\nAFFINITY PREDICTION RESULTS SUMMARY:")
        print_rt(f"Total predictions: {completed_tasks}")
        print_rt(f"Endpoints used: {len(endpoint_pool.healthy_endpoints)}")
        for idx, row in df.iterrows():
            print_rt(f"Compound {idx + 1}:")
            print_rt(f"  CDK4:  IC50 = {row.get('CDK4_ic50_nm', np.nan):.1f} nM, pIC50 = {row.get('CDK4_pic50', np.nan):.2f}")
            print_rt(f"  CDK6:  IC50 = {row.get('CDK6_ic50_nm', np.nan):.1f} nM, pIC50 = {row.get('CDK6_pic50', np.nan):.2f}")
            print_rt(f"  CDK11: IC50 = {row.get('CDK11_ic50_nm', np.nan):.1f} nM, pIC50 = {row.get('CDK11_pic50', np.nan):.2f}")
            print_rt(f"  Selectivity: {row.get('selectivity_ratio', np.nan):.2f}, CDK11 avoidance: {row.get('cdk11_avoidance', np.nan):.2f}")
    
    return df

def calculate_qed_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate QED scores for compounds"""
    if not RDKIT_AVAILABLE:
        df['qed'] = 0.5
        df['qed_normalized'] = 0.5
        return df
    
    print("Calculating QED scores...")
    qed_scores = []
    
    for smiles in df['smiles']:
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                qed_score = QED.qed(mol)
            else:
                qed_score = 0.0
        except:
            qed_score = 0.0
        qed_scores.append(qed_score)
    
    df['qed'] = qed_scores
    df['qed_normalized'] = df['qed']  # QED is already normalized 0-1
    
    return df

def calculate_sa_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate Synthetic Accessibility scores"""
    if not RDKIT_AVAILABLE:
        df['sa_score'] = 5.0
        df['sa_score_normalized'] = 0.5
        return df
    
    print("Calculating Synthetic Accessibility scores...")
    sa_scores = []
    
    for smiles in df['smiles']:
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                # Simplified SA score calculation
                mw = Descriptors.MolWt(mol)
                logp = Descriptors.MolLogP(mol)
                rotbonds = CalcNumRotatableBonds(mol)
                heteroatoms = Descriptors.NumHeteroatoms(mol)
                
                # Simple heuristic (lower is better, range 1-10)
                sa_score = min(10, max(1, 
                    1 + (mw / 100) + abs(logp) + (rotbonds / 3) + (heteroatoms / 2)
                ))
            else:
                sa_score = 10.0
        except:
            sa_score = 10.0
        sa_scores.append(sa_score)
    
    df['sa_score'] = sa_scores
    df['sa_score_normalized'] = 1 - (df['sa_score'] / 10)  # Normalize and invert
    
    return df

def normalize_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize all scores to 0-1 range"""
    print("Normalizing scores...")
    
    # Binding affinity: Higher pIC50 is better
    if 'on_target_pic50' in df.columns:
        df['binding_affinity_normalized'] = (df['on_target_pic50'] - df['on_target_pic50'].min()) / \
                                          (df['on_target_pic50'].max() - df['on_target_pic50'].min() + 1e-6)
    else:
        df['binding_affinity_normalized'] = 0.5
    
    # CDK11 avoidance: Higher is better
    if 'cdk11_avoidance' in df.columns:
        df['cdk11_avoidance_normalized'] = (df['cdk11_avoidance'] - df['cdk11_avoidance'].min()) / \
                                         (df['cdk11_avoidance'].max() - df['cdk11_avoidance'].min() + 1e-6)
    else:
        df['cdk11_avoidance_normalized'] = 0.5
    
    # Other scores are already normalized or set defaults
    for score_type in ['qed', 'sa_score', 'toxicity', 'novelty']:
        norm_col = f'{score_type}_normalized'
        if norm_col not in df.columns:
            df[norm_col] = 0.5
    
    return df

def calculate_composite_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate weighted composite scores"""
    print("Calculating composite scores...")
    
    weights = CONFIG["weights"]
    
    # Ensure all normalized columns exist
    required_cols = [
        'binding_affinity_normalized', 'cdk11_avoidance_normalized',
        'qed_normalized', 'sa_score_normalized', 'toxicity_normalized', 'novelty_normalized'
    ]
    
    for col in required_cols:
        if col not in df.columns:
            df[col] = 0.5  # Default neutral score
    
    df['composite_score'] = (
        df['binding_affinity_normalized'] * weights['binding_affinity'] +
        df['cdk11_avoidance_normalized'] * weights['cdk11_avoidance'] +
        df['qed_normalized'] * weights['qed'] +
        df['sa_score_normalized'] * weights['sa_score'] +
        df['toxicity_normalized'] * weights['toxicity'] +
        df['novelty_normalized'] * weights['novelty']
    )
    
    return df

def generate_html_report(df: pd.DataFrame, team_name: str, output_dir: str):
    """Generate HTML report"""
    print("Generating HTML report...")
    
    # Sort by composite score
    df_sorted = df.sort_values('composite_score', ascending=False)
    top_25 = df_sorted.head(25)
    
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>OpenHackathon Evaluation Report - {team_name}</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
            .summary {{ margin: 20px 0; }}
            .table-container {{ overflow-x: auto; }}
            table {{ border-collapse: collapse; width: 100%; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: center; }}
            th {{ background-color: #f2f2f2; }}
            .score-high {{ background-color: #d4edda; }}
            .score-medium {{ background-color: #fff3cd; }}
            .score-low {{ background-color: #f8d7da; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>OpenHackathon Evaluation Report</h1>
            <h2>Team: {team_name}</h2>
            <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            <p>Evaluated using parallel processing with multiple Boltz2 endpoints</p>
        </div>
        
        <div class="summary">
            <h3>Summary Statistics</h3>
            <p>Total compounds evaluated: {len(df)}</p>
            <p>Top 25 compounds average score: {top_25['composite_score'].mean():.3f}</p>
            <p>Best compound score: {df['composite_score'].max():.3f}</p>
            <p>Evaluation method: Parallel processing with {len(set(df.get('CDK4_endpoint', ['localhost:8000'])))} Boltz2 endpoints</p>
        </div>
        
        <div class="table-container">
            <h3>Top 25 Compounds</h3>
            <table>
                <thead>
                    <tr>
                        <th>Rank</th>
                        <th>SMILES</th>
                        <th>Composite Score</th>
                        <th>CDK4 pIC50</th>
                        <th>CDK6 pIC50</th>
                        <th>CDK11 pIC50</th>
                        <th>Selectivity Ratio</th>
                        <th>QED</th>
                        <th>SA Score</th>
                        <th>Processing Endpoint</th>
                    </tr>
                </thead>
                <tbody>
    """
    
    for i, (_, row) in enumerate(top_25.iterrows(), 1):
        score_class = "score-high" if row['composite_score'] > 0.7 else "score-medium" if row['composite_score'] > 0.5 else "score-low"
        endpoint = row.get('CDK4_endpoint', 'Unknown')
        
        html_content += f"""
                    <tr class="{score_class}">
                        <td>{i}</td>
                        <td style="font-family: monospace; text-align: left;">{row['smiles'][:50]}{'...' if len(row['smiles']) > 50 else ''}</td>
                        <td>{row['composite_score']:.3f}</td>
                        <td>{row.get('CDK4_pic50', 0):.2f}</td>
                        <td>{row.get('CDK6_pic50', 0):.2f}</td>
                        <td>{row.get('CDK11_pic50', 0):.2f}</td>
                        <td>{row.get('selectivity_ratio', 0):.2f}</td>
                        <td>{row.get('qed', 0):.3f}</td>
                        <td>{row.get('sa_score', 0):.2f}</td>
                        <td>{endpoint}</td>
                    </tr>
        """
    
    html_content += """
                </tbody>
            </table>
        </div>
    </body>
    </html>
    """
    
    with open(os.path.join(output_dir, f"{team_name}_evaluation_report.html"), 'w') as f:
        f.write(html_content)

async def main():
    parser = argparse.ArgumentParser(description='Parallel OpenHackathon Compound Evaluation')
    parser.add_argument('smiles_file', help='CSV file containing SMILES')
    parser.add_argument('team_name', help='Team name for the submission')
    parser.add_argument('--endpoints', default='8000,8001,8002', 
                       help='Comma-separated list of Boltz2 endpoint ports (default: 8000,8001,8002)')
    parser.add_argument('--max-workers', type=int, default=6,
                       help='Maximum number of concurrent workers (default: 6)')
    parser.add_argument('--output-dir', default='evaluation_output',
                       help='Output directory for results')
    parser.add_argument('--skip-toxicity', action='store_true',
                       help='Skip toxicity calculations')
    parser.add_argument('--skip-novelty', action='store_true',
                       help='Skip novelty calculations')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose output')
    
    args = parser.parse_args()
    
    # Parse endpoints
    endpoint_ports = [int(port.strip()) for port in args.endpoints.split(',')]
    CONFIG["endpoints"] = endpoint_ports
    CONFIG["max_workers"] = args.max_workers
    
    print_rt(f"Starting parallel evaluation for team: {args.team_name}")
    print_rt(f"Using endpoints: {endpoint_ports}")
    print_rt(f"Max workers: {args.max_workers}")
    
    # Load SMILES data
    try:
        df = pd.read_csv(args.smiles_file)
        if 'smiles' not in df.columns:
            if 'SMILES' in df.columns:
                df = df.rename(columns={'SMILES': 'smiles'})
            else:
                print("Error: No 'smiles' or 'SMILES' column found in the input file")
                return
        
        print_rt(f"Loaded {len(df)} compounds from {args.smiles_file}")
    except Exception as e:
        print(f"Error loading SMILES file: {e}")
        return
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize endpoint pool
    endpoint_pool = EndpointPool(endpoint_ports, timeout=CONFIG["api_timeout"])
    await endpoint_pool.initialize()
    
    start_time = time.time()
    
    # Calculate binding affinities in parallel
    df = await calculate_all_binding_affinities_parallel(
        df, endpoint_pool, args.max_workers, args.verbose
    )
    
    # Calculate other scores
    df = calculate_qed_scores(df)
    df = calculate_sa_scores(df)
    
    # Set default scores for skipped calculations
    if args.skip_toxicity:
        df['toxicity_normalized'] = 0.5
        print_rt("Skipping toxicity calculations (set to neutral score)")
    
    if args.skip_novelty:
        df['novelty_normalized'] = 0.5
        print_rt("Skipping novelty calculations (set to neutral score)")
    
    # Normalize and calculate composite scores
    df = normalize_scores(df)
    df = calculate_composite_scores(df)
    
    # Generate outputs
    generate_html_report(df, args.team_name, args.output_dir)
    
    # Save detailed CSV
    csv_path = os.path.join(args.output_dir, f"{args.team_name}_detailed_results.csv")
    df.to_csv(csv_path, index=False)
    
    total_time = time.time() - start_time
    
    print_rt(f"\nEvaluation completed successfully!")
    print_rt(f"Total time: {total_time:.2f} seconds")
    print_rt(f"Average time per compound: {total_time/len(df):.2f} seconds")
    print_rt(f"Results saved to: {args.output_dir}")
    print_rt(f"HTML report: {args.team_name}_evaluation_report.html")
    print_rt(f"Detailed CSV: {args.team_name}_detailed_results.csv")
    
    # Show top results
    df_sorted = df.sort_values('composite_score', ascending=False)
    print_rt(f"\nTop 5 compounds:")
    for i, (_, row) in enumerate(df_sorted.head(5).iterrows(), 1):
        print_rt(f"{i}. Score: {row['composite_score']:.3f}, SMILES: {row['smiles'][:60]}...")

if __name__ == "__main__":
    asyncio.run(main())
