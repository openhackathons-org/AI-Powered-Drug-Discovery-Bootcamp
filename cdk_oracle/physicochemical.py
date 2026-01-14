"""
Physicochemical Properties Module

Calculates drug-likeness metrics and filters for compound evaluation.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass

from rdkit import Chem
from rdkit.Chem import Descriptors, QED, AllChem, FilterCatalog, rdMolDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold

from .config import CDKConfig


@dataclass
class MoleculeProperties:
    """Container for molecular properties."""
    smiles: str
    valid: bool
    mw: float = None
    logp: float = None
    hbd: int = None
    hba: int = None
    tpsa: float = None
    rotatable_bonds: int = None
    rings: int = None
    aromatic_rings: int = None
    qed: float = None
    sa_score: float = None
    lipinski_violations: int = None
    pains_alerts: int = None
    scaffold: str = None


class PhysicochemCalculator:
    """Calculate physicochemical properties and drug-likeness metrics."""
    
    def __init__(self, config: CDKConfig = None):
        self.config = config or CDKConfig()
        self._pains_catalog = None
        self._setup_pains_filter()
    
    def _setup_pains_filter(self):
        """Initialize PAINS filter catalog."""
        try:
            params = FilterCatalog.FilterCatalogParams()
            params.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.PAINS)
            self._pains_catalog = FilterCatalog.FilterCatalog(params)
        except Exception:
            self._pains_catalog = None
    
    def calculate_properties(self, smiles: str) -> MoleculeProperties:
        """Calculate all properties for a single molecule.
        
        Args:
            smiles: SMILES string
            
        Returns:
            MoleculeProperties dataclass
        """
        mol = Chem.MolFromSmiles(smiles)
        
        if mol is None:
            return MoleculeProperties(smiles=smiles, valid=False)
        
        props = MoleculeProperties(
            smiles=smiles,
            valid=True,
            mw=Descriptors.MolWt(mol),
            logp=Descriptors.MolLogP(mol),
            hbd=Descriptors.NumHDonors(mol),
            hba=Descriptors.NumHAcceptors(mol),
            tpsa=Descriptors.TPSA(mol),
            rotatable_bonds=Descriptors.NumRotatableBonds(mol),
            rings=rdMolDescriptors.CalcNumRings(mol),
            aromatic_rings=rdMolDescriptors.CalcNumAromaticRings(mol),
            qed=QED.qed(mol),
        )
        
        # Calculate SA score
        props.sa_score = self._calculate_sa_score(mol)
        
        # Calculate Lipinski violations
        props.lipinski_violations = self._count_lipinski_violations(props)
        
        # Calculate PAINS alerts
        props.pains_alerts = self._count_pains_alerts(mol)
        
        # Get Murcko scaffold
        try:
            scaffold = MurckoScaffold.GetScaffoldForMol(mol)
            props.scaffold = Chem.MolToSmiles(scaffold)
        except Exception:
            props.scaffold = None
        
        return props
    
    def _calculate_sa_score(self, mol) -> float:
        """Calculate synthetic accessibility score (1-10, lower is better)."""
        try:
            # Simplified SA score based on complexity
            from rdkit.Chem import rdMolDescriptors
            
            # Factors that increase complexity
            n_chiral = len(Chem.FindMolChiralCenters(mol, includeUnassigned=True))
            n_rings = rdMolDescriptors.CalcNumRings(mol)
            n_bridgehead = rdMolDescriptors.CalcNumBridgeheadAtoms(mol)
            n_spiro = rdMolDescriptors.CalcNumSpiroAtoms(mol)
            
            # Base score
            score = 1.0
            score += 0.5 * n_chiral
            score += 0.3 * max(0, n_rings - 2)
            score += 1.0 * n_bridgehead
            score += 0.8 * n_spiro
            
            # Normalize to 1-10 range
            return min(10.0, max(1.0, score))
        except Exception:
            return 5.0  # Default middle score
    
    def _count_lipinski_violations(self, props: MoleculeProperties) -> int:
        """Count Lipinski Rule of 5 violations."""
        violations = 0
        if props.mw and props.mw > self.config.mw_max:
            violations += 1
        if props.logp and props.logp > self.config.logp_max:
            violations += 1
        if props.hbd and props.hbd > self.config.hbd_max:
            violations += 1
        if props.hba and props.hba > self.config.hba_max:
            violations += 1
        return violations
    
    def _count_pains_alerts(self, mol) -> int:
        """Count PAINS filter alerts."""
        if self._pains_catalog is None:
            return 0
        
        try:
            entries = self._pains_catalog.GetMatches(mol)
            return len(entries)
        except Exception:
            return 0
    
    def calculate_batch(self, smiles_list: List[str], verbose: bool = False) -> pd.DataFrame:
        """Calculate properties for multiple molecules.
        
        Args:
            smiles_list: List of SMILES strings
            verbose: Print progress
            
        Returns:
            DataFrame with all properties
        """
        results = []
        for i, smiles in enumerate(smiles_list):
            if verbose and (i + 1) % 100 == 0:
                print(f"Calculating properties: {i+1}/{len(smiles_list)}")
            
            props = self.calculate_properties(smiles)
            results.append(vars(props))
        
        return pd.DataFrame(results)
    
    def passes_filters(self, smiles: str, strict: bool = False) -> Dict[str, Any]:
        """Check if molecule passes all filters.
        
        Args:
            smiles: SMILES string
            strict: If True, fail on any violation
            
        Returns:
            Dict with pass/fail for each filter and overall
        """
        props = self.calculate_properties(smiles)
        
        if not props.valid:
            return {"overall": False, "reason": "Invalid SMILES"}
        
        filters = {
            "valid_smiles": props.valid,
            "lipinski": props.lipinski_violations <= (0 if strict else 1),
            "pains_free": props.pains_alerts == 0,
            "mw_ok": props.mw <= self.config.mw_max,
            "logp_ok": props.logp <= self.config.logp_max,
            "qed_ok": props.qed >= 0.3,
            "sa_ok": props.sa_score <= 6.0,
        }
        
        filters["overall"] = all(filters.values())
        
        if not filters["overall"]:
            failed = [k for k, v in filters.items() if not v and k != "overall"]
            filters["reason"] = ", ".join(failed)
        else:
            filters["reason"] = None
        
        return filters
    
    def get_property_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Get summary statistics for a batch of molecules.
        
        Args:
            df: DataFrame with calculated properties
            
        Returns:
            Dict with summary statistics
        """
        valid_df = df[df["valid"] == True]
        
        return {
            "total_compounds": len(df),
            "valid_compounds": len(valid_df),
            "invalid_compounds": len(df) - len(valid_df),
            "mean_mw": valid_df["mw"].mean() if len(valid_df) > 0 else None,
            "mean_logp": valid_df["logp"].mean() if len(valid_df) > 0 else None,
            "mean_qed": valid_df["qed"].mean() if len(valid_df) > 0 else None,
            "mean_sa": valid_df["sa_score"].mean() if len(valid_df) > 0 else None,
            "lipinski_pass": (valid_df["lipinski_violations"] <= 1).sum() if len(valid_df) > 0 else 0,
            "pains_free": (valid_df["pains_alerts"] == 0).sum() if len(valid_df) > 0 else 0,
            "unique_scaffolds": valid_df["scaffold"].nunique() if len(valid_df) > 0 else 0,
        }


def calculate_tanimoto_similarity(smiles1: str, smiles2: str, radius: int = 2) -> float:
    """Calculate Tanimoto similarity between two molecules.
    
    Args:
        smiles1: First SMILES string
        smiles2: Second SMILES string
        radius: Morgan fingerprint radius
        
    Returns:
        Tanimoto similarity (0-1)
    """
    mol1 = Chem.MolFromSmiles(smiles1)
    mol2 = Chem.MolFromSmiles(smiles2)
    
    if mol1 is None or mol2 is None:
        return 0.0
    
    fp1 = AllChem.GetMorganFingerprintAsBitVect(mol1, radius, nBits=2048)
    fp2 = AllChem.GetMorganFingerprintAsBitVect(mol2, radius, nBits=2048)
    
    from rdkit.DataStructs import TanimotoSimilarity
    return TanimotoSimilarity(fp1, fp2)


def calculate_novelty_score(smiles: str, reference_smiles: List[str], cutoff: float = 0.85) -> float:
    """Calculate novelty score based on maximum similarity to reference set.
    
    DEPRECATED: Use calculate_novelty_score_chembl() for ChEMBL-based novelty.
    This function is kept for backward compatibility with simple reference sets.
    
    Args:
        smiles: Query SMILES
        reference_smiles: List of reference SMILES
        cutoff: Similarity cutoff for novelty
        
    Returns:
        Novelty score (0-1, higher is more novel)
    """
    if not reference_smiles:
        return 1.0
    
    max_similarity = 0.0
    for ref_smi in reference_smiles:
        sim = calculate_tanimoto_similarity(smiles, ref_smi)
        max_similarity = max(max_similarity, sim)
    
    # Convert similarity to novelty (1 - max_sim, but penalize more if above cutoff)
    if max_similarity > cutoff:
        return 0.0  # Not novel
    else:
        return 1.0 - (max_similarity / cutoff)


# Global cache for ChEMBL fingerprints
_CHEMBL_FP_CACHE = None


def load_chembl_fingerprints(chembl_path: str = None) -> Dict[str, Any]:
    """Load pre-computed ChEMBL fingerprints from pickle file.
    
    The fingerprints are cached globally for efficiency.
    
    Args:
        chembl_path: Path to chembl_data directory containing chembl_fingerprints.pkl
                     If None, uses default path from scoring/chembl_data/
    
    Returns:
        Dictionary mapping ChEMBL IDs to Morgan fingerprint bit vectors
    """
    global _CHEMBL_FP_CACHE
    
    if _CHEMBL_FP_CACHE is not None:
        return _CHEMBL_FP_CACHE
    
    import os
    import pickle
    from pathlib import Path
    
    # Try to find chembl_fingerprints.pkl
    search_paths = []
    
    if chembl_path:
        search_paths.append(Path(chembl_path) / "chembl_fingerprints.pkl")
    
    # Default locations relative to this file
    base_dir = Path(__file__).parent.parent
    search_paths.extend([
        base_dir / "scoring" / "chembl_data" / "chembl_fingerprints.pkl",
        base_dir / "data" / "chembl_data" / "chembl_fingerprints.pkl",
        Path("./chembl_data") / "chembl_fingerprints.pkl",
    ])
    
    fp_path = None
    for path in search_paths:
        if path.exists():
            fp_path = path
            break
    
    if fp_path is None:
        print("Warning: ChEMBL fingerprints not found. Novelty scoring will be limited.")
        print("To enable ChEMBL-based novelty, run:")
        print("  cd scoring && python create_chembl_database.py")
        _CHEMBL_FP_CACHE = {}
        return _CHEMBL_FP_CACHE
    
    print(f"Loading ChEMBL fingerprints from {fp_path}...")
    with open(fp_path, 'rb') as f:
        _CHEMBL_FP_CACHE = pickle.load(f)
    print(f"Loaded {len(_CHEMBL_FP_CACHE)} ChEMBL fingerprints")
    
    return _CHEMBL_FP_CACHE


def calculate_novelty_score_chembl(
    smiles: str, 
    cutoff: float = 0.85,
    chembl_path: str = None,
    additional_refs: List[str] = None
) -> Tuple[float, float, bool]:
    """Calculate novelty score by comparing to ChEMBL database.
    
    This is the same method used in evaluate_submission.py for scoring.
    
    Algorithm:
    1. Compute Morgan fingerprint (radius=2, 2048 bits) for query
    2. Compare to all ChEMBL fingerprints using Tanimoto similarity
    3. Find maximum similarity
    4. If max_sim >= cutoff: not novel (score = 0)
    5. If max_sim < cutoff: novel (score = 1 - max_sim/cutoff)
    
    Args:
        smiles: Query SMILES string
        cutoff: Tanimoto similarity threshold (default: 0.85)
        chembl_path: Path to ChEMBL fingerprint database
        additional_refs: Additional SMILES to check against (e.g., seed molecules)
    
    Returns:
        Tuple of (novelty_score, max_similarity, is_novel)
    """
    from rdkit.DataStructs import TanimotoSimilarity
    
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return 0.0, 1.0, False  # Invalid = not novel
    
    # Generate query fingerprint
    query_fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
    
    # Load ChEMBL fingerprints
    chembl_fps = load_chembl_fingerprints(chembl_path)
    
    # Find max similarity
    max_sim = 0.0
    
    # Check against ChEMBL
    for ref_fp in chembl_fps.values():
        sim = TanimotoSimilarity(query_fp, ref_fp)
        if sim > max_sim:
            max_sim = sim
            if max_sim >= cutoff:
                # Early exit - already not novel
                break
    
    # Check against additional references (e.g., seed molecules, generated compounds)
    if additional_refs and max_sim < cutoff:
        for ref_smi in additional_refs:
            ref_mol = Chem.MolFromSmiles(ref_smi)
            if ref_mol:
                ref_fp = AllChem.GetMorganFingerprintAsBitVect(ref_mol, radius=2, nBits=2048)
                sim = TanimotoSimilarity(query_fp, ref_fp)
                if sim > max_sim:
                    max_sim = sim
                    if max_sim >= cutoff:
                        break
    
    # Calculate novelty score (same formula as evaluate_submission.py)
    is_novel = max_sim < cutoff
    if is_novel:
        novelty_score = 1.0 - (max_sim / cutoff)
    else:
        novelty_score = 0.0
    
    return novelty_score, max_sim, is_novel


def batch_calculate_novelty_chembl(
    smiles_list: List[str],
    cutoff: float = 0.85,
    chembl_path: str = None,
    additional_refs: List[str] = None,
    verbose: bool = True
) -> List[Dict[str, Any]]:
    """Calculate novelty scores for multiple molecules efficiently.
    
    Args:
        smiles_list: List of SMILES strings
        cutoff: Tanimoto similarity threshold
        chembl_path: Path to ChEMBL fingerprint database
        additional_refs: Additional reference SMILES
        verbose: Print progress
    
    Returns:
        List of dicts with novelty_score, max_chembl_similarity, is_novel
    """
    from rdkit.DataStructs import TanimotoSimilarity
    
    # Pre-load ChEMBL fingerprints
    chembl_fps = load_chembl_fingerprints(chembl_path)
    chembl_fp_list = list(chembl_fps.values())
    
    # Pre-compute additional reference fingerprints
    additional_fps = []
    if additional_refs:
        for ref_smi in additional_refs:
            ref_mol = Chem.MolFromSmiles(ref_smi)
            if ref_mol:
                additional_fps.append(
                    AllChem.GetMorganFingerprintAsBitVect(ref_mol, radius=2, nBits=2048)
                )
    
    results = []
    
    iterator = smiles_list
    if verbose:
        from tqdm import tqdm
        iterator = tqdm(smiles_list, desc="Computing novelty")
    
    for smiles in iterator:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            results.append({
                "novelty_score": 0.0,
                "max_chembl_similarity": 1.0,
                "is_novel": False
            })
            continue
        
        query_fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
        
        # Find max similarity against ChEMBL
        max_sim = 0.0
        for ref_fp in chembl_fp_list:
            sim = TanimotoSimilarity(query_fp, ref_fp)
            if sim > max_sim:
                max_sim = sim
                if max_sim >= cutoff:
                    break
        
        # Check additional refs
        if max_sim < cutoff:
            for ref_fp in additional_fps:
                sim = TanimotoSimilarity(query_fp, ref_fp)
                if sim > max_sim:
                    max_sim = sim
                    if max_sim >= cutoff:
                        break
        
        is_novel = max_sim < cutoff
        novelty_score = (1.0 - max_sim / cutoff) if is_novel else 0.0
        
        results.append({
            "novelty_score": novelty_score,
            "max_chembl_similarity": max_sim,
            "is_novel": is_novel
        })
    
    return results

