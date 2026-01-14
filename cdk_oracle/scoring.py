"""
Scoring Module - Composite scoring for CDK inhibitor design

Combines binding affinity, selectivity, and drug-likeness metrics.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

from .config import CDKConfig
from .physicochemical import PhysicochemCalculator, calculate_novelty_score_chembl


@dataclass
class CompoundScore:
    """Container for compound scoring results."""
    smiles: str
    compound_id: str
    
    # Affinity scores
    cdk4_ic50_nm: float = None
    cdk11_ic50_nm: float = None
    cdk4_pic50: float = None
    cdk11_pic50: float = None
    selectivity_ratio: float = None
    
    # Physicochemical scores
    qed: float = None
    sa_score: float = None
    pains_alerts: int = None
    lipinski_violations: int = None
    
    # Novelty (ChEMBL-based)
    novelty_score: float = None
    max_chembl_similarity: float = None
    is_novel: bool = None
    
    # Component scores (0-1 normalized)
    binding_score: float = None
    selectivity_score: float = None
    avoidance_score: float = None
    qed_score: float = None
    sa_score_norm: float = None
    pains_score: float = None
    novelty_score_norm: float = None
    
    # Final composite score
    total_score: float = None
    rank: int = None


class CDKScorer:
    """Composite scorer for CDK inhibitor design."""
    
    def __init__(self, config: CDKConfig = None):
        self.config = config or CDKConfig()
        self.physchem = PhysicochemCalculator(config)
        self.reference_smiles = []  # For novelty calculation
    
    def set_reference_compounds(self, smiles_list: List[str]):
        """Set reference compounds for novelty calculation."""
        self.reference_smiles = smiles_list
    
    def _normalize_ic50_to_score(self, ic50_nm: float, best: float = 1.0, worst: float = 10000.0) -> float:
        """Convert IC50 to 0-1 score (lower IC50 = higher score)."""
        if ic50_nm is None or np.isnan(ic50_nm):
            return 0.0
        
        ic50_nm = max(best, min(worst, ic50_nm))
        # Log-scale normalization
        log_ic50 = np.log10(ic50_nm)
        log_best = np.log10(best)
        log_worst = np.log10(worst)
        
        return 1.0 - (log_ic50 - log_best) / (log_worst - log_best)
    
    def _normalize_selectivity_to_score(self, ratio: float) -> float:
        """Convert selectivity ratio to 0-1 score."""
        if ratio is None or np.isnan(ratio):
            return 0.0
        
        # Score based on thresholds
        if ratio >= self.config.selectivity_excellent_threshold:
            return 1.0
        elif ratio >= self.config.selectivity_good_threshold:
            # Linear interpolation between good and excellent
            return 0.7 + 0.3 * (ratio - self.config.selectivity_good_threshold) / \
                   (self.config.selectivity_excellent_threshold - self.config.selectivity_good_threshold)
        elif ratio >= 1.0:
            # Some selectivity
            return 0.3 + 0.4 * (ratio - 1.0) / (self.config.selectivity_good_threshold - 1.0)
        else:
            # Wrong direction (binds CDK11 more)
            return max(0.0, 0.3 * ratio)
    
    def _normalize_avoidance_to_score(self, cdk11_ic50: float) -> float:
        """Score for avoiding CDK11 binding (higher IC50 = higher score)."""
        if cdk11_ic50 is None or np.isnan(cdk11_ic50):
            return 0.5  # Unknown
        
        # Higher IC50 = better avoidance
        if cdk11_ic50 >= 10000:
            return 1.0
        elif cdk11_ic50 >= 1000:
            return 0.8 + 0.2 * (cdk11_ic50 - 1000) / 9000
        elif cdk11_ic50 >= 100:
            return 0.5 + 0.3 * (cdk11_ic50 - 100) / 900
        else:
            return 0.5 * cdk11_ic50 / 100
    
    def _normalize_sa_to_score(self, sa: float) -> float:
        """Convert SA score to 0-1 (lower SA = higher score)."""
        if sa is None or np.isnan(sa):
            return 0.5
        
        # SA score is 1-10, lower is better
        return max(0.0, 1.0 - (sa - 1.0) / 9.0)
    
    def _normalize_pains_to_score(self, pains: int) -> float:
        """Convert PAINS alerts to 0-1 score."""
        if pains is None:
            return 1.0
        return 1.0 if pains == 0 else 0.0
    
    def score_compound(
        self,
        smiles: str,
        compound_id: str = None,
        cdk4_ic50: float = None,
        cdk11_ic50: float = None,
        cdk4_pic50: float = None,
        cdk11_pic50: float = None,
    ) -> CompoundScore:
        """Score a single compound.
        
        Args:
            smiles: SMILES string
            compound_id: Optional compound identifier
            cdk4_ic50: CDK4 IC50 in nM (from Boltz2)
            cdk11_ic50: CDK11 IC50 in nM (from Boltz2)
            cdk4_pic50: CDK4 pIC50 (optional)
            cdk11_pic50: CDK11 pIC50 (optional)
            
        Returns:
            CompoundScore with all metrics
        """
        score = CompoundScore(
            smiles=smiles,
            compound_id=compound_id or f"COMP_{hash(smiles) % 10000:04d}",
            cdk4_ic50_nm=cdk4_ic50,
            cdk11_ic50_nm=cdk11_ic50,
            cdk4_pic50=cdk4_pic50,
            cdk11_pic50=cdk11_pic50,
        )
        
        # Calculate selectivity ratio
        if cdk4_ic50 and cdk11_ic50 and cdk4_ic50 > 0:
            score.selectivity_ratio = cdk11_ic50 / cdk4_ic50
        
        # Get physicochemical properties
        props = self.physchem.calculate_properties(smiles)
        score.qed = props.qed
        score.sa_score = props.sa_score
        score.pains_alerts = props.pains_alerts
        score.lipinski_violations = props.lipinski_violations
        
        # Calculate novelty using ChEMBL database (same as evaluate_submission.py)
        novelty, max_sim, is_novel = calculate_novelty_score_chembl(
            smiles, 
            cutoff=self.config.novelty_similarity_cutoff,
            additional_refs=self.reference_smiles if self.reference_smiles else None
        )
        score.novelty_score = novelty
        score.max_chembl_similarity = max_sim
        score.is_novel = is_novel
        
        # Normalize component scores
        score.binding_score = self._normalize_ic50_to_score(cdk4_ic50)
        score.selectivity_score = self._normalize_selectivity_to_score(score.selectivity_ratio)
        score.avoidance_score = self._normalize_avoidance_to_score(cdk11_ic50)
        score.qed_score = props.qed if props.qed else 0.0
        score.sa_score_norm = self._normalize_sa_to_score(props.sa_score)
        score.pains_score = self._normalize_pains_to_score(props.pains_alerts)
        score.novelty_score_norm = score.novelty_score
        
        # Calculate weighted composite score
        weights = self.config.weights
        score.total_score = (
            weights["binding_affinity"] * score.binding_score +
            weights["selectivity"] * score.selectivity_score +
            weights["cdk11_avoidance"] * score.avoidance_score +
            weights["qed"] * score.qed_score +
            weights["sa"] * score.sa_score_norm +
            weights["pains"] * score.pains_score +
            weights["novelty"] * score.novelty_score_norm
        )
        
        return score
    
    def score_batch(
        self,
        compounds_df: pd.DataFrame,
        smiles_col: str = "smiles",
        id_col: str = None,
        verbose: bool = True
    ) -> pd.DataFrame:
        """Score a batch of compounds.
        
        Args:
            compounds_df: DataFrame with SMILES and affinity predictions
            smiles_col: Column name for SMILES
            id_col: Column name for compound IDs (optional)
            verbose: Print progress
            
        Returns:
            DataFrame with all scores
        """
        results = []
        
        for i, row in compounds_df.iterrows():
            if verbose and (i + 1) % 10 == 0:
                print(f"Scoring: {i+1}/{len(compounds_df)}")
            
            smiles = row[smiles_col]
            compound_id = row[id_col] if id_col and id_col in row else None
            
            # Get affinity values from columns
            cdk4_ic50 = row.get(f"{self.config.on_target}_IC50_pred")
            cdk11_ic50 = row.get(f"{self.config.anti_target}_IC50_pred")
            cdk4_pic50 = row.get(f"{self.config.on_target}_pIC50_pred")
            cdk11_pic50 = row.get(f"{self.config.anti_target}_pIC50_pred")
            
            score = self.score_compound(
                smiles=smiles,
                compound_id=compound_id,
                cdk4_ic50=cdk4_ic50,
                cdk11_ic50=cdk11_ic50,
                cdk4_pic50=cdk4_pic50,
                cdk11_pic50=cdk11_pic50,
            )
            
            results.append(vars(score))
        
        results_df = pd.DataFrame(results)
        
        # Add ranks
        results_df = results_df.sort_values("total_score", ascending=False)
        results_df["rank"] = range(1, len(results_df) + 1)
        
        return results_df
    
    def get_top_compounds(self, scores_df: pd.DataFrame, n: int = None) -> pd.DataFrame:
        """Get top N compounds by total score.
        
        Args:
            scores_df: DataFrame with scores
            n: Number of top compounds (default: config.top_n_compounds)
            
        Returns:
            DataFrame with top compounds
        """
        n = n or self.config.top_n_compounds
        return scores_df.nsmallest(n, "rank")
    
    def get_scoring_summary(self, scores_df: pd.DataFrame) -> Dict[str, Any]:
        """Get summary statistics for scored compounds.
        
        Args:
            scores_df: DataFrame with scores
            
        Returns:
            Dict with summary statistics
        """
        return {
            "total_compounds": len(scores_df),
            "mean_total_score": scores_df["total_score"].mean(),
            "max_total_score": scores_df["total_score"].max(),
            "mean_cdk4_ic50": scores_df["cdk4_ic50_nm"].mean(),
            "mean_cdk11_ic50": scores_df["cdk11_ic50_nm"].mean(),
            "mean_selectivity": scores_df["selectivity_ratio"].mean(),
            "potent_cdk4_count": (scores_df["cdk4_ic50_nm"] < self.config.ic50_potent_threshold).sum(),
            "excellent_cdk4_count": (scores_df["cdk4_ic50_nm"] < self.config.ic50_excellent_threshold).sum(),
            "selective_count": (scores_df["selectivity_ratio"] > self.config.selectivity_good_threshold).sum(),
            "highly_selective_count": (scores_df["selectivity_ratio"] > self.config.selectivity_excellent_threshold).sum(),
            "pains_free_count": (scores_df["pains_alerts"] == 0).sum(),
            "drug_like_count": (scores_df["lipinski_violations"] <= 1).sum(),
        }

