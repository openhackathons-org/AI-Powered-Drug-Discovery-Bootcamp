"""
Configuration module for CDK Oracle

All configurable parameters are centralized here for easy tweaking.
"""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional


def _parse_boltz2_endpoints() -> List[Dict[str, str]]:
    """Parse BOLTZ2_ENDPOINTS environment variable.
    
    Supports formats:
      - Single URL: "http://localhost:8000"
      - Multiple URLs: "http://gpu1:8000,http://gpu2:8000,http://gpu3:8000"
      - With API keys: "http://gpu1:8000|key1,http://gpu2:8000|key2"
    
    Returns:
        List of endpoint configs: [{"url": str, "api_key": str}, ...]
    """
    endpoints_str = os.environ.get("BOLTZ2_ENDPOINTS", "")
    default_api_key = os.environ.get("BOLTZ2_API_KEY", os.environ.get("NVIDIA_API_KEY", ""))
    
    if not endpoints_str:
        # Fall back to single endpoint
        return [{
            "url": os.environ.get("BOLTZ2_URL", "http://localhost:8000"),
            "api_key": default_api_key
        }]
    
    endpoints = []
    for entry in endpoints_str.split(","):
        entry = entry.strip()
        if not entry:
            continue
        
        if "|" in entry:
            # URL|api_key format
            parts = entry.split("|", 1)
            endpoints.append({
                "url": parts[0].strip(),
                "api_key": parts[1].strip() if len(parts) > 1 else default_api_key
            })
        else:
            # Just URL
            endpoints.append({
                "url": entry,
                "api_key": default_api_key
            })
    
    return endpoints if endpoints else [{
        "url": os.environ.get("BOLTZ2_URL", "http://localhost:8000"),
        "api_key": default_api_key
    }]


@dataclass
class CDKConfig:
    """Central configuration for CDK inhibitor design pipeline.
    
    Attributes can be overridden via environment variables or constructor args.
    """
    
    # === NIM Service URLs ===
    molmim_url: str = field(default_factory=lambda: os.environ.get("MOLMIM_URL", "http://localhost:8001"))
    boltz2_url: str = field(default_factory=lambda: os.environ.get("BOLTZ2_URL", "http://localhost:8000"))
    
    # === Multiple Boltz2 Endpoints (for parallel predictions) ===
    # Set via BOLTZ2_ENDPOINTS env var as comma-separated URLs
    # e.g., "http://gpu1:8000,http://gpu2:8000,http://gpu3:8000"
    boltz2_endpoints: List[Dict[str, str]] = field(default_factory=lambda: _parse_boltz2_endpoints())
    
    # === API Keys ===
    nvidia_api_key: str = field(default_factory=lambda: os.environ.get("NVIDIA_API_KEY", ""))
    boltz2_api_key: str = field(default_factory=lambda: os.environ.get("BOLTZ2_API_KEY", os.environ.get("NVIDIA_API_KEY", "")))
    
    # === Data Paths ===
    data_dir: Path = field(default_factory=lambda: Path(__file__).parent.parent / "data")
    output_dir: Path = field(default_factory=lambda: Path("./output"))
    
    # === Target Proteins ===
    on_target: str = "CDK4"
    anti_target: str = "CDK11"
    
    # === Binding Sites (residue indices) ===
    binding_sites: Dict[str, List[int]] = field(default_factory=lambda: {
        "CDK4": [35, 71, 96, 112, 158, 164, 196],
        "CDK11": [41, 87, 113, 128, 175, 182, 206]
    })
    
    # === MolMIM Optimization Parameters ===
    molmim_latent_dims: int = 512
    cma_popsize: int = 20
    cma_sigma: float = 1.0
    cma_iterations: int = 10
    
    # === Scoring Weights ===
    weights: Dict[str, float] = field(default_factory=lambda: {
        "binding_affinity": 0.25,   # CDK4 binding potency
        "selectivity": 0.20,        # CDK11/CDK4 ratio
        "cdk11_avoidance": 0.15,    # Penalty for CDK11 binding
        "qed": 0.15,                # Drug-likeness
        "sa": 0.10,                 # Synthetic accessibility
        "pains": 0.10,              # PAINS filter (penalty)
        "novelty": 0.05             # Structural novelty
    })
    
    # === Scoring Thresholds ===
    ic50_potent_threshold: float = 100.0      # nM - considered potent
    ic50_excellent_threshold: float = 10.0     # nM - considered excellent
    selectivity_good_threshold: float = 10.0   # 10x selectivity
    selectivity_excellent_threshold: float = 100.0  # 100x selectivity
    
    # === Physicochemical Thresholds (Lipinski) ===
    mw_max: float = 500.0
    logp_max: float = 5.0
    hbd_max: int = 5
    hba_max: int = 10
    tpsa_max: float = 140.0
    rotatable_bonds_max: int = 10
    
    # === Novelty Settings ===
    novelty_similarity_cutoff: float = 0.85  # Tanimoto similarity cutoff
    
    # === Output Settings ===
    top_n_compounds: int = 25
    save_all_generations: bool = False
    generate_plots: bool = True
    
    def __post_init__(self):
        """Ensure paths exist."""
        self.data_dir = Path(self.data_dir)
        self.output_dir = Path(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def get_fasta_path(self, protein: str) -> Path:
        """Get FASTA file path for a protein."""
        return self.data_dir / "fasta" / f"{protein}.fasta"
    
    def get_msa_path(self, protein: str) -> Path:
        """Get MSA file path for a protein."""
        return self.data_dir / "msa" / f"{protein}.a3m"
    
    def load_sequence(self, protein: str) -> str:
        """Load protein sequence from FASTA file."""
        fasta_path = self.get_fasta_path(protein)
        if not fasta_path.exists():
            raise FileNotFoundError(f"FASTA file not found: {fasta_path}")
        
        with open(fasta_path) as f:
            lines = f.readlines()
            return "".join([l.strip() for l in lines if not l.startswith(">")])
    
    def load_msa(self, protein: str) -> Optional[str]:
        """Load MSA content from A3M file."""
        msa_path = self.get_msa_path(protein)
        if not msa_path.exists():
            return None
        
        with open(msa_path) as f:
            return f.read()
    
    @classmethod
    def from_env(cls) -> "CDKConfig":
        """Create config from environment variables."""
        return cls()
    
    def to_dict(self) -> dict:
        """Export config as dictionary."""
        return {
            "molmim_url": self.molmim_url,
            "boltz2_url": self.boltz2_url,
            "on_target": self.on_target,
            "anti_target": self.anti_target,
            "cma_popsize": self.cma_popsize,
            "cma_sigma": self.cma_sigma,
            "cma_iterations": self.cma_iterations,
            "weights": self.weights,
        }

