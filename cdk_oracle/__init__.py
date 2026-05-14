# Copyright (c) 2026, NVIDIA CORPORATION. Licensed under the Apache License, Version 2.0 (the "License") you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0 Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.

"""
CDK Oracle - Modular Package for CDK4/CDK11 Inhibitor Design

This package provides modular components for:
- NIM service connections (MolMIM, Boltz2)
- Molecule generation and optimization
- Binding affinity prediction
- Physicochemical property calculation
- Composite scoring and visualization

Usage:
    from cdk_oracle import CDKDesignPipeline
    
    pipeline = CDKDesignPipeline()
    results = pipeline.run(seed_smiles=["CCO", "CCN"])
"""

from .config import CDKConfig
from .nim_client import NIMHealthChecker, MolMIMClient, Boltz2AffinityClient
from .physicochemical import PhysicochemCalculator
from .scoring import CDKScorer
from .visualization import CDKVisualizer
from .pipeline import CDKDesignPipeline

__version__ = "1.0.0"
__all__ = [
    "CDKConfig",
    "NIMHealthChecker", 
    "MolMIMClient",
    "Boltz2AffinityClient",
    "PhysicochemCalculator",
    "CDKScorer",
    "CDKVisualizer",
    "CDKDesignPipeline",
]

