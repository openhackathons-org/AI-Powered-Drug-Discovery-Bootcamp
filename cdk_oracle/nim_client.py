"""
NIM Client Module - Connections to MolMIM and Boltz2 services

Provides health checking and client wrappers for NVIDIA NIMs.
"""

import os
import json
import asyncio
import requests
import numpy as np
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

from .config import CDKConfig


@dataclass
class NIMStatus:
    """Status of a NIM service."""
    name: str
    url: str
    available: bool
    message: str


class NIMHealthChecker:
    """Check health status of NIM services."""
    
    def __init__(self, config: CDKConfig = None):
        self.config = config or CDKConfig()
    
    def check_molmim(self) -> NIMStatus:
        """Check MolMIM NIM health."""
        url = self.config.molmim_url
        try:
            # Try health endpoint
            response = requests.get(f"{url}/v1/health/ready", timeout=5)
            if response.status_code == 200:
                return NIMStatus("MolMIM", url, True, "Service is healthy")
            
            # Try root endpoint
            response = requests.get(url, timeout=5)
            if response.status_code in [200, 404]:
                return NIMStatus("MolMIM", url, True, "Service reachable")
            
            return NIMStatus("MolMIM", url, False, f"HTTP {response.status_code}")
        except requests.exceptions.ConnectionError:
            return NIMStatus("MolMIM", url, False, "Connection refused")
        except requests.exceptions.Timeout:
            return NIMStatus("MolMIM", url, False, "Connection timeout")
        except Exception as e:
            return NIMStatus("MolMIM", url, False, str(e))
    
    def check_boltz2(self) -> NIMStatus:
        """Check Boltz2 NIM health."""
        url = self.config.boltz2_url
        try:
            headers = {}
            if self.config.boltz2_api_key:
                headers["Authorization"] = f"Bearer {self.config.boltz2_api_key}"
            
            response = requests.get(f"{url}/v1/health/ready", headers=headers, timeout=5)
            if response.status_code == 200:
                return NIMStatus("Boltz2", url, True, "Service is healthy")
            
            return NIMStatus("Boltz2", url, False, f"HTTP {response.status_code}")
        except requests.exceptions.ConnectionError:
            return NIMStatus("Boltz2", url, False, "Connection refused")
        except requests.exceptions.Timeout:
            return NIMStatus("Boltz2", url, False, "Connection timeout")
        except Exception as e:
            return NIMStatus("Boltz2", url, False, str(e))
    
    def check_all(self) -> Dict[str, NIMStatus]:
        """Check all NIM services."""
        return {
            "molmim": self.check_molmim(),
            "boltz2": self.check_boltz2()
        }
    
    def print_status(self) -> Dict[str, bool]:
        """Print formatted status of all services."""
        print("=" * 60)
        print("NIM Service Health Check")
        print("=" * 60)
        
        status = self.check_all()
        results = {}
        
        for name, nim_status in status.items():
            icon = "✓" if nim_status.available else "✗"
            print(f"{icon} {nim_status.name}")
            print(f"    URL: {nim_status.url}")
            print(f"    Status: {nim_status.message}")
            results[name] = nim_status.available
        
        print("=" * 60)
        return results


class MolMIMClient:
    """Client for MolMIM NIM - molecule encoding/decoding."""
    
    def __init__(self, config: CDKConfig = None):
        self.config = config or CDKConfig()
        self.base_url = self.config.molmim_url
        self.latent_dims = self.config.molmim_latent_dims
    
    @property
    def num_latent_dims(self) -> int:
        return self.latent_dims
    
    def encode(self, smiles: List[str]) -> np.ndarray:
        """Encode SMILES strings into latent space.
        
        Args:
            smiles: List of SMILES strings
            
        Returns:
            np.ndarray of shape (1, len(smiles), latent_dims)
        """
        url = f"{self.base_url}/hidden"
        headers = {"accept": "application/json", "Content-Type": "application/json"}
        data = json.dumps({"sequences": smiles})
        
        response = requests.post(url, headers=headers, data=data, timeout=60)
        response.raise_for_status()
        
        embeddings = response.json()["hiddens"]
        embeddings_array = np.squeeze(np.array(embeddings))
        return np.expand_dims(embeddings_array, 0)
    
    def decode(self, latent_features: np.ndarray) -> List[str]:
        """Decode latent features into SMILES strings.
        
        Args:
            latent_features: np.ndarray of shape (1, n, latent_dims)
            
        Returns:
            List of generated SMILES strings
        """
        dims = list(latent_features.shape)
        if len(dims) == 2:
            latent_features = np.expand_dims(latent_features, 0)
        
        url = f"{self.base_url}/decode"
        headers = {"accept": "application/json", "Content-Type": "application/json"}
        
        latent_squeezed = np.squeeze(latent_features)
        latent_array = np.expand_dims(np.array(latent_squeezed), axis=1)
        
        payload = {
            "hiddens": latent_array.tolist(),
            "mask": [[True] for _ in range(latent_array.shape[0])]
        }
        
        response = requests.post(url, headers=headers, json=payload, timeout=60)
        response.raise_for_status()
        
        return response.json()["generated"]
    
    def sample(self, smiles: str, num_samples: int = 10) -> List[str]:
        """Sample similar molecules to a seed SMILES.
        
        Args:
            smiles: Seed SMILES string
            num_samples: Number of samples to generate
            
        Returns:
            List of generated SMILES strings
        """
        url = f"{self.base_url}/sampling"
        headers = {"accept": "application/json", "Content-Type": "application/json"}
        
        payload = {
            "smiles": smiles,
            "num_samples": num_samples,
            "temperature": 1.0
        }
        
        response = requests.post(url, headers=headers, json=payload, timeout=60)
        response.raise_for_status()
        
        return response.json().get("samples", [])


class Boltz2AffinityClient:
    """Client for Boltz2 NIM - binding affinity prediction.
    
    Supports multiple endpoints for parallel predictions.
    """
    
    def __init__(self, config: CDKConfig = None, endpoints: List[Dict[str, str]] = None):
        """Initialize Boltz2 client.
        
        Args:
            config: CDK configuration
            endpoints: List of endpoint configs, each with 'url' and optional 'api_key'
                       e.g., [{"url": "http://gpu1:8000"}, {"url": "http://gpu2:8000"}]
                       If None, uses single endpoint from config.
        """
        self.config = config or CDKConfig()
        
        # Set up endpoint pool
        if endpoints:
            self.endpoints = endpoints
        else:
            self.endpoints = [{
                "url": self.config.boltz2_url,
                "api_key": self.config.boltz2_api_key
            }]
        
        # For single-endpoint backward compatibility
        self.base_url = self.endpoints[0]["url"]
        self.api_key = self.endpoints[0].get("api_key", self.config.boltz2_api_key)
        
        # Endpoint tracking for load balancing
        self._endpoint_idx = 0
        self._endpoint_busy = {i: 0 for i in range(len(self.endpoints))}
        
        # Lazy import boltz2_client
        self._boltz2_available = False
        self._client_module = None
        self._load_client()
        
        # Concurrency control
        self._semaphore = None  # Initialized when needed
    
    def _load_client(self):
        """Load boltz2_client module."""
        try:
            from boltz2_client import Boltz2Client, Polymer, Ligand, PredictionRequest, PocketConstraint
            from boltz2_client.models import AlignmentFileRecord
            self._boltz2_available = True
            self._client_module = {
                "Boltz2Client": Boltz2Client,
                "Polymer": Polymer,
                "Ligand": Ligand,
                "PredictionRequest": PredictionRequest,
                "PocketConstraint": PocketConstraint,
                "AlignmentFileRecord": AlignmentFileRecord
            }
        except ImportError:
            self._boltz2_available = False
    
    @property
    def available(self) -> bool:
        return self._boltz2_available
    
    def _get_next_endpoint(self) -> Dict[str, str]:
        """Get next endpoint using round-robin load balancing."""
        endpoint = self.endpoints[self._endpoint_idx]
        self._endpoint_idx = (self._endpoint_idx + 1) % len(self.endpoints)
        return endpoint
    
    def _get_least_busy_endpoint(self) -> tuple:
        """Get endpoint with least active requests."""
        min_busy = min(self._endpoint_busy.values())
        for idx, busy in self._endpoint_busy.items():
            if busy == min_busy:
                return idx, self.endpoints[idx]
        return 0, self.endpoints[0]
    
    async def predict_affinity_async(
        self,
        smiles: str,
        protein: str,
        use_msa: bool = True,
        endpoint: Dict[str, str] = None,
        return_structures: bool = False
    ) -> Dict[str, Any]:
        """Predict binding affinity asynchronously.
        
        Args:
            smiles: Ligand SMILES string
            protein: Target protein name ("CDK4" or "CDK11")
            use_msa: Whether to use MSA for better predictions
            endpoint: Specific endpoint to use (optional, uses load balancing if None)
            return_structures: Whether to include structure data (mmCIF) in response
            
        Returns:
            Dict with ic50_nm, pic50, confidence, success, and optionally structures
        """
        if not self._boltz2_available:
            return {"error": "Boltz2 client not available", "success": False}
        
        # Select endpoint
        if endpoint is None:
            endpoint = self._get_next_endpoint()
        
        base_url = endpoint.get("url", self.base_url)
        api_key = endpoint.get("api_key", self.api_key)
        
        # Get protein data
        sequence = self.config.load_sequence(protein)
        binding_sites = self.config.binding_sites.get(protein, [])
        msa_content = self.config.load_msa(protein) if use_msa else None
        
        # Build request
        Boltz2Client = self._client_module["Boltz2Client"]
        Polymer = self._client_module["Polymer"]
        Ligand = self._client_module["Ligand"]
        PredictionRequest = self._client_module["PredictionRequest"]
        PocketConstraint = self._client_module["PocketConstraint"]
        AlignmentFileRecord = self._client_module["AlignmentFileRecord"]
        
        client = Boltz2Client(base_url=base_url, api_key=api_key)
        
        try:
            polymer_kwargs = {"id": "A", "molecule_type": "protein", "sequence": sequence}
            if msa_content:
                polymer_kwargs["msa"] = {
                    "Uniref30_2302": {
                        "a3m": AlignmentFileRecord(alignment=msa_content, format="a3m")
                    }
                }
            polymer = Polymer(**polymer_kwargs)
            
            ligand = Ligand(id="B", smiles=smiles, predict_affinity=True)
            
            pocket = PocketConstraint(
                ligand_id="B",
                polymer_id="A",
                residue_ids=binding_sites,
                binder="B"
            )
            
            request = PredictionRequest(
                polymers=[polymer],
                ligands=[ligand],
                pocket_constraints=[pocket]
            )
            
            response = await client.predict(request)
            
            # Parse response
            if hasattr(response, "affinities") and response.affinities and "B" in response.affinities:
                affinity = response.affinities["B"]
                pic50_list = getattr(affinity, "affinity_pic50", None)
                pic50 = pic50_list[0] if pic50_list else None
                ic50_nm = (10 ** (-pic50)) * 1e9 if pic50 else None
                conf_list = getattr(affinity, "affinity_probability_binary", None)
                confidence = conf_list[0] if conf_list else None
                
                result = {
                    "ic50_nm": ic50_nm,
                    "pic50": pic50,
                    "confidence": confidence,
                    "endpoint": base_url,
                    "success": True
                }
                
                # Extract structure data if requested
                if return_structures:
                    structures = self._extract_structures(response, verbose=False)
                    result["structures"] = structures
                    result["_has_structures"] = len(structures) > 0
                
                return result
            else:
                return {"error": "No affinity data in response", "success": False}
                
        except Exception as e:
            return {"error": str(e), "endpoint": base_url, "success": False}
    
    def _extract_structures(self, response: Any, verbose: bool = False) -> List[Dict[str, str]]:
        """Extract structure data (mmCIF) from Boltz2 response.
        
        Handles multiple formats:
        - Direct mmCIF string in response
        - File paths to CIF files
        - Nested structure objects
        
        Args:
            response: Boltz2 prediction response
            verbose: Print debug info
            
        Returns:
            List of dicts with structure data: [{"mmcif": str, "idx": int}, ...]
        """
        from pathlib import Path
        structures = []
        
        # Debug: print response attributes
        if verbose:
            attrs = [attr for attr in dir(response) if not attr.startswith('_')]
            print(f"    [DEBUG] Response attributes: {attrs}")
        
        raw_structures = getattr(response, "structures", None)
        if not raw_structures:
            if verbose:
                print(f"    [DEBUG] No 'structures' attribute in response")
            return structures
        
        if verbose:
            print(f"    [DEBUG] Found {len(raw_structures)} raw structures")
        
        for idx, structure in enumerate(raw_structures):
            cif_data = None
            structure_path = None
            
            # Debug: print structure type and attributes
            if verbose and idx == 0:
                print(f"    [DEBUG] Structure type: {type(structure)}")
                if hasattr(structure, '__dict__'):
                    print(f"    [DEBUG] Structure attrs: {list(structure.__dict__.keys())}")
                if hasattr(structure, 'model_dump'):
                    sd = structure.model_dump()
                    print(f"    [DEBUG] Structure dict keys: {list(sd.keys())}")
                    # Show format and whether structure has content
                    print(f"    [DEBUG] format={sd.get('format')}, structure_len={len(sd.get('structure', '') or '')}")
            
            # Try to get mmCIF data directly
            # boltz2_client.models.StructureData has: format, structure, name, source
            if hasattr(structure, "structure") and structure.structure:
                cif_data = structure.structure  # This is where boltz2_client stores CIF data
            elif hasattr(structure, "mmcif") and structure.mmcif:
                cif_data = structure.mmcif
            elif hasattr(structure, "mmCIF") and structure.mmCIF:
                cif_data = structure.mmCIF
            
            # Try to get file path
            if hasattr(structure, "path") and structure.path:
                structure_path = structure.path
            elif hasattr(structure, "file_path") and structure.file_path:
                structure_path = structure.file_path
            elif hasattr(structure, "source") and structure.source:
                # boltz2_client might use 'source' as file path
                structure_path = structure.source
            
            # Try model_dump() for Pydantic models
            if hasattr(structure, "model_dump"):
                structure_dict = structure.model_dump()
                cif_data = cif_data or structure_dict.get("structure") or structure_dict.get("mmcif") or structure_dict.get("mmCIF") or structure_dict.get("cif")
                structure_path = structure_path or structure_dict.get("path") or structure_dict.get("file_path") or structure_dict.get("source")
            elif isinstance(structure, dict):
                cif_data = cif_data or structure.get("structure") or structure.get("mmcif") or structure.get("mmCIF") or structure.get("cif")
                structure_path = structure_path or structure.get("path") or structure.get("file_path") or structure.get("source")
            
            # Also try direct string (some APIs return CIF as string directly)
            if not cif_data and isinstance(structure, str):
                if structure.startswith("data_"):
                    cif_data = structure
                elif Path(structure).exists():
                    structure_path = structure
            
            # If we have a file path, read the content
            if not cif_data and structure_path:
                try:
                    path_obj = Path(structure_path)
                    if path_obj.exists():
                        cif_data = path_obj.read_text()
                        if verbose:
                            print(f"    [DEBUG] Read structure from file: {structure_path}")
                except Exception as e:
                    if verbose:
                        print(f"    [DEBUG] Failed to read structure file {structure_path}: {e}")
            
            if cif_data:
                structures.append({
                    "mmcif": cif_data,
                    "idx": idx
                })
                if verbose:
                    print(f"    [DEBUG] Extracted structure {idx} ({len(cif_data)} chars)")
            elif verbose:
                print(f"    [DEBUG] Structure {idx} has no mmCIF data (path={structure_path})")
        
        return structures
    
    def predict_affinity(
        self,
        smiles: str,
        protein: str,
        use_msa: bool = True,
        return_structures: bool = False
    ) -> Dict[str, Any]:
        """Synchronous wrapper for predict_affinity_async."""
        try:
            import nest_asyncio
            nest_asyncio.apply()
        except ImportError:
            pass
        
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(
            self.predict_affinity_async(smiles, protein, use_msa, return_structures=return_structures)
        )
    
    def predict_batch(
        self,
        smiles_list: List[str],
        proteins: List[str] = None,
        use_msa: bool = True,
        verbose: bool = True,
        parallel: bool = True,
        max_concurrent: int = None,
        return_structures: bool = False
    ) -> List[Dict[str, Any]]:
        """Predict affinities for multiple compounds.
        
        Supports parallel predictions across multiple endpoints.
        
        Args:
            smiles_list: List of SMILES strings
            proteins: List of proteins (default: [on_target, anti_target])
            use_msa: Whether to use MSA
            verbose: Print progress
            parallel: Enable parallel predictions (uses multiple endpoints)
            max_concurrent: Max concurrent requests (default: num_endpoints * 2)
            return_structures: Whether to include structure data (mmCIF) in results
            
        Returns:
            List of result dictionaries
        """
        if proteins is None:
            proteins = [self.config.on_target, self.config.anti_target]
        
        # Use parallel if multiple endpoints available
        if parallel and len(self.endpoints) > 1:
            return self.predict_batch_parallel(
                smiles_list, proteins, use_msa, verbose, max_concurrent, return_structures
            )
        
        # Sequential fallback
        results = []
        for i, smiles in enumerate(smiles_list):
            if verbose:
                print(f"Predicting {i+1}/{len(smiles_list)}: {smiles[:30]}...")
            
            result = {"smiles": smiles}
            for protein in proteins:
                pred = self.predict_affinity(smiles, protein, use_msa, return_structures)
                result[f"{protein}_IC50_pred"] = pred.get("ic50_nm")
                result[f"{protein}_pIC50_pred"] = pred.get("pic50")
                result[f"{protein}_confidence"] = pred.get("confidence")
                result[f"{protein}_success"] = pred.get("success", False)
                if return_structures and pred.get("structures"):
                    result[f"{protein}_structures"] = pred["structures"]
            
            results.append(result)
        
        return results
    
    def predict_batch_parallel(
        self,
        smiles_list: List[str],
        proteins: List[str] = None,
        use_msa: bool = True,
        verbose: bool = True,
        max_concurrent: int = None,
        return_structures: bool = False
    ) -> List[Dict[str, Any]]:
        """Predict affinities in parallel using multiple endpoints.
        
        Distributes predictions across all available endpoints for faster throughput.
        
        Args:
            smiles_list: List of SMILES strings
            proteins: List of proteins
            use_msa: Whether to use MSA
            verbose: Print progress
            max_concurrent: Max concurrent requests
            return_structures: Whether to include structure data (mmCIF) in results
            
        Returns:
            List of result dictionaries (preserves order)
        """
        if proteins is None:
            proteins = [self.config.on_target, self.config.anti_target]
        
        # Determine concurrency
        if max_concurrent is None:
            max_concurrent = len(self.endpoints) * 2
        
        try:
            import nest_asyncio
            nest_asyncio.apply()
        except ImportError:
            pass
        
        import time
        start_time = time.time()
        
        async def run_parallel():
            semaphore = asyncio.Semaphore(max_concurrent)
            
            async def predict_one(idx: int, smiles: str) -> Dict[str, Any]:
                async with semaphore:
                    task_start = time.time()
                    result = {"smiles": smiles, "_idx": idx}
                    
                    for protein in proteins:
                        pred = await self.predict_affinity_async(
                            smiles, protein, use_msa, return_structures=return_structures
                        )
                        result[f"{protein}_IC50_pred"] = pred.get("ic50_nm")
                        result[f"{protein}_pIC50_pred"] = pred.get("pic50")
                        result[f"{protein}_confidence"] = pred.get("confidence")
                        result[f"{protein}_success"] = pred.get("success", False)
                        if return_structures and pred.get("structures"):
                            result[f"{protein}_structures"] = pred["structures"]
                    
                    task_time = time.time() - task_start
                    if verbose:
                        elapsed = time.time() - start_time
                        print(f"  [{idx+1}/{len(smiles_list)}] {smiles[:30]}... done ({task_time:.1f}s, @{elapsed:.1f}s)")
                    
                    return result
            
            if verbose:
                print(f"Running parallel predictions with {len(self.endpoints)} endpoint(s)")
                print(f"  Max concurrent: {max_concurrent}")
                print(f"  Total compounds: {len(smiles_list)}")
                print(f"  (timing: task_time, @elapsed_total)")
            
            # Create all tasks
            tasks = [
                predict_one(i, smi) 
                for i, smi in enumerate(smiles_list)
            ]
            
            # Run with progress
            results = await asyncio.gather(*tasks, return_exceptions=True)
            return results
        
        loop = asyncio.get_event_loop()
        results = loop.run_until_complete(run_parallel())
        
        total_time = time.time() - start_time
        
        # Handle exceptions and sort by original index
        processed = []
        for r in results:
            if isinstance(r, Exception):
                processed.append({"error": str(r), "success": False})
            else:
                processed.append(r)
        
        # Sort by original index and remove _idx
        processed.sort(key=lambda x: x.get("_idx", 0))
        for r in processed:
            r.pop("_idx", None)
        
        if verbose and len(smiles_list) > 0:
            avg_per_compound = total_time / len(smiles_list)
            sequential_estimate = avg_per_compound * len(smiles_list) * max_concurrent
            speedup = sequential_estimate / total_time if total_time > 0 else 1
            print(f"\n  ⚡ Parallel execution complete:")
            print(f"     Total time: {total_time:.1f}s for {len(smiles_list)} compounds")
            print(f"     Avg per compound: {avg_per_compound:.1f}s")
            print(f"     Estimated speedup: ~{speedup:.1f}x vs sequential")
        
        return processed
    
    def check_endpoints_health(self, verbose: bool = True) -> Dict[int, bool]:
        """Check health of all configured endpoints.
        
        Args:
            verbose: Print status
            
        Returns:
            Dict mapping endpoint index to health status
        """
        status = {}
        
        if verbose:
            print(f"Checking {len(self.endpoints)} Boltz2 endpoint(s)...")
        
        for i, endpoint in enumerate(self.endpoints):
            url = endpoint.get("url")
            api_key = endpoint.get("api_key", self.api_key)
            
            try:
                headers = {}
                if api_key:
                    headers["Authorization"] = f"Bearer {api_key}"
                
                response = requests.get(f"{url}/v1/health/ready", headers=headers, timeout=5)
                healthy = response.status_code == 200
                status[i] = healthy
                
                if verbose:
                    icon = "✓" if healthy else "✗"
                    print(f"  {icon} Endpoint {i+1}: {url} - {'healthy' if healthy else 'unreachable'}")
            except Exception as e:
                status[i] = False
                if verbose:
                    print(f"  ✗ Endpoint {i+1}: {url} - error: {e}")
        
        return status

