#!/usr/bin/env python3
"""
Parallelized ChEMBL database creation script
Creates a local ChEMBL database with compound fingerprints using multiprocessing
"""

import os
import gzip
import sqlite3
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.DataStructs import TanimotoSimilarity
from tqdm import tqdm
import pickle
import argparse
from datetime import datetime
import urllib.request
import shutil
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import time
from typing import List, Tuple, Optional
import queue
import threading

# Disable RDKit warnings for cleaner output
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

def download_chembl_file(output_dir="chembl_data"):
    """Download ChEMBL chemical representations file"""
    os.makedirs(output_dir, exist_ok=True)
    
    url = "https://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/latest/chembl_35_chemreps.txt.gz"
    gz_file = os.path.join(output_dir, "chembl_35_chemreps.txt.gz")
    txt_file = os.path.join(output_dir, "chembl_35_chemreps.txt")
    
    # Check if already downloaded and extracted
    if os.path.exists(txt_file):
        print(f"ChEMBL file already exists at {txt_file}")
        return txt_file
    
    # Download the file
    print(f"Downloading ChEMBL chemical representations from {url}")
    print("This may take a few minutes...")
    
    def download_progress(block_num, block_size, total_size):
        downloaded = block_num * block_size
        percent = min(downloaded * 100 / total_size, 100)
        print(f"Download progress: {percent:.1f}%", end='\r')
    
    try:
        urllib.request.urlretrieve(url, gz_file, reporthook=download_progress)
        print("\nDownload completed!")
        
        # Extract the gzip file
        print("Extracting file...")
        with gzip.open(gz_file, 'rb') as f_in:
            with open(txt_file, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        print("Extraction completed!")
        
        # Remove the gzip file to save space
        os.remove(gz_file)
        
        return txt_file
        
    except Exception as e:
        print(f"Error downloading file: {e}")
        raise


def process_compound_batch(compound_data: List[Tuple[str, str]]) -> List[Tuple[str, str, str, bytes, object]]:
    """
    Process a batch of compounds to generate fingerprints
    
    Args:
        compound_data: List of (chembl_id, smiles) tuples
        
    Returns:
        List of (chembl_id, smiles, canonical_smiles, fp_bytes, fp_object) tuples
    """
    results = []
    
    for chembl_id, smiles in compound_data:
        try:
            # Validate SMILES and generate fingerprint
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                # Canonical SMILES for consistency
                canonical_smiles = Chem.MolToSmiles(mol, canonical=True)
                
                # Generate Morgan fingerprint (radius=2, 2048 bits)
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
                fp_bytes = fp.ToBitString().encode()
                
                results.append((
                    chembl_id, 
                    smiles, 
                    canonical_smiles,
                    fp_bytes,
                    fp  # Keep the fingerprint object for pickle
                ))
        except Exception:
            # Skip invalid compounds
            continue
    
    return results


def read_file_in_chunks(filename: str, chunk_size: int = 10000, limit: Optional[int] = None) -> List[List[Tuple[str, str]]]:
    """
    Read ChEMBL file and split into chunks for parallel processing
    
    Args:
        filename: Path to ChEMBL file
        chunk_size: Number of compounds per chunk
        limit: Maximum number of compounds to process
        
    Returns:
        List of chunks, each containing (chembl_id, smiles) tuples
    """
    chunks = []
    current_chunk = []
    total_read = 0
    
    print("Reading and chunking ChEMBL file...")
    
    with open(filename, 'r') as f:
        # Read header to understand the format
        header = next(f).strip().split('\t')
        chembl_id_idx = header.index('chembl_id')
        smiles_idx = header.index('canonical_smiles')
        
        for line in f:
            if limit and total_read >= limit:
                break
                
            try:
                parts = line.strip().split('\t')
                chembl_id = parts[chembl_id_idx]
                smiles = parts[smiles_idx]
                
                current_chunk.append((chembl_id, smiles))
                total_read += 1
                
                # Create new chunk when current one is full
                if len(current_chunk) >= chunk_size:
                    chunks.append(current_chunk)
                    current_chunk = []
                    
            except (IndexError, ValueError):
                # Skip malformed lines
                continue
    
    # Add remaining compounds
    if current_chunk:
        chunks.append(current_chunk)
    
    print(f"Created {len(chunks)} chunks with {total_read} total compounds")
    return chunks


class DatabaseWriter:
    """Thread-safe database writer for parallel processing results"""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.cursor = self.conn.cursor()
        self.lock = threading.Lock()
        
        # Create table
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS compounds (
                chembl_id TEXT PRIMARY KEY,
                smiles TEXT,
                canonical_smiles TEXT,
                fingerprint BLOB,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Create index for faster searches
        self.cursor.execute('CREATE INDEX IF NOT EXISTS idx_smiles ON compounds(canonical_smiles)')
        self.conn.commit()
    
    def write_batch(self, batch_results: List[Tuple[str, str, str, bytes, object]]) -> int:
        """Write a batch of results to the database"""
        if not batch_results:
            return 0
            
        db_data = [(chembl_id, smiles, canonical_smiles, fp_bytes) 
                   for chembl_id, smiles, canonical_smiles, fp_bytes, _ in batch_results]
        
        with self.lock:
            self.cursor.executemany(
                "INSERT OR IGNORE INTO compounds (chembl_id, smiles, canonical_smiles, fingerprint) VALUES (?, ?, ?, ?)",
                db_data
            )
            self.conn.commit()
        
        return len(db_data)
    
    def get_count(self) -> int:
        """Get total number of compounds in database"""
        with self.lock:
            self.cursor.execute("SELECT COUNT(*) FROM compounds")
            return self.cursor.fetchone()[0]
    
    def close(self):
        """Close database connection"""
        self.conn.close()


def create_chembl_database_parallel(chembl_file: str, 
                                   output_dir: str = "chembl_data", 
                                   limit: Optional[int] = None, 
                                   chunk_size: int = 10000,
                                   n_workers: Optional[int] = None):
    """
    Create a local ChEMBL database using parallel processing
    
    Args:
        chembl_file: Path to the ChEMBL chemical representations file
        output_dir: Directory to store the database files
        limit: Number of compounds to process (None for all compounds)
        chunk_size: Number of compounds per processing chunk
        n_workers: Number of parallel workers (None for auto-detect)
    """
    # Setup
    os.makedirs(output_dir, exist_ok=True)
    
    if n_workers is None:
        n_workers = min(mp.cpu_count(), 8)  # Cap at 8 to avoid memory issues
    
    print(f"Using {n_workers} parallel workers")
    
    db_path = os.path.join(output_dir, "chembl_compounds.db")
    fp_path = os.path.join(output_dir, "chembl_fingerprints.pkl")
    
    # Check if database already exists
    if os.path.exists(db_path) and os.path.exists(fp_path):
        response = input(f"ChEMBL database already exists at {db_path}. Overwrite? (y/N): ")
        if response.lower() != 'y':
            print("Keeping existing database.")
            return db_path, fp_path
    
    print(f"Creating ChEMBL database from {chembl_file}")
    
    # Read file in chunks
    chunks = read_file_in_chunks(chembl_file, chunk_size, limit)
    total_chunks = len(chunks)
    
    if total_chunks == 0:
        print("No compounds to process!")
        return None, None
    
    # Initialize database writer
    db_writer = DatabaseWriter(db_path)
    
    # Storage for all fingerprints
    all_fingerprints = {}
    
    # Process chunks in parallel
    total_processed = 0
    total_valid = 0
    start_time = time.time()
    
    print(f"Processing {total_chunks} chunks in parallel...")
    
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        # Submit all chunks for processing
        future_to_chunk = {
            executor.submit(process_compound_batch, chunk): i 
            for i, chunk in enumerate(chunks)
        }
        
        # Process results as they complete
        with tqdm(total=total_chunks, desc="Processing chunks") as pbar:
            for future in as_completed(future_to_chunk):
                chunk_idx = future_to_chunk[future]
                
                try:
                    # Get results from this chunk
                    chunk_results = future.result()
                    
                    # Write to database
                    written_count = db_writer.write_batch(chunk_results)
                    
                    # Store fingerprints
                    for chembl_id, _, _, _, fp_obj in chunk_results:
                        all_fingerprints[chembl_id] = fp_obj
                    
                    # Update counters
                    chunk_size_actual = len(chunks[chunk_idx])
                    total_processed += chunk_size_actual
                    total_valid += len(chunk_results)
                    
                    # Update progress
                    pbar.set_postfix({
                        'valid': f"{total_valid}/{total_processed}",
                        'rate': f"{len(chunk_results)}/{chunk_size_actual}",
                        'db_size': db_writer.get_count()
                    })
                    pbar.update(1)
                    
                except Exception as e:
                    print(f"Error processing chunk {chunk_idx}: {e}")
                    pbar.update(1)
    
    # Get final database count
    db_count = db_writer.get_count()
    db_writer.close()
    
    processing_time = time.time() - start_time
    
    # Save fingerprints as pickle for faster loading
    print(f"\nSaving {len(all_fingerprints)} fingerprints to {fp_path}...")
    fp_start_time = time.time()
    with open(fp_path, 'wb') as f:
        pickle.dump(all_fingerprints, f, protocol=pickle.HIGHEST_PROTOCOL)
    fp_time = time.time() - fp_start_time
    
    # Create a summary file
    summary_path = os.path.join(output_dir, "database_summary.txt")
    with open(summary_path, 'w') as f:
        f.write(f"ChEMBL Database Creation Summary (Parallel)\n")
        f.write(f"===========================================\n")
        f.write(f"Created at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Source: ChEMBL 35 Chemical Representations\n")
        f.write(f"Processing mode: Parallel ({n_workers} workers)\n")
        f.write(f"Chunk size: {chunk_size}\n")
        f.write(f"Total compounds processed: {total_processed}\n")
        f.write(f"Valid compounds with fingerprints: {total_valid}\n")
        f.write(f"Compounds in database: {db_count}\n")
        f.write(f"Processing time: {processing_time:.2f} seconds\n")
        f.write(f"Fingerprint save time: {fp_time:.2f} seconds\n")
        f.write(f"Processing rate: {total_processed/processing_time:.0f} compounds/second\n")
        f.write(f"Database path: {db_path}\n")
        f.write(f"Fingerprints path: {fp_path}\n")
        f.write(f"\nFingerprint specifications:\n")
        f.write(f"- Type: Morgan (circular)\n")
        f.write(f"- Radius: 2\n")
        f.write(f"- Bits: 2048\n")
    
    print(f"\n{'='*60}")
    print(f"PARALLEL PROCESSING COMPLETE")
    print(f"{'='*60}")
    print(f"Total compounds processed: {total_processed:,}")
    print(f"Valid compounds: {total_valid:,}")
    print(f"Success rate: {total_valid/total_processed*100:.1f}%")
    print(f"Database compounds: {db_count:,}")
    print(f"Processing time: {processing_time:.2f} seconds")
    print(f"Processing rate: {total_processed/processing_time:.0f} compounds/second")
    print(f"Workers used: {n_workers}")
    print(f"Chunk size: {chunk_size}")
    print(f"Database: {db_path}")
    print(f"Fingerprints: {fp_path}")
    print(f"Summary: {summary_path}")
    
    return db_path, fp_path


def test_database(db_path: str, fp_path: str):
    """Test the created database by running some queries"""
    print(f"\n{'='*40}")
    print(f"TESTING DATABASE")
    print(f"{'='*40}")
    
    # Test SQLite database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Get some statistics
    cursor.execute("SELECT COUNT(*) FROM compounds")
    total = cursor.fetchone()[0]
    print(f"✓ Total compounds: {total:,}")
    
    # Sample some compounds
    cursor.execute("SELECT chembl_id, canonical_smiles FROM compounds LIMIT 5")
    samples = cursor.fetchall()
    print(f"\n✓ Sample compounds:")
    for chembl_id, smiles in samples:
        print(f"  {chembl_id}: {smiles[:50]}...")
    
    # Test query performance
    print(f"\n✓ Testing query performance...")
    start_time = time.time()
    cursor.execute("SELECT COUNT(*) FROM compounds WHERE canonical_smiles LIKE 'C%'")
    count = cursor.fetchone()[0]
    query_time = time.time() - start_time
    print(f"  Query returned {count:,} compounds in {query_time:.3f} seconds")
    
    conn.close()
    
    # Test fingerprint pickle
    print(f"\n✓ Testing fingerprint file...")
    start_time = time.time()
    with open(fp_path, 'rb') as f:
        fingerprints = pickle.load(f)
    load_time = time.time() - start_time
    print(f"  Loaded {len(fingerprints):,} fingerprints in {load_time:.3f} seconds")
    
    # Test similarity calculation
    if len(fingerprints) >= 2:
        print(f"\n✓ Testing similarity calculation...")
        fps = list(fingerprints.values())
        start_time = time.time()
        sim = TanimotoSimilarity(fps[0], fps[1])
        sim_time = time.time() - start_time
        print(f"  Similarity between first two compounds: {sim:.3f} (calculated in {sim_time:.6f} seconds)")
        
        # Test batch similarity
        if len(fps) >= 100:
            print(f"  Testing batch similarity (100 comparisons)...")
            start_time = time.time()
            similarities = []
            for i in range(100):
                sim = TanimotoSimilarity(fps[0], fps[i])
                similarities.append(sim)
            batch_time = time.time() - start_time
            avg_sim = sum(similarities) / len(similarities)
            print(f"  Average similarity: {avg_sim:.3f}, batch time: {batch_time:.3f} seconds")
    
    print(f"\n✅ Database tests completed successfully!")


def install_dependencies():
    """Check and install required dependencies"""
    dependencies = {
        'rdkit': 'rdkit-pypi',
        'tqdm': 'tqdm',
        'pandas': 'pandas'
    }
    
    print("Checking dependencies...")
    missing = []
    
    for module, package in dependencies.items():
        try:
            __import__(module)
        except ImportError:
            missing.append(package)
    
    if missing:
        print(f"Missing dependencies: {', '.join(missing)}")
        response = input("Install missing dependencies? (y/N): ")
        if response.lower() == 'y':
            import subprocess
            import sys
            
            for package in missing:
                print(f"Installing {package}...")
                subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print("Dependencies installed!")
        else:
            print("Please install dependencies manually:")
            print(f"pip install {' '.join(missing)}")
            return False
    else:
        print("✓ All dependencies are installed!")
    
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create local ChEMBL database for novelty assessment (Parallelized)")
    parser.add_argument("--output-dir", default="chembl_data", help="Output directory for database files")
    parser.add_argument("--limit", type=int, default=None, help="Number of compounds to process (default: all)")
    parser.add_argument("--chunk-size", type=int, default=10000, help="Chunk size for parallel processing")
    parser.add_argument("--workers", type=int, default=None, help="Number of parallel workers (default: auto)")
    parser.add_argument("--test", action="store_true", help="Test the database after creation")
    parser.add_argument("--skip-download", action="store_true", help="Skip download if file already exists")
    parser.add_argument("--check-deps", action="store_true", help="Check and install dependencies")
    parser.add_argument("--benchmark", action="store_true", help="Run performance benchmarks")
    
    args = parser.parse_args()
    
    # Check dependencies if requested
    if args.check_deps:
        if not install_dependencies():
            exit(1)
    
    # Show system info
    print(f"System Information:")
    print(f"  CPU cores: {mp.cpu_count()}")
    print(f"  Workers to use: {args.workers if args.workers else min(mp.cpu_count(), 8)}")
    print(f"  Chunk size: {args.chunk_size}")
    
    # Download ChEMBL file
    if not args.skip_download:
        chembl_file = download_chembl_file(args.output_dir)
    else:
        chembl_file = os.path.join(args.output_dir, "chembl_35_chemreps.txt")
        if not os.path.exists(chembl_file):
            print(f"ChEMBL file not found at {chembl_file}. Please run without --skip-download")
            exit(1)
    
    # Create the database
    db_path, fp_path = create_chembl_database_parallel(
        chembl_file=chembl_file,
        output_dir=args.output_dir,
        limit=args.limit,
        chunk_size=args.chunk_size,
        n_workers=args.workers
    )
    
    # Test if requested
    if args.test and db_path and fp_path:
        test_database(db_path, fp_path)
    
    # Benchmark if requested
    if args.benchmark and db_path and fp_path:
        print(f"\n{'='*40}")
        print(f"PERFORMANCE BENCHMARK")
        print(f"{'='*40}")
        
        # Time for loading fingerprints
        print("Benchmarking fingerprint loading...")
        start = time.time()
        with open(fp_path, 'rb') as f:
            fps = pickle.load(f)
        load_time = time.time() - start
        print(f"  Loaded {len(fps):,} fingerprints in {load_time:.3f} seconds")
        print(f"  Loading rate: {len(fps)/load_time:.0f} fingerprints/second")
        
        # Time for similarity calculations
        if len(fps) >= 1000:
            print("Benchmarking similarity calculations...")
            fp_list = list(fps.values())
            
            # Single similarity
            start = time.time()
            for i in range(1000):
                TanimotoSimilarity(fp_list[0], fp_list[i])
            single_time = time.time() - start
            print(f"  1000 similarities in {single_time:.3f} seconds")
            print(f"  Similarity rate: {1000/single_time:.0f} comparisons/second")
