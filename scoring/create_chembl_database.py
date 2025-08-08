#!/usr/bin/env python3
"""
Create a local ChEMBL database with compound fingerprints for novelty assessment
Downloads and processes ChEMBL chemical representations file
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


def create_chembl_database_from_file(chembl_file, output_dir="chembl_data", limit=None, batch_size=10000):
    """
    Create a local ChEMBL database from the downloaded file
    
    Args:
        chembl_file: Path to the ChEMBL chemical representations file
        output_dir: Directory to store the database files
        limit: Number of compounds to process (None for all compounds)
        batch_size: Number of compounds to process before committing to database
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    db_path = os.path.join(output_dir, "chembl_compounds.db")
    fp_path = os.path.join(output_dir, "chembl_fingerprints.pkl")
    
    # Check if database already exists
    if os.path.exists(db_path) and os.path.exists(fp_path):
        response = input(f"ChEMBL database already exists at {db_path}. Overwrite? (y/N): ")
        if response.lower() != 'y':
            print("Keeping existing database.")
            return db_path, fp_path
    
    print(f"Creating ChEMBL database from {chembl_file}")
    
    # First, count total lines for progress bar
    print("Counting compounds in file...")
    with open(chembl_file, 'r') as f:
        # Skip header
        next(f)
        total_lines = sum(1 for _ in f)
    
    if limit:
        total_lines = min(total_lines, limit)
    
    print(f"Processing {total_lines} compounds...")
    
    # Create database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Create table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS compounds (
            chembl_id TEXT PRIMARY KEY,
            smiles TEXT,
            canonical_smiles TEXT,
            fingerprint BLOB,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Create index for faster searches
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_smiles ON compounds(canonical_smiles)')
    
    fingerprints = {}
    batch_data = []
    total_processed = 0
    total_valid = 0
    
    # Process the file
    with open(chembl_file, 'r') as f:
        # Read header to understand the format
        header = next(f).strip().split('\t')
        chembl_id_idx = header.index('chembl_id')
        smiles_idx = header.index('canonical_smiles')
        
        # Process compounds
        with tqdm(total=total_lines, desc="Processing compounds") as pbar:
            for line in f:
                if limit and total_processed >= limit:
                    break
                
                try:
                    parts = line.strip().split('\t')
                    chembl_id = parts[chembl_id_idx]
                    smiles = parts[smiles_idx]
                    
                    # Validate SMILES and generate fingerprint
                    mol = Chem.MolFromSmiles(smiles)
                    if mol:
                        # Canonical SMILES for consistency
                        canonical_smiles = Chem.MolToSmiles(mol, canonical=True)
                        
                        # Generate Morgan fingerprint (radius=2, 2048 bits)
                        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
                        fp_bytes = fp.ToBitString().encode()
                        
                        batch_data.append((
                            chembl_id, 
                            smiles, 
                            canonical_smiles,
                            fp_bytes
                        ))
                        fingerprints[chembl_id] = fp
                        total_valid += 1
                        
                        # Insert in batches
                        if len(batch_data) >= batch_size:
                            cursor.executemany(
                                "INSERT OR IGNORE INTO compounds (chembl_id, smiles, canonical_smiles, fingerprint) VALUES (?, ?, ?, ?)",
                                batch_data
                            )
                            conn.commit()
                            batch_data = []
                    
                    total_processed += 1
                    pbar.update(1)
                    
                except Exception as e:
                    total_processed += 1
                    pbar.update(1)
                    continue
    
    # Insert remaining data
    if batch_data:
        cursor.executemany(
            "INSERT OR IGNORE INTO compounds (chembl_id, smiles, canonical_smiles, fingerprint) VALUES (?, ?, ?, ?)",
            batch_data
        )
        conn.commit()
    
    # Add database statistics
    cursor.execute("SELECT COUNT(*) FROM compounds")
    db_count = cursor.fetchone()[0]
    
    conn.close()
    
    # Save fingerprints as pickle for faster loading
    print(f"\nSaving fingerprints to {fp_path}...")
    with open(fp_path, 'wb') as f:
        pickle.dump(fingerprints, f)
    
    # Create a summary file
    summary_path = os.path.join(output_dir, "database_summary.txt")
    with open(summary_path, 'w') as f:
        f.write(f"ChEMBL Database Creation Summary\n")
        f.write(f"================================\n")
        f.write(f"Created at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Source: ChEMBL 35 Chemical Representations\n")
        f.write(f"Total compounds processed: {total_processed}\n")
        f.write(f"Valid compounds with fingerprints: {total_valid}\n")
        f.write(f"Compounds in database: {db_count}\n")
        f.write(f"Database path: {db_path}\n")
        f.write(f"Fingerprints path: {fp_path}\n")
        f.write(f"\nFingerprint specifications:\n")
        f.write(f"- Type: Morgan (circular)\n")
        f.write(f"- Radius: 2\n")
        f.write(f"- Bits: 2048\n")
    
    print(f"\n=== Summary ===")
    print(f"Created ChEMBL database with {db_count} compounds")
    print(f"Total processed: {total_processed}")
    print(f"Valid compounds: {total_valid}")
    print(f"Database saved to: {db_path}")
    print(f"Fingerprints saved to: {fp_path}")
    print(f"Summary saved to: {summary_path}")
    
    return db_path, fp_path


def test_database(db_path, fp_path):
    """Test the created database by running some queries"""
    print("\n=== Testing Database ===")
    
    # Test SQLite database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Get some statistics
    cursor.execute("SELECT COUNT(*) FROM compounds")
    total = cursor.fetchone()[0]
    print(f"Total compounds: {total}")
    
    # Sample some compounds
    cursor.execute("SELECT chembl_id, canonical_smiles FROM compounds LIMIT 5")
    samples = cursor.fetchall()
    print("\nSample compounds:")
    for chembl_id, smiles in samples:
        print(f"  {chembl_id}: {smiles[:50]}...")
    
    conn.close()
    
    # Test fingerprint pickle
    print(f"\nTesting fingerprint file...")
    with open(fp_path, 'rb') as f:
        fingerprints = pickle.load(f)
    print(f"Loaded {len(fingerprints)} fingerprints")
    
    # Test similarity calculation
    if len(fingerprints) >= 2:
        print("\nTesting similarity calculation...")
        fps = list(fingerprints.values())
        sim = TanimotoSimilarity(fps[0], fps[1])
        print(f"Similarity between first two compounds: {sim:.3f}")


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
        print("All dependencies are installed!")
    
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create local ChEMBL database for novelty assessment")
    parser.add_argument("--output-dir", default="chembl_data", help="Output directory for database files")
    parser.add_argument("--limit", type=int, default=None, help="Number of compounds to process (default: all)")
    parser.add_argument("--batch-size", type=int, default=10000, help="Batch size for database inserts")
    parser.add_argument("--test", action="store_true", help="Test the database after creation")
    parser.add_argument("--skip-download", action="store_true", help="Skip download if file already exists")
    parser.add_argument("--check-deps", action="store_true", help="Check and install dependencies")
    
    args = parser.parse_args()
    
    # Check dependencies if requested
    if args.check_deps:
        if not install_dependencies():
            exit(1)
    
    # Download ChEMBL file
    if not args.skip_download:
        chembl_file = download_chembl_file(args.output_dir)
    else:
        chembl_file = os.path.join(args.output_dir, "chembl_35_chemreps.txt")
        if not os.path.exists(chembl_file):
            print(f"ChEMBL file not found at {chembl_file}. Please run without --skip-download")
            exit(1)
    
    # Create the database
    db_path, fp_path = create_chembl_database_from_file(
        chembl_file=chembl_file,
        output_dir=args.output_dir,
        limit=args.limit,
        batch_size=args.batch_size
    )
    
    # Test if requested
    if args.test:
        test_database(db_path, fp_path)