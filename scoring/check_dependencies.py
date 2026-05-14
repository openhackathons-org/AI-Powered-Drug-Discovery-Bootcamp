#!/usr/bin/env python3
"""Check and install all required dependencies"""

import subprocess
import sys
import importlib
import socket
from urllib.parse import urlparse

from endpoint_env import boltz2_endpoint_urls, load_openhackathon_env

load_openhackathon_env()

# Required packages and their pip names
REQUIRED_PACKAGES = {
    'pandas': 'pandas',
    'numpy': 'numpy',
    'matplotlib': 'matplotlib',
    'seaborn': 'seaborn',
    'rdkit': 'rdkit',
    'tqdm': 'tqdm',
    'boltz2_client': 'boltz2-python-client',
    'sqlalchemy': 'sqlalchemy',  # For database operations
}

def check_package(module_name):
    """Check if a package is installed"""
    try:
        importlib.import_module(module_name)
        return True
    except ImportError:
        return False

def install_package(pip_name):
    """Install a package using pip"""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", pip_name])
        return True
    except subprocess.CalledProcessError:
        return False

def check_port(host='localhost', port=8000):
    """Check if a port is open"""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(2)
    result = sock.connect_ex((host, port))
    sock.close()
    return result == 0

def check_boltz2_service():
    """Check if Boltz2 service is running"""
    import requests
    try:
        response = requests.get(f'{boltz2_endpoint_urls()[0]}/v1/health/ready', timeout=5)
        return response.status_code == 200
    except:
        return False

print("Checking dependencies for OpenHackathon evaluation script...")
print("=" * 60)

# Check Python packages
missing_packages = []
for module_name, pip_name in REQUIRED_PACKAGES.items():
    if check_package(module_name):
        print(f"✓ {module_name} is installed")
    else:
        print(f"✗ {module_name} is NOT installed")
        missing_packages.append((module_name, pip_name))

# Install missing packages
if missing_packages:
    print(f"\nInstalling {len(missing_packages)} missing packages...")
    for module_name, pip_name in missing_packages:
        print(f"Installing {pip_name}...")
        if install_package(pip_name):
            print(f"✓ Successfully installed {pip_name}")
        else:
            print(f"✗ Failed to install {pip_name}")
            print(f"  Try: pip install {pip_name}")

# Check Boltz2 service
print("\n" + "=" * 60)
print("Checking Boltz2 NIM service...")
boltz2_url = boltz2_endpoint_urls()[0]
parsed = urlparse(boltz2_url)
host = parsed.hostname or "localhost"
port = parsed.port or (443 if parsed.scheme == "https" else 80)

if check_port(host, port):
    print(f"✓ {boltz2_url} is listening")
    if check_package('requests'):
        if check_boltz2_service():
            print("✓ Boltz2 service is healthy")
        else:
            print(f"✗ {boltz2_url} is open but Boltz2 service is not responding")
            print("  The service might need to be restarted")
    else:
        print("  (Install 'requests' to check service health)")
else:
    print(f"✗ {boltz2_url} is not open - Boltz2 service is not running")
    print("\nTo start Boltz2 NIM locally:")
    print("   scripts/openhackathon_services.sh start --boltz2 1 --no-molmim")
    print("   source .openhackathon-nims.env")

print("\n" + "=" * 60)
print("Dependency check complete!")
