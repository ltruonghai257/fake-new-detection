#!/usr/bin/env python3
"""
Download trained models and artifacts from Vast.ai instance to local machine.
Supports downloading checkpoints, logs, mlruns, and other training outputs.
"""

import subprocess
import sys
import os
import json
import argparse
from pathlib import Path
from typing import Optional, Tuple, List

# ANSI colors for terminal output
class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

def print_header(text: str):
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'='*60}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{text.center(60)}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{'='*60}{Colors.ENDC}\n")

def print_success(text: str):
    print(f"{Colors.OKGREEN}✓ {text}{Colors.ENDC}")

def print_error(text: str):
    print(f"{Colors.FAIL}✗ {text}{Colors.ENDC}")

def print_warning(text: str):
    print(f"{Colors.WARNING}⚠ {text}{Colors.ENDC}")

def print_info(text: str):
    print(f"{Colors.OKCYAN}ℹ {text}{Colors.ENDC}")

def run_command(cmd: str, check: bool = True, capture: bool = False) -> Tuple[bool, str]:
    """Run shell command and return (success, output)."""
    try:
        if capture:
            result = subprocess.run(
                cmd,
                shell=True,
                check=check,
                capture_output=True,
                text=True
            )
            return True, result.stdout.strip()
        else:
            subprocess.run(cmd, shell=True, check=check)
            return True, ""
    except subprocess.CalledProcessError as e:
        if capture:
            return False, e.stderr.strip()
        return False, str(e)

def load_connection_config() -> Optional[dict]:
    """Load connection details from config file."""
    config_path = Path(__file__).parent.parent / ".vastai_config.json"
    
    if config_path.exists():
        with open(config_path, 'r') as f:
            return json.load(f)
    
    return None

def get_connection_details() -> Tuple[str, str]:
    """Get SSH connection details from user."""
    print_header("Vast.ai Instance Connection Details")
    print_info("Find these details in your Vast.ai console:")
    print_info("  https://cloud.vast.ai/console/instances/")
    print()
    
    ip = input("Enter instance IP address: ").strip()
    port = input("Enter SSH port: ").strip()
    
    if not ip or not port:
        print_error("IP and port are required")
        sys.exit(1)
    
    return ip, port

def check_ssh_key() -> Optional[str]:
    """Check if SSH key exists, return path if found."""
    ssh_dir = Path.home() / ".ssh"
    key_names = ["id_ed25519", "id_rsa", "id_ecdsa"]
    
    for key_name in key_names:
        key_path = ssh_dir / key_name
        pub_key_path = ssh_dir / f"{key_name}.pub"
        
        if key_path.exists() and pub_key_path.exists():
            print_success(f"Found SSH key: {key_path}")
            return str(key_path)
    
    return None

def list_remote_files(ip: str, port: str, key_path: str, remote_path: str) -> List[str]:
    """List files in remote directory."""
    cmd = f'ssh -p {port} -i {key_path} -o StrictHostKeyChecking=no root@{ip} "ls -lh {remote_path} 2>/dev/null || echo \'Directory not found\'"'
    success, output = run_command(cmd, check=False, capture=True)
    
    if success:
        print_info(f"Remote files in {remote_path}:")
        print(output)
        return output.split('\n')
    else:
        print_error(f"Failed to list remote files: {output}")
        return []

def download_directory(
    ip: str, 
    port: str, 
    key_path: str, 
    remote_path: str, 
    local_path: str,
    exclude_patterns: List[str] = None
) -> bool:
    """Download directory from Vast.ai instance using rsync."""
    print_info(f"Downloading {remote_path} to {local_path}...")
    print_warning("This may take several minutes depending on file size.")
    
    # Create local directory
    Path(local_path).mkdir(parents=True, exist_ok=True)
    
    # Build exclude arguments
    exclude_args = ""
    if exclude_patterns:
        exclude_args = " ".join([f"--exclude='{exc}'" for exc in exclude_patterns])
    
    cmd = f'rsync -avz -e "ssh -p {port} -i {key_path} -o StrictHostKeyChecking=no" {exclude_args} root@{ip}:{remote_path}/ {local_path}/'
    
    success, _ = run_command(cmd, check=False)
    
    if success:
        print_success(f"Successfully downloaded {remote_path}!")
        return True
    else:
        print_error(f"Download failed for {remote_path}. Trying with scp instead...")
        # Fallback to scp
        cmd = f'scp -P {port} -i {key_path} -r root@{ip}:{remote_path} {local_path}/'
        success, _ = run_command(cmd, check=False)
        
        if not success:
            print_error(f"Both rsync and scp failed for {remote_path}")
            return False
        else:
            print_success(f"Successfully downloaded {remote_path} via scp!")
            return True

def download_file(
    ip: str, 
    port: str, 
    key_path: str, 
    remote_file: str, 
    local_file: str
) -> bool:
    """Download single file from Vast.ai instance."""
    print_info(f"Downloading {remote_file} to {local_file}...")
    
    cmd = f'scp -P {port} -i {key_path} root@{ip}:{remote_file} {local_file}'
    
    success, _ = run_command(cmd, check=False)
    
    if success:
        print_success(f"Successfully downloaded {remote_file}!")
        return True
    else:
        print_error(f"Failed to download {remote_file}")
        return False

def download_all(
    ip: str, 
    port: str, 
    key_path: str, 
    remote_base: str = "/workspace/fake-new-detection",
    local_base: str = None
):
    """Download all training artifacts."""
    if local_base is None:
        local_base = str(Path(__file__).parent.parent)
    
    print_header("Download All Training Artifacts")
    
    # Define directories to download
    artifacts = [
        ("checkpoints/", "checkpoints/"),
        ("logs/", "logs/"),
        ("mlruns/", "mlruns/"),
        ("processed_data/hdf5/", "processed_data/hdf5/"),
    ]
    
    downloaded = []
    failed = []
    
    for remote_path, local_path in artifacts:
        full_remote = f"{remote_base}/{remote_path}"
        full_local = f"{local_base}/{local_path}"
        
        # Check if remote directory exists
        list_cmd = f'ssh -p {port} -i {key_path} -o StrictHostKeyChecking=no root@{ip} "test -d {full_remote} && echo \'exists\' || echo \'not found\'"'
        success, output = run_command(list_cmd, check=False, capture=True)
        
        if success and "exists" in output:
            if download_directory(ip, port, key_path, full_remote, full_local):
                downloaded.append(remote_path)
            else:
                failed.append(remote_path)
        else:
            print_warning(f"Remote directory not found: {full_remote}")
            failed.append(remote_path)
    
    # Summary
    print_header("Download Summary")
    if downloaded:
        print_success(f"Downloaded {len(downloaded)} directories:")
        for d in downloaded:
            print(f"  - {d}")
    
    if failed:
        print_warning(f"Failed to download {len(failed)} directories:")
        for f in failed:
            print(f"  - {f}")
    
    # Calculate total size
    size_cmd = f'du -sh {local_base}/checkpoints {local_base}/logs {local_base}/mlruns 2>/dev/null | awk \'{{sum+=$1}} END {{print sum}}\''
    # This is approximate, just show individual sizes
    for local_path in ["checkpoints", "logs", "mlruns"]:
        full_local = f"{local_base}/{local_path}"
        if Path(full_local).exists():
            success, size = run_command(f'du -sh {full_local}', check=False, capture=True)
            if success:
                print_info(f"{local_path}: {size}")

def main():
    parser = argparse.ArgumentParser(description="Download trained models from Vast.ai to local machine")
    parser.add_argument("--ip", type=str, help="Instance IP address")
    parser.add_argument("--port", type=str, help="SSH port")
    parser.add_argument("--remote-path", type=str, default="/workspace/fake-new-detection", help="Remote base path")
    parser.add_argument("--local-path", type=str, help="Local base path (default: project root)")
    parser.add_argument("--checkpoints", action="store_true", help="Download only checkpoints")
    parser.add_argument("--logs", action="store_true", help="Download only logs")
    parser.add_argument("--mlruns", action="store_true", help="Download only mlruns")
    parser.add_argument("--data", action="store_true", help="Download only processed data")
    parser.add_argument("--all", action="store_true", help="Download all artifacts (default)")
    parser.add_argument("--list", action="store_true", help="List available files without downloading")
    parser.add_argument("--file", type=str, help="Download specific file")
    
    args = parser.parse_args()
    
    print_header("Vast.ai Download for Fake News Detection")
    
    # Step 1: Check SSH key
    print_header("Step 1: SSH Key Setup")
    key_path = check_ssh_key()
    
    if not key_path:
        print_error("No SSH key found. Please generate one first using vastai/setup_vastai.py")
        sys.exit(1)
    
    # Step 2: Get connection details
    print_header("Step 2: Connection Details")
    
    # Try to load from config
    config = load_connection_config()
    
    if config and not args.ip and not args.port:
        print_info(f"Found saved config: {config['ip']}:{config['port']}")
        use_saved = input("Use saved config? (Y/n): ").strip().lower()
        if use_saved != 'n':
            ip, port = config['ip'], config['port']
        else:
            ip, port = get_connection_details()
    elif args.ip and args.port:
        ip, port = args.ip, args.port
    else:
        ip, port = get_connection_details()
    
    # Step 3: Set local path
    if args.local_path:
        local_base = args.local_path
    else:
        local_base = str(Path(__file__).parent.parent)
    
    remote_base = args.remote_path
    
    # Step 4: List files if requested
    if args.list:
        print_header("Available Files on Remote")
        list_remote_files(ip, port, key_path, remote_base)
        list_remote_files(ip, port, key_path, f"{remote_base}/checkpoints")
        list_remote_files(ip, port, key_path, f"{remote_base}/logs")
        list_remote_files(ip, port, key_path, f"{remote_base}/mlruns")
        sys.exit(0)
    
    # Step 5: Download specific file
    if args.file:
        print_header("Download Specific File")
        local_file = Path(args.file).name
        download_file(ip, port, key_path, args.file, local_file)
        sys.exit(0)
    
    # Step 6: Download based on flags
    if args.all or not any([args.checkpoints, args.logs, args.mlruns, args.data]):
        # Default: download all
        download_all(ip, port, key_path, remote_base, local_base)
    else:
        # Download specific artifacts
        artifacts = []
        if args.checkpoints:
            artifacts.append(("checkpoints/", "checkpoints/"))
        if args.logs:
            artifacts.append(("logs/", "logs/"))
        if args.mlruns:
            artifacts.append(("mlruns/", "mlruns/"))
        if args.data:
            artifacts.append(("processed_data/hdf5/", "processed_data/hdf5/"))
        
        downloaded = []
        failed = []
        
        for remote_path, local_path in artifacts:
            full_remote = f"{remote_base}/{remote_path}"
            full_local = f"{local_base}/{local_path}"
            
            if download_directory(ip, port, key_path, full_remote, full_local):
                downloaded.append(remote_path)
            else:
                failed.append(remote_path)
        
        # Summary
        print_header("Download Summary")
        if downloaded:
            print_success(f"Downloaded {len(downloaded)} directories:")
            for d in downloaded:
                print(f"  - {d}")
        
        if failed:
            print_warning(f"Failed to download {len(failed)} directories:")
            for f in failed:
                print(f"  - {f}")

if __name__ == "__main__":
    main()
