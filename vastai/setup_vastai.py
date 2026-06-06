#!/usr/bin/env python3
"""
One-click Vast.ai setup script for Fake News Detection project.
Handles SSH key generation, connection, and environment setup.
"""

import subprocess
import sys
import os
import json
import argparse
from pathlib import Path
from typing import Optional, Tuple

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

def generate_ssh_key(email: str = "vastai@local") -> str:
    """Generate new SSH key pair."""
    print_info("Generating new SSH key pair...")
    
    ssh_dir = Path.home() / ".ssh"
    ssh_dir.mkdir(exist_ok=True, mode=0o700)
    
    key_path = ssh_dir / "id_ed25519"
    
    # Check if key already exists
    if key_path.exists():
        response = input(f"Key {key_path} already exists. Overwrite? (y/N): ")
        if response.lower() != 'y':
            print_info("Using existing key")
            return str(key_path)
    
    # Generate key
    cmd = f'ssh-keygen -t ed25519 -C "{email}" -f {key_path} -N ""'
    success, _ = run_command(cmd)
    
    if not success:
        print_error("Failed to generate SSH key")
        sys.exit(1)
    
    print_success(f"SSH key generated at {key_path}")
    
    # Display public key
    pub_key_path = ssh_dir / "id_ed25519.pub"
    with open(pub_key_path, 'r') as f:
        pub_key = f.read().strip()
    
    print_header("Your SSH Public Key")
    print(pub_key)
    print_header("Add this key to Vast.ai")
    print_info("1. Go to: https://cloud.vast.ai/manage-keys/")
    print_info("2. Click 'Add SSH Key'")
    print_info("3. Paste the key above")
    print_info("4. Give it a name (e.g., 'fake-news-detection')")
    
    input("\nPress Enter once you've added the key to Vast.ai...")
    
    return str(key_path)

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

def test_ssh_connection(ip: str, port: str, key_path: str) -> bool:
    """Test SSH connection to instance."""
    print_info("Testing SSH connection...")
    
    cmd = f'ssh -p {port} -i {key_path} -o StrictHostKeyChecking=no -o ConnectTimeout=10 root@{ip} "echo \'Connection successful\'"'
    success, output = run_command(cmd, check=False, capture=True)
    
    if success:
        print_success("SSH connection successful!")
        return True
    else:
        print_error(f"SSH connection failed: {output}")
        return False

def upload_project(ip: str, port: str, key_path: str, local_path: str, remote_path: str = "/workspace/fake-new-detection"):
    """Upload project to Vast.ai instance using rsync."""
    print_info("Uploading project to Vast.ai instance...")
    print_warning("This may take several minutes depending on your data size.")
    
    # Create exclude list
    excludes = [
        ".git/", ".idea/", ".vscode/", ".cursor/", ".windsurf/", ".agent/",
        "notebooks/data/", "notebooks/results/", "*.jpg", "*.jpeg",
        ".openspec/", ".DS_Store", ".env*", "__pycache__/", "*.pyc",
        ".venv", ".pytest_cache/", "mlruns/", "logs/", "checkpoints/",
        "data/jpg/", "openspec/", "*.zip", "*.npz", "*.tar.gz",
        ".ipynb_checkpoints/", "processed_data/", "*.h5", "*.hdf5"
    ]
    
    exclude_args = " ".join([f"--exclude='{exc}'" for exc in excludes])
    
    cmd = f'rsync -avz -e "ssh -p {port} -i {key_path} -o StrictHostKeyChecking=no" {exclude_args} {local_path}/ root@{ip}:{remote_path}/'
    
    success, _ = run_command(cmd, check=False)
    
    if not success:
        print_error("Upload failed. Trying with scp instead...")
        # Fallback to scp
        cmd = f'scp -P {port} -i {key_path} -r {local_path} root@{ip}:/workspace/'
        success, _ = run_command(cmd, check=False)
        
        if not success:
            print_error("Both rsync and scp failed. Please upload manually.")
            return False
    
    print_success("Project uploaded successfully!")
    return True

def run_remote_setup(ip: str, port: str, key_path: str):
    """Run setup commands on remote instance."""
    print_info("Running setup on remote instance...")
    
    setup_commands = [
        "cd /workspace/fake-new-detection",
        "chmod +x vastai/setup_vastai.sh",
        "./vastai/setup_vastai.sh"
    ]
    
    cmd = f'ssh -p {port} -i {key_path} -o StrictHostKeyChecking=no root@{ip} "{" && ".join(setup_commands)}"'
    
    success, _ = run_command(cmd, check=False)
    
    if success:
        print_success("Remote setup completed!")
    else:
        print_error("Remote setup failed. Please run manually:")
        print_info(f"ssh -p {port} -i {key_path} root@{ip}")
        print_info("cd /workspace/fake-new-detection && ./vastai/setup_vastai.sh")

def save_connection_config(ip: str, port: str, key_path: str):
    """Save connection details to config file."""
    config = {
        "ip": ip,
        "port": port,
        "key_path": key_path,
        "user": "root"
    }
    
    config_path = Path(__file__).parent / ".vastai_config.json"
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print_success(f"Connection config saved to {config_path}")

def load_connection_config() -> Optional[dict]:
    """Load connection details from config file."""
    config_path = Path(__file__).parent / ".vastai_config.json"
    
    if config_path.exists():
        with open(config_path, 'r') as f:
            return json.load(f)
    
    return None

def connect_to_instance(ip: str, port: str, key_path: str, port_forwarding: list = None):
    """Connect to Vast.ai instance with optional port forwarding."""
    print_info(f"Connecting to root@{ip}:{port}...")
    
    cmd = f'ssh -p {port} -i {key_path} root@{ip}'
    
    if port_forwarding:
        for local_port, remote_port in port_forwarding:
            cmd += f' -L {local_port}:localhost:{remote_port}'
        print_info(f"Port forwarding: {port_forwarding}")
    
    print_header("SSH Connection Established")
    print_info("You are now connected to your Vast.ai instance.")
    print_info("Type 'exit' to disconnect.")
    print()
    
    os.system(cmd)

def main():
    parser = argparse.ArgumentParser(description="One-click Vast.ai setup for Fake News Detection")
    parser.add_argument("--skip-upload", action="store_true", help="Skip project upload")
    parser.add_argument("--skip-setup", action="store_true", help="Skip remote setup")
    parser.add_argument("--connect-only", action="store_true", help="Only connect, don't setup")
    parser.add_argument("--jupyter", action="store_true", help="Enable Jupyter port forwarding (8888)")
    parser.add_argument("--mlflow", action="store_true", help="Enable MLflow port forwarding (5000)")
    parser.add_argument("--ip", type=str, help="Instance IP address")
    parser.add_argument("--port", type=str, help="SSH port")
    
    args = parser.parse_args()
    
    print_header("Vast.ai Setup for Fake News Detection")
    
    # Step 1: Check SSH key
    print_header("Step 1: SSH Key Setup")
    key_path = check_ssh_key()
    
    if not key_path:
        email = input("Enter email for SSH key (default: vastai@local): ").strip()
        if not email:
            email = "vastai@local"
        key_path = generate_ssh_key(email)
    
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
    
    # Step 3: Test connection
    print_header("Step 3: Test Connection")
    if not test_ssh_connection(ip, port, key_path):
        print_error("Cannot proceed without successful SSH connection")
        sys.exit(1)
    
    # Save config
    save_connection_config(ip, port, key_path)
    
    # Step 4: Upload project
    if not args.skip_upload and not args.connect_only:
        print_header("Step 4: Upload Project")
        local_path = str(Path(__file__).parent)
        
        upload_now = input("Upload project to instance? (Y/n): ").strip().lower()
        if upload_now != 'n':
            upload_project(ip, port, key_path, local_path)
        else:
            print_info("Skipping upload")
    
    # Step 5: Run remote setup
    if not args.skip_setup and not args.connect_only and not args.skip_upload:
        print_header("Step 5: Remote Setup")
        run_now = input("Run setup script on remote instance? (Y/n): ").strip().lower()
        if run_now != 'n':
            run_remote_setup(ip, port, key_path)
        else:
            print_info("Skipping remote setup")
    
    # Step 6: Connect
    if not args.connect_only:
        print_header("Step 6: Connect")
        connect_now = input("Connect to instance now? (Y/n): ").strip().lower()
        if connect_now != 'n':
            port_forwarding = []
            if args.jupyter:
                port_forwarding.append((8888, 8888))
            if args.mlflow:
                port_forwarding.append((5000, 5000))
            
            connect_to_instance(ip, port, key_path, port_forwarding if port_forwarding else None)
    
    # If connect-only mode
    if args.connect_only:
        port_forwarding = []
        if args.jupyter:
            port_forwarding.append((8888, 8888))
        if args.mlflow:
            port_forwarding.append((5000, 5000))
        
        connect_to_instance(ip, port, key_path, port_forwarding if port_forwarding else None)

if __name__ == "__main__":
    main()
