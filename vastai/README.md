# Vast.ai Integration

This folder contains all scripts and documentation for Vast.ai GPU instance management for the Fake News Detection project.

## Files

- **setup_vastai.py** - One-click setup script for Vast.ai instances
  - Generates SSH keys
  - Uploads project to remote instance
  - Runs environment setup
  - Establishes SSH connection with optional port forwarding

- **setup_vastai.sh** - Shell script for remote environment setup
  - Installs Python dependencies
  - Sets up PyTorch with CUDA support
  - Configures Jupyter and other tools

- **download_from_vastai.py** - Script to download trained models and artifacts
  - Downloads checkpoints
  - Downloads logs and mlruns
  - Downloads processed data
  - Supports selective or bulk downloads

- **VASTAI_GUIDE.md** - Comprehensive documentation
  - Step-by-step setup instructions
  - SSH connection guide
  - Port forwarding instructions
  - Troubleshooting tips

## Quick Start

### Setup and Upload
```bash
python vastai/setup_vastai.py
```

### Connect with Port Forwarding
```bash
python vastai/setup_vastai.py --ip <IP> --port <PORT> --jupyter --mlflow
```

### Download Results
```bash
# Download all artifacts
python vastai/download_from_vastai.py --all

# Download only checkpoints
python vastai/download_from_vastai.py --checkpoints

# Download specific file
python vastai/download_from_vastai.py --file /workspace/fake-new-detection/checkpoints/model.pth
```

## Workflow

1. **Initial Setup**: Run `setup_vastai.py` to configure SSH keys and upload project
2. **Training**: Connect to instance and run training scripts
3. **Download**: Use `download_from_vastai.py` to fetch trained models and logs

## Configuration

Connection details are saved to `.vastai_config.json` in the project root for convenience.

## See Also

- [VASTAI_GUIDE.md](VASTAI_GUIDE.md) - Detailed documentation
- Project README for training instructions
