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
  - Sets up PyTorch with CUDA support (cu118 → cu124+ auto-detected)
  - Configures Jupyter and other tools

- **autosync_vastai.py** - Continuous background sync to local machine
  - Polls every N minutes (default 5)
  - Syncs checkpoints, logs, mlruns, stage2_results
  - Safe to leave running overnight — handles connection drops gracefully
  - Stop the instance any time; your local copy stays current

- **download_from_vastai.py** - One-shot download of all artifacts
  - Downloads checkpoints, logs, mlruns, processed data
  - Supports selective or bulk downloads

- **VASTAI_GUIDE.md** - Comprehensive documentation

## Quick Start

### 1. Setup and Upload
```bash
python vastai/setup_vastai.py --ip <IP> --port <PORT> --jupyter
```

### 2. Start overnight training (on remote via SSH)
```bash
source .venv/bin/activate
tmux new -s train
python examples/train_coolant_official.py   # or run notebooks via Jupyter
# Ctrl+B, D to detach
```

### 3. Auto-sync to local (run this on your local machine)
```bash
# Syncs every 5 min — safe to leave overnight
python vastai/autosync_vastai.py

# Or custom interval
python vastai/autosync_vastai.py --interval 3

# Override connection if needed
python vastai/autosync_vastai.py --ip <IP> --port <PORT> --interval 5
```
Stop the Vast.ai instance whenever you want — the last sync is already local.

### 4. One-shot download (after training finishes)
```bash
python vastai/download_from_vastai.py --all
```

## Overnight Training Workflow

1. `python vastai/setup_vastai.py --ip <IP> --port <PORT> --jupyter` — setup + connect
2. On remote: `tmux new -s train && source .venv/bin/activate && jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root`
3. On local (second terminal): `python vastai/autosync_vastai.py` — starts polling
4. Run notebooks in browser at http://localhost:8888
5. Detach tmux, go to sleep
6. Stop instance in Vast.ai console whenever convenient — auto-sync already grabbed your checkpoints

## Configuration

Connection details are saved to `.vastai_config.json` in the project root for convenience.

## See Also

- [VASTAI_GUIDE.md](VASTAI_GUIDE.md) - Detailed documentation
- Project README for training instructions
