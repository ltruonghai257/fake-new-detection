# Vast.ai Setup Guide for Fake News Detection

One-click setup and SSH connection guide for training on Vast.ai GPU instances.

## Quick Start

```bash
# Run the one-click setup script
python vastai/setup_vastai.py
```

This script will:
- Check/generate SSH keys
- Guide you through adding the key to Vast.ai
- Upload your project
- Run environment setup
- Connect to your instance

## Step-by-Step Manual Setup

### Step 1: Generate SSH Key (One-time)

**Option A: Using the Python script (Recommended)**
```bash
python vastai/setup_vastai.py
```
The script will automatically guide you through SSH key generation.

**Option B: Manual generation**
```bash
ssh-keygen -t ed25519 -C "your_email@example.com"
```

This creates:
- `~/.ssh/id_ed25519` - Private key (NEVER share)
- `~/.ssh/id_ed25519.pub` - Public key (safe to share)

### Step 2: Add SSH Key to Vast.ai

1. Go to https://cloud.vast.ai/manage-keys/
2. Click "Add SSH Key"
3. Paste your public key:
   ```bash
   cat ~/.ssh/id_ed25519.pub
   ```
4. Give it a name (e.g., "fake-news-detection")
5. Click "Add"

**Note**: Keys only apply to NEW instances. Existing instances won't get the key automatically.

### Step 3: Launch Vast.ai Instance

1. Go to https://vast.ai/console/create
2. Filter by:
   - **GPU**: RTX 3090 (24GB VRAM) or RTX 4090
   - **Disk Space**: 30GB+
   - **CUDA Version**: 12.1+
   - **Min RAM**: 16GB+
   - **Price**: $0.20-$0.50/hour for RTX 3090 (spot)

3. Configure:
   - **Disk Size**: 30GB+
   - **SSH Key**: Select your key from dropdown
   - **Enable SSH Web**: Check for browser terminal backup
   - **Template**: `pytorch/pytorch:2.1.0-cuda12.1-cudnn8-devel`

4. Click "Deploy"
5. Copy the SSH command: `ssh -p <port> root@<ip>`

### Step 4: Connect Using Python Script

**Basic connection:**
```bash
python vastai/setup_vastai.py --ip <IP> --port <PORT>
```

**With Jupyter port forwarding:**
```bash
python vastai/setup_vastai.py --ip <IP> --port <PORT> --jupyter
```
Then open http://localhost:8888 in your browser.

**With MLflow port forwarding:**
```bash
python vastai/setup_vastai.py --ip <IP> --port <PORT> --mlflow
```
Then open http://localhost:5000 in your browser.

**Connect only (skip upload/setup):**
```bash
python vastai/setup_vastai.py --ip <IP> --port <PORT> --connect-only
```

**Interactive mode (prompts for everything):**
```bash
python vastai/setup_vastai.py
```

### Step 5: Manual SSH Connection (Alternative)

If you prefer manual SSH:

```bash
ssh -p <PORT> root@<IP>
```

**With port forwarding:**
```bash
# Forward Jupyter (8888) and MLflow (5000)
ssh -p <PORT> root@<IP> -L 8888:localhost:8888 -L 5000:localhost:5000
```

**First time connection**: You'll see a fingerprint verification:
```
The authenticity of host '[IP]:PORT' can't be established.
ED25519 key fingerprint is SHA256:...
Are you sure you want to continue connecting (yes/no/[fingerprint])? yes
```
Type `yes` to continue.

### Step 6: Environment Setup (Manual)

If you skipped the Python script's auto-setup:

```bash
# On the remote instance
cd /workspace/fake-new-detection
chmod +x vastai/setup_vastai.sh
./vastai/setup_vastai.sh
```

This installs:
- Python 3.10 with uv
- PyTorch with CUDA support
- All project dependencies
- Jupyter kernel

### Step 7: Verify Setup

```bash
# Check CUDA
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

# Check data
ls -lh data/
ls -lh processed_data/hdf5/
```

## SSH Connection Types

### Direct SSH (Recommended)
- **Faster**: Direct connection to instance
- **More reliable**: No proxy overhead
- **Requirement**: Your machine must have open ports
- **Use when**: Port forwarding works, you have stable internet

### Proxy SSH (Fallback)
- **Slower**: Routes through Vast proxy
- **Works everywhere**: No port requirements
- **Use when**: Direct connection fails, behind firewall
- **How**: Vast.ai provides proxy URL in instance details

## Port Forwarding Guide

Forward local ports to access remote services:

```bash
# Basic syntax
ssh -p <SSH_PORT> root@<IP> -L <LOCAL_PORT>:localhost:<REMOTE_PORT>

# Jupyter Notebook
ssh -p 20544 root@142.214.185.187 -L 8888:localhost:8888
# Access at: http://localhost:8888

# MLflow
ssh -p 20544 root@142.214.185.187 -L 5000:localhost:5000
# Access at: http://localhost:5000

# Multiple ports
ssh -p 20544 root@142.214.185.187 -L 8888:localhost:8888 -L 5000:localhost:5000
```

## Data from Google Drive (rclone)

This project stores all large data on Google Drive (via `DATA_ROOT`). On Vast.ai, Google
Drive is not mounted — use **rclone** to sync data to and from the instance.

### `DATA_ROOT` by Environment

| Platform | Variable | Example Path |
|----------|----------|--------------|
| **Vast.ai** | `DATA_ROOT` | `/workspace/fake-news-data-for-thesis` |
| **Google Colab** | `DATA_ROOT` | `/content/drive/MyDrive/Thesis_Final/fake-news-data-for-thesis` |
| **macOS** | `DATA_ROOT` | `/Users/[USERNAME]/Library/CloudStorage/GoogleDrive-[EMAIL]/My Drive/Thesis_Final/fake-news-data-for-thesis` |
| **Windows** | `DATA_ROOT` | `C:/Users/[USERNAME]/Google Drive/My Drive/Thesis_Final/fake-news-data-for-thesis` |

**Platform Notes:**

- **Google Colab**: Mount Drive first before running the pipeline:
  ```python
  from google.colab import drive
  drive.mount('/content/drive')
  ```
- **Windows**: Use forward slashes in `.env` (`C:/Users/...`) OR use raw strings in Python (`r"C:\Users\..."`).
- **macOS**: The CloudStorage path format varies by Google account email. Check your exact path with:
  ```bash
  ls ~/Library/CloudStorage/
  ```

### One-time: Install and configure rclone on Vast.ai

```bash
# On the Vast.ai instance
curl https://rclone.org/install.sh | sudo bash

# Configure Google Drive remote (opens browser link for auth)
rclone config
# → New remote → name it "gdrive" → type: Google Drive → follow OAuth prompts
```

> **Tip**: If the instance has no browser, run `rclone authorize "drive"` on your local
> machine, copy the token, then paste it during `rclone config` on the remote.

### Sync data from Drive → Vast.ai (before training)

```bash
GDRIVE_FOLDER="Thesis_Final/fake-news-data-for-thesis"
LOCAL_ROOT="/workspace/fake-news-data-for-thesis"

# Sync preprocessed data and JSON (skip raw images to save time/disk if not needed)
rclone copy "gdrive:${GDRIVE_FOLDER}/data/json"           "${LOCAL_ROOT}/data/json"           --progress
rclone copy "gdrive:${GDRIVE_FOLDER}/processed_data"      "${LOCAL_ROOT}/processed_data"      --progress

# Optionally sync images (large, only needed if training uses raw images)
rclone copy "gdrive:${GDRIVE_FOLDER}/data/jpg"            "${LOCAL_ROOT}/data/jpg"            --progress
```

### Set DATA_ROOT on Vast.ai

```bash
cp .env.vastai.example .env
# .env already contains: DATA_ROOT=/workspace/fake-news-data-for-thesis
```

### Sync results back to Drive (after training)

```bash
# Upload training checkpoints and results back to Drive
rclone copy "${LOCAL_ROOT}/training" "gdrive:${GDRIVE_FOLDER}/training" --progress
rclone copy "${LOCAL_ROOT}/mlruns"   "gdrive:${GDRIVE_FOLDER}/mlruns"   --progress
```

---

## File Transfer (Code)

### Upload Project (Python Script)
```bash
python vastai/setup_vastai.py
# Follow prompts to upload
```

### Manual Upload with rsync
```bash
rsync -avz -e 'ssh -p <PORT>' \
  --exclude='.git/' \
  --exclude='__pycache__/' \
  --exclude='*.pyc' \
  --exclude='.venv' \
  --exclude='data/' \
  --exclude='processed_data/' \
  --exclude='training/' \
  --exclude='mlruns/' \
  /path/to/fake-new-detection/ \
  root@<IP>:/workspace/fake-new-detection/
```

> **Note**: Data directories (`data/`, `processed_data/`, `training/`) are excluded from
> rsync — they come from Google Drive via rclone (see section above).

### Download Results
```bash
# Download checkpoints via rsync (alternative to rclone)
rsync -avz -e 'ssh -p <PORT>' \
  root@<IP>:/workspace/fake-news-data-for-thesis/training/ \
  ./training/
```

### Using SCP
```bash
# Upload single file
scp -P <PORT> local_file.py root@<IP>:/workspace/fake-new-detection/

# Download single file
scp -P <PORT> root@<IP>:/workspace/fake-news-data-for-thesis/training/best_model.pth ./
```

## Running Training

### Option A: Training Script
```bash
# On remote instance
source .venv/bin/activate
python examples/train_coolant_official.py --batch_size 32 --epochs 30
```

### Option B: Jupyter Notebook
```bash
# Start Jupyter on remote
jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root

# On local: Forward port (if not already done)
ssh -p <PORT> root@<IP> -L 8888:localhost:8888

# Open browser: http://localhost:8888
# Use token from Jupyter output
```

### Option C: Tmux (Long-running jobs)
```bash
# On remote instance
tmux new -s training

# Run training
source .venv/bin/activate
python examples/train_coolant_official.py

# Detach: Ctrl+B, then D
# Reattach: tmux attach -s training
# List sessions: tmux ls
# Kill session: tmux kill-session -t training
```

**Disable auto-tmux on Vast.ai** (if it interferes):
```bash
touch ~/.no_auto_tmux
```

## Monitoring

### GPU Usage
```bash
# Real-time monitoring
watch -n 1 nvidia-smi

# Or use gpustat
pip install gpustat
gpustat -i
```

### Training Progress
```bash
# Tail logs
tail -f logs/training.log

# Check disk space
df -h
du -sh /workspace/fake-new-detection/
```

## Troubleshooting

### Permission Denied (publickey)
```
Permission denied (publickey)
```
**Solutions**:
1. Verify SSH key is added to Vast.ai account
2. Check you're using the correct key: `ssh -i ~/.ssh/id_ed25519 -p <PORT> root@<IP>`
3. For existing instances, add key via instance SSH settings
4. Regenerate key if corrupted

### Connection Timeout
```
ssh: connect to host <IP> port <PORT>: Connection timed out
```
**Solutions**:
1. Check instance is running in Vast.ai console
2. Verify IP and PORT are correct
3. Try proxy SSH if direct fails
4. Check firewall settings

### Host Key Verification Failed
```
WARNING: REMOTE HOST IDENTIFICATION HAS CHANGED!
```
**Solutions**:
```bash
# Remove old host key
ssh-keygen -R [<IP>]:<PORT>

# Or remove all known hosts for this IP
ssh-keygen -R <IP>
```

### Instance Interruption (Spot)
**Symptoms**: Instance stops unexpectedly
**Solutions**:
- Spot instances can be interrupted with 2-minute notice
- Save checkpoints every epoch
- Use persistent storage for checkpoints
- Monitor instance health in Vast.ai console
- Switch to on-demand if interruptions are frequent

### Out of Memory (OOM)
**Symptoms**: CUDA out of memory error
**Solutions**:
- Reduce batch size (32 → 16 → 8)
- Use gradient accumulation
- Enable mixed precision: `torch.cuda.amp.autocast()`
- Clear cache: `torch.cuda.empty_cache()`
- Switch to GPU with more VRAM

## VS Code Remote SSH Integration

### Install Remote SSH Extension
1. Install "Remote - SSH" extension in VS Code
2. Press `Cmd+Shift+P` → "Remote-SSH: Connect to Host"
3. Enter: `root@<IP> -p <PORT>`
4. Or add to `~/.ssh/config`:

```
Host vastai
    HostName <IP>
    Port <PORT>
    User root
    IdentityFile ~/.ssh/id_ed25519
    StrictHostKeyChecking no
```

Then connect via: `Cmd+Shift+P` → "Remote-SSH: Connect to Host" → "vastai"

## Cost Optimization

### Use Spot Instances
- **Savings**: 50-70% cheaper than on-demand
- **Risk**: Can be interrupted with 2-minute notice
- **Mitigation**: Save checkpoints frequently

### Right-size Your GPU
- **RTX 3090 (24GB)**: $0.20-$0.30/hour - Good for most models
- **RTX 4090 (24GB)**: $0.40-$0.60/hour - Faster training
- **A100 40GB**: $0.80-$1.20/hour - Large models
- **A100 80GB**: $1.50-$2.00/hour - Very large models

### Minimize Idle Time
- Stop instances when not training
- Use tmux for long-running jobs
- Terminate immediately after downloading results

## Data Size Considerations

Your project data:
- `data/json/` - Raw JSON news data (~500MB)
- `data/jpg/` - Images directory (~2GB)
- `processed_data/hdf5/` - Preprocessed HDF5 files (~5GB)
- `checkpoints/` - Model checkpoints (~1-2GB)

**Total**: ~8-10GB

**Recommendations**:
- Use 30GB+ disk for comfort
- Upload preprocessed HDF5 files if available (faster than raw data)
- Use persistent storage if running multiple experiments
- Download checkpoints regularly to avoid loss

## Quick Reference

```bash
# One-click setup
python vastai/setup_vastai.py

# Connect with port forwarding
python vastai/setup_vastai.py --ip <IP> --port <PORT> --jupyter --mlflow

# Manual SSH
ssh -p <PORT> root@<IP>

# SSH with port forwarding
ssh -p <PORT> root@<IP> -L 8888:localhost:8888 -L 5000:localhost:5000

# Upload project
rsync -avz -e 'ssh -p <PORT>' ./ root@<IP>:/workspace/fake-new-detection/

# Download results
rsync -avz -e 'ssh -p <PORT>' root@<IP>:/workspace/checkpoints/ ./checkpoints/

# Monitor GPU
watch -n 1 nvidia-smi

# Tmux session
tmux new -s training  # Create
Ctrl+B, D             # Detach
tmux attach -s training  # Reattach

# Activate environment
source .venv/bin/activate

# Run training
python examples/train_coolant_official.py
```

## Official Vast.ai Documentation

For more details, see:
- **SSH Connection**: https://docs.vast.ai/guides/instances/connect/ssh
- **Instance Management**: https://docs.vast.ai/guides/instances/
- **Vast CLI**: https://docs.vast.ai/cli/hello-world
