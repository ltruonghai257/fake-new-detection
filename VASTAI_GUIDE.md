# Complete vast.ai Workflow for Fake News Detection

## Phase 1: Account Setup (One-time)

### 1. Create vast.ai Account

1. Go to https://vast.ai
2. Sign up with email or GitHub
3. Add payment method (credit card required)
4. Verify email address

### 2. Generate SSH Keys (Recommended)

```bash
# On your local machine
ssh-keygen -t ed25519 -C "vastai" -f ~/.ssh/vastai_key
cat ~/.ssh/vastai_key.pub
```

-   Copy the public key output
-   Add it to vast.ai account settings under "SSH Keys"

### 3. Estimate Your Data Size

```bash
# Check your data size locally
du -sh data/
du -sh processed_data/
du -sh checkpoints/
```

-   Total data size determines required disk space
-   For this project: ~5GB (data) + ~2GB (checkpoints) = ~7GB minimum
-   Add buffer: Recommend 20-50GB disk

---

## Phase 2: Launch Instance

### 1. Browse Available Instances

1. Go to https://vast.ai/console/create
2. Filter by:
    - **GPU**: RTX 3090, RTX 4090, or A100 40GB
    - **Disk Space**: 20GB+ (or use persistent storage)
    - **CUDA Version**: 12.1+
    - **Min RAM**: 16GB+
    - **Price**: $0.20-$0.50/hour for RTX 3090 (spot)

### 2. Select Instance Configuration

**Recommended for this project:**

-   **GPU**: RTX 3090 (24GB VRAM) - $0.20-$0.30/hour (spot)
-   **RAM**: 16GB+
-   **Disk**: 30GB+ SSD
-   **Template**: `pytorch/pytorch:2.1.0-cuda12.1-cudnn8-devel`

**Instance Types:**

-   **Spot instances**: Cheapest (~50-70% discount), can be interrupted
-   **On-demand**: More expensive, guaranteed uptime
-   **Recommendation**: Use spot for training, save checkpoints frequently

### 3. Configure Launch Options

-   **Disk Size**: 30GB+ (or add persistent storage)
-   **SSH Key**: Select your SSH key from dropdown
-   **Enable SSH Web**: Check for browser-based terminal backup
-   **Port Forwarding**: Add port 8888 for Jupyter (optional)
-   **Startup Script**: Leave empty (we'll use setup_vastai.sh)

### 4. Launch and Note Connection Details

-   Copy the SSH command: `ssh -p <port> root@<ip>`
-   Note the instance IP and port
-   Save these for later use

---

## Phase 3: Upload Code and Data

### Option A: rsync (Recommended - Fastest for small datasets)

```bash
# From your local machine
# Use the SSH command from vast.ai (includes port)
rsync -avz -e 'ssh -p <port>' \
  /Users/haila/Library/CloudStorage/GoogleDrive-latruonghai@gmail.com/My\ Drive/Thesis_Final/fake-new-detection/ \
  root@<ip>:/workspace/fake-new-detection/

# Exclude unnecessary files to speed up transfer
rsync -avz -e 'ssh -p 50769' \
    --exclude='.git/' \
    --exclude='.idea/' \
    --exclude='.vscode/' \
    --exclude='.cursor/' \
    --exclude='notebooks/data/' \
    --exclude='notebooks/results/' \
    --exclude='*.jpg' \
    --exclude='*.jpeg' \
    --exclude='.openspec/' \
    --exclude='.windsurf/' \
    --exclude='.agent/' \
    --exclude='.gemini/' \
    --exclude='*.pkl'\
    --exclude='.DS_Store' \
    --exclude='.env*' \
    --exclude='__pycache__/' \
    --exclude='*.pyc' \
    --exclude='.venv' \
    --exclude='.pytest_cache/' \
    --exclude='mlruns/' \
    --exclude='logs/' \
    --exclude='checkpoints/' \
    --exclude='data/jpg/' \
    --exclude='openspec/' \
    --exclude='*.zip'\
    --exclude='*.npz' \
    --exclude='*.tar.gz' \
    /Users/haila/Library/CloudStorage/GoogleDrive-latruonghai@gmail.com/My\ Drive/Thesis_Final/fake-new-detection/ \
    root@220.82.52.202:/workspace/fake-new-detection/
```

### Option B: Compressed Archive (Best for large datasets)

```bash
# On local machine: Create compressed archive in parent directory
cd /Users/haila/Library/CloudStorage/GoogleDrive-latruonghai@gmail.com/My\ Drive/Thesis_Final/
tar -czf fake-news-detection.tar.gz \
  --exclude='fake-new-detection/.git' \
  --exclude='fake-new-detection/__pycache__' \
  --exclude='fake-new-detection/*.pyc' \
  --exclude='fake-new-detection/notebooks/.ipynb_checkpoints' \
  --exclude='fake-new-detection/.pytest_cache' \
  --exclude='fake-new-detection/mlruns' \
  fake-new-detection

# Upload archive
scp -P <port> fake-news-detection.tar.gz root@<ip>:/workspace/

# On vast.ai: Extract
ssh -p <port> root@<ip>
cd /workspace
tar -xzf fake-news-detection.tar.gz
rm fake-news-detection.tar.gz
```

### Option C: Git Clone (Best for code-only, separate data upload)

```bash
# On vast.ai instance
ssh -p <port> root@<ip>
cd /workspace
git clone <your-repo-url> fake-new-detection
cd fake-new-detection

# Upload data separately using rsync or scp
```

### Option D: Google Drive (If data is already on GDrive)

```bash
# On vast.ai instance
ssh -p <port> root@<ip>
pip install gdown

# Download specific files (need shareable GDrive links)
gdown <google-drive-file-id> -O data/news_data.json
gdown <google-drive-file-id> -O data/images.zip
```

---

## Phase 4: Environment Setup

### 1. SSH into Instance

```bash
ssh -p <port> root@<ip>
```

### 2. Run Setup Script

```bash
cd /workspace/fake-new-detection
chmod +x setup_vastai.sh
./setup_vastai.sh
```

### 3. Activate Environment

```bash
source ~/.bashrc
conda activate fake_news
```

### 4. Verify Setup

```bash
# Check CUDA
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

# Check data integrity
ls -lh data/
ls -lh processed_data/hdf5/
du -sh .

# Test imports
python -c "import transformers, h5py, sklearn; print('All imports successful')"
```

---

## Phase 5: Run Training

### Option A: Training Script (Recommended for production)

```bash
cd /workspace/fake-new-detection
conda activate fake_news

# Run with specific config
python examples/train_coolant_official.py \
  --batch_size 32 \
  --epochs 30 \
  --lr 1e-3 \
  --device cuda \
  --checkpoint_dir checkpoints/
```

### Option B: Jupyter Notebook (For experimentation)

```bash
# Install Jupyter
pip install jupyter

# Start Jupyter with remote access
jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root

# On local machine: Create SSH tunnel
ssh -L 8888:localhost:8888 -p <port> root@<ip>

# Open browser to: http://localhost:8888
# Use token from Jupyter output to authenticate
```

### Option C: Screen/Tmux Session (For long-running jobs)

```bash
# Install tmux if not present
apt-get install tmux

# Create session
tmux new -s training

# Run training
conda activate fake_news
python examples/train_coolant_official.py

# Detach: Ctrl+B, then D
# Reattach: tmux attach -s training
```

---

## Phase 6: Monitor Training

### 1. Monitor GPU Usage

```bash
# Real-time GPU monitoring
watch -n 1 nvidia-smi

# Or use gpustats
pip install gpustats
gpustat -i
```

### 2. Monitor Training Progress

```bash
# Tail training logs
tail -f logs/training.log

# Check MLflow (if using)
mlflow ui --host 0.0.0.0 --port 5000

# Access via SSH tunnel
ssh -L 5000:localhost:5000 -p <port> root@<ip>
# Open browser to: http://localhost:5000
```

### 3. Check Disk Space

```bash
df -h
du -sh /workspace/fake-new-detection/
du -sh /workspace/fake-new-detection/checkpoints/
```

---

## Phase 7: Download Results

### 1. Download Checkpoints

```bash
# From local machine
rsync -avz -e 'ssh -p <port>' \
  root@<ip>:/workspace/fake-new-detection/checkpoints/ \
  /Users/haila/Library/CloudStorage/GoogleDrive-latruonghai@gmail.com/My\ Drive/Thesis_Final/fake-new-detection/checkpoints/
```

### 2. Download Logs

```bash
rsync -avz -e 'ssh -p <port>' \
  root@<ip>:/workspace/fake-new-detection/logs/ \
  /Users/haila/Library/CloudStorage/GoogleDrive-latruonghai@gmail.com/My\ Drive/Thesis_Final/fake-new-detection/logs/
```

### 3. Download MLflow Artifacts

```bash
# If using MLflow
rsync -avz -e 'ssh -p <port>' \
  root@<ip>:/workspace/fake-new-detection/mlruns/ \
  /Users/haila/Library/CloudStorage/GoogleDrive-latruonghai@gmail.com/My\ Drive/Thesis_Final/fake-new-detection/mlruns/
```

---

## Phase 8: Instance Management

### 1. Stop Instance (When Pausing)

-   Go to vast.ai console
-   Click "Stop" on your instance
-   You're not billed while stopped
-   Data persists on disk

### 2. Terminate Instance (When Done)

-   **WARNING**: This deletes all data on the instance disk
-   Download all important data first
-   Go to vast.ai console
-   Click "Destroy" on your instance
-   You're no longer billed

### 3. Use Persistent Storage (For Long-term Projects)

-   Add persistent storage add-on when launching
-   Data survives instance termination
-   Can be attached to new instances
-   Costs extra but worth it for long projects

---

## Cost Optimization Strategies

### 1. Use Spot Instances

-   **Savings**: 50-70% cheaper than on-demand
-   **Risk**: Can be interrupted with 2-minute notice
-   **Mitigation**: Save checkpoints frequently (every epoch)
-   **Best for**: Training experiments, hyperparameter tuning

### 2. Right-size Your GPU

-   **RTX 3090 (24GB)**: $0.20-$0.30/hour - Good for most models
-   **RTX 4090 (24GB)**: $0.40-$0.60/hour - Faster training
-   **A100 40GB**: $0.80-$1.20/hour - Large models, big batches
-   **A100 80GB**: $1.50-$2.00/hour - Very large models

### 3. Optimize Training Time

-   Use mixed precision training (faster, less VRAM)
-   Increase batch size to utilize GPU fully
-   Use gradient accumulation for effective large batches
-   Profile code to find bottlenecks

### 4. Minimize Idle Time

-   Stop instances when not training
-   Use tmux/screen for long-running jobs
-   Monitor training to catch issues early
-   Terminate immediately after downloading results

### 5. Data Strategy

-   Upload data once, reuse across runs
-   Use persistent storage for datasets
-   Preprocess data into HDF5 for faster loading
-   Cache features to avoid recomputation

---

## Troubleshooting

### Out of Memory (OOM)

**Symptoms**: CUDA out of memory error
**Solutions**:

-   Reduce batch size (e.g., 32 → 16 → 8)
-   Use gradient accumulation
-   Enable mixed precision: `torch.cuda.amp.autocast()`
-   Clear cache: `torch.cuda.empty_cache()`
-   Switch to GPU with more VRAM

### Slow Data Loading

**Symptoms**: GPU utilization < 50%, training slow
**Solutions**:

-   Ensure data is on SSD (not network storage)
-   Use HDF5 format (you have `hdf5_dataset.py`)
-   Increase `num_workers` in DataLoader (but keep at 0 for HDF5)
-   Pin memory: `pin_memory=True` in DataLoader
-   Preload data to RAM if small enough

### Connection Issues

**Symptoms**: SSH connection drops, timeout
**Solutions**:

-   Use SSH keys instead of passwords
-   Enable "SSH Web" for browser-based terminal backup
-   Add to SSH config: `ServerAliveInterval 60`
-   Use tmux/screen to resume sessions
-   Check vast.ai status page for outages

### Instance Interruption (Spot)

**Symptoms**: Instance stops unexpectedly
**Solutions**:

-   Spot instances can be interrupted
-   Save checkpoints every epoch
-   Use persistent storage for checkpoints
-   Monitor instance health in vast.ai console
-   Switch to on-demand if interruptions are frequent

### Permission Errors

**Symptoms**: Permission denied when accessing files
**Solutions**:

```bash
# Fix permissions
chmod -R 755 /workspace/fake-new-detection/
chown -R root:root /workspace/fake-new-detection/

# Or run with sudo
sudo python script.py
```

---

## Quick Reference Commands

```bash
# SSH connect
ssh -p <port> root@<ip>

# Upload code
rsync -avz -e 'ssh -p <port>' local_dir/ root@<ip>:/workspace/

# Download results
rsync -avz -e 'ssh -p <port>' root@<ip>:/workspace/checkpoints/ local_dir/

# Monitor GPU
watch -n 1 nvidia-smi

# Check disk space
df -h

# Activate environment
conda activate fake_news

# Start training
python examples/train_coolant_official.py

# Start Jupyter
jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root

# SSH tunnel for Jupyter
ssh -L 8888:localhost:8888 -p <port> root@<ip>

# Create tmux session
tmux new -s training

# Detach tmux
Ctrl+B, then D

# Reattach tmux
tmux attach -s training
```

---

## Data Considerations for This Project

Your project has data in:

-   `data/json/` - Raw JSON news data (~500MB)
-   `data/jpg/` - Images directory (~2GB)
-   `processed_data/hdf5/` - Preprocessed HDF5 files (~5GB)
-   `checkpoints/` - Model checkpoints (~1-2GB)

**Total estimated size**: ~8-10GB

**Recommendations**:

-   Use 30GB+ disk for comfort
-   Upload preprocessed HDF5 files if available (faster than raw data)
-   Use persistent storage if running multiple experiments
-   Download checkpoints regularly to avoid loss
