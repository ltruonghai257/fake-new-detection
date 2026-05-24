#!/bin/bash
# Setup script for vast.ai deployment of fake news detection project

set -e

echo "=== Setting up fake news detection on vast.ai ==="

# Update system packages
echo "Updating system packages..."
apt-get update && apt-get install -y git wget curl python3 python3-pip python3-venv

# Clean up disk space
echo "Cleaning up disk space..."
apt-get clean
rm -rf /var/cache/apt/archives/*
rm -rf /tmp/*

# Install uv if not present
if ! command -v uv &> /dev/null; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
else
    echo "uv already installed"
fi

# Create virtual environment with uv
echo "Creating virtual environment with Python 3.10..."
uv venv --python 3.10 .venv
source .venv/bin/activate

# Install PyTorch with CUDA support
echo "Installing PyTorch with CUDA..."
# Detect GPU and compute capability
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
echo "Detected GPU: $GPU_NAME"

# Check if it's an older GPU that needs CUDA 11.8 (GTX 1070 Ti, 1080, etc.)
if [[ "$GPU_NAME" == *"GTX 1070"* ]] || [[ "$GPU_NAME" == *"GTX 1080"* ]] || [[ "$GPU_NAME" == *"GTX 1060"* ]]; then
    echo "Older GPU detected (compute capability 6.x), installing PyTorch with CUDA 11.8..."
    uv pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118 --extra-index-url https://pypi.org/simple
else
    # Detect CUDA version for newer GPUs
    CUDA_VERSION=$(nvcc --version | grep "release" | sed 's/.*release \([0-9]*\.[0-9]*\).*/\1/' | head -1)
    echo "Detected CUDA version: $CUDA_VERSION"

    if [ -z "$CUDA_VERSION" ]; then
        echo "CUDA not found via nvcc, trying nvidia-smi..."
        CUDA_VERSION=$(nvidia-smi | grep "CUDA Version" | sed 's/.*CUDA Version: \([0-9]*\.[0-9]*\).*/\1/')
        echo "Detected CUDA version from nvidia-smi: $CUDA_VERSION"
    fi

    # Install PyTorch based on CUDA version using uv pip (without extra_index_url to avoid NCCL)
    if [[ "$CUDA_VERSION" == "13"* ]] || [[ "$CUDA_VERSION" == "12.1"* ]]; then
        echo "Installing PyTorch for CUDA 12.1 (backward compatible with CUDA 13)..."
        uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121 --extra-index-url https://pypi.org/simple
    elif [[ "$CUDA_VERSION" == "12.0"* ]]; then
        echo "Installing PyTorch for CUDA 12.0..."
        uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu120 --extra-index-url https://pypi.org/simple
    elif [[ "$CUDA_VERSION" == "11.8"* ]]; then
        echo "Installing PyTorch for CUDA 11.8..."
        uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118 --extra-index-url https://pypi.org/simple
    elif [[ "$CUDA_VERSION" == "11.7"* ]]; then
        echo "Installing PyTorch for CUDA 11.7..."
        uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu117 --extra-index-url https://pypi.org/simple
    else
        echo "Unknown CUDA version $CUDA_VERSION, installing latest PyTorch..."
        uv pip install torch torchvision
    fi
fi

# Install other dependencies
echo "Installing project dependencies..."
uv pip install "numpy<2"
uv pip install -r requirements.txt

# Install additional dependencies
echo "Installing additional dependencies..."
uv pip install h5py underthesea

# Downgrade protobuf for compatibility (fixes descriptor error)
echo "Downgrading protobuf for compatibility..."
uv pip install "protobuf==3.20.3"

# Install Jupyter and create kernel
echo "Installing Jupyter..."
uv pip install jupyter jupyterlab ipykernel

# Create Jupyter kernel
echo "Creating Jupyter kernel..."
python -m ipykernel install --user --name=fake_news --display-name="Python (fake_news)"

# Verify CUDA availability
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}')"

echo "=== Setup complete! ==="
echo "Activate environment with: source .venv/bin/activate"
echo "Jupyter kernel 'fake_news' created"
