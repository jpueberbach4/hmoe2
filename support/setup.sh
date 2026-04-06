#!/bin/bash
# ===============================================================
# Setup - High Performance HMoE2 Environment
# ===============================================================
# This script sets up a modern MoE development environment:
# - Verifies GPU & CUDA
# - Installs system dependencies
# - Upgrades CMake if needed
# - Sets up Python venv
# - Installs CUDA-optimized PyTorch
# - Compiles Signatory (Rough Paths) from source
# - Installs remaining Python dependencies
# ===============================================================

set -e  # Exit immediately on any error

echo "🚀 Initializing Quantitative Environment Setup..."

# ===============================================================
# 1️⃣ Hardware Verification
# Check for CUDA-capable GPU via nvidia-smi
# ===============================================================
if ! command -v nvidia-smi &> /dev/null; then
    echo "❌ ERROR: nvidia-smi not found. CUDA GPU is required for HMoE2."
    exit 1
fi
echo "✅ CUDA-capable GPU detected."

# ===============================================================
# 2️⃣ System Dependencies & Toolchain
# Install build-essential, OpenMP, wget, Python dev tools
# ===============================================================
echo "📦 Installing base system dependencies..."
sudo apt-get update
sudo apt-get install -y build-essential libomp-dev wget gpg python3-venv python3-pip python3-dev

# ===============================================================
# 3️⃣ Modernize CMake
# C++ extensions like Signatory require >=3.18
# ===============================================================
CURRENT_CMAKE_VER=$(cmake --version 2>/dev/null | grep -oE '[0-9]+\.[0-9]+' | head -1 || echo "0.0")
if $(awk 'BEGIN {exit !('"$CURRENT_CMAKE_VER"' < 3.18)}'); then
    echo "📦 Upgrading CMake to >=3.18 from Kitware..."
    wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null \
        | gpg --dearmor - \
        | sudo tee /usr/share/keyrings/kitware-archive-keyring.gpg >/dev/null
    echo 'deb [signed-by=/usr/share/keyrings/kitware-archive-keyring.gpg] https://apt.kitware.com/ubuntu/ focal main' \
        | sudo tee /etc/apt/sources.list.d/kitware.list >/dev/null
    sudo apt-get update
    sudo apt-get install -y cmake
else
    echo "✅ CMake version $CURRENT_CMAKE_VER is sufficient."
fi

# Install NCCL for multi-GPU PyTorch communications
sudo apt-get install -y libnccl-dev

# ===============================================================
# 4️⃣ Python Virtual Environment Setup
# ===============================================================
echo "🐍 Creating Python virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
fi
source venv/bin/activate

echo "⬆️ Upgrading pip, setuptools, wheel, and ninja..."
pip install --upgrade pip setuptools wheel ninja

# ===============================================================
# 5️⃣ Install CUDA-Optimized PyTorch
# ===============================================================
echo "🔥 Installing PyTorch with CUDA 12.1..."
# This must be installed BEFORE signatory so the C++ compiler can link against libtorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# ===============================================================
# 6️⃣ Compile Signatory from Source
# ===============================================================
echo "🧮 Compiling Signatory (Rough Path Signatures) from source..."
# --no-binary forces pip to download the sdist and compile the C++ extensions locally
pip install signatory --no-binary signatory

# ===============================================================
# 7️⃣ Install Additional Python Dependencies
# ===============================================================
echo "📦 Installing other required Python modules..."
pip install -r requirements.txt

echo "✅ HMoE2 Environment Setup Complete!"