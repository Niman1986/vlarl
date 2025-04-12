# Installation

The following guidance works well for a machine with 3090 GPU | cuda 12.1 | ubuntu 22.04 LTS, a machine with 3090 GPU | cuda 11.6 | ubuntu 20.04, and more machines.

For possible errors, please see [ERROR_CATCH.md](ERROR_CATCH.md). If you encounter any other problem, feel free to open an issue.

```bash
# Conda environment setup
conda create -n vlarl python=3.10
conda activate vlarl

git clone https://github.com/GuanxingLu/vlarl.git
cd openvla
pip install -e .

# PyTorch installation (CUDA 12.1)
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121

# Install Flash Attention 2 for training (https://github.com/Dao-AILab/flash-attention)
#   =>> If you run into difficulty, try `pip cache remove flash_attn` first
pip install packaging ninja
ninja --version; echo $?  # Verify Ninja --> should return exit code "0"
pip install "flash-attn==2.5.5" --no-build-isolation

# Install other Python dependencies
pip install -r requirements.txt

# Set up Weights & Biases for experiment logging
wandb login

# LIBERO simulation
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git
cd LIBERO
pip install -e .

cd 
pip install -r experiments/robot/libero/libero_requirements.txt

# Download LIBERO SFT dataset
git lfs clone https://huggingface.co/datasets/openvla/modified_libero_rlds
```