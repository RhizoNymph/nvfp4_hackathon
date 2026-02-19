#!/bin/bash

apt update
apt install nano
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
git clone https://github.com/RhizoNymph/nvfp4_hackathon
cd nvfp4_hackathon/kernels
uv init
uv add nvidia-cutlass-dsl torch pyyaml
cd group_gemm/
chmod +x profile_group_gemm.sh
sudo apt install nsight-compute-2025.4.1
sudo apt install nsight-systems
export PATH=/opt/nvidia/nsight-compute/2025.4.1/:$PATH  