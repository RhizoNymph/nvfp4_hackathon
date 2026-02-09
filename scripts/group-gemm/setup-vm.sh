#!/bin/bash

apt update
apt install nano
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
git clone https://github.com/gpu-mode/reference-kernels
cd reference-kernels
uv init
uv add nvidia-cutlass-dsl torch pyyaml
cd problems/nvidia/nvfp4_group_gemm/
wget https://raw.githubusercontent.com/RhizoNymph/nvfp4_hackathon/refs/heads/main/profile_group_gemm.py
wget https://raw.githubusercontent.com/RhizoNymph/nvfp4_hackathon/refs/heads/main/profile_group_gemm.sh
chmod +x profile_group_gemm.sh
