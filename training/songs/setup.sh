#!/bin/sh
# One-shot setup: ACE-Step 1.5 (system python + torch 2.8), checkpoints, basic-pitch venv, demucs.
mkdir -p "${WORK:-work}" && cd "${WORK:-work}";
git clone --depth 1 https://github.com/ace-step/ACE-Step-1.5.git > s_clone.log 2>&1
(uvx --from 'huggingface_hub[hf_xet]' hf download ACE-Step/Ace-Step1.5 --local-dir ACE-Step-1.5/checkpoints > s_dl.log 2>&1; echo DLDONE >> s_dl.log) &
(uv pip install --system --break-system-packages 'transformers>=4.51.0,<4.58.0' 'diffusers>=0.37.0' scipy soundfile loguru einops accelerate diskcache numba vector-quantize-pytorch toml modelscope pytorch-wavelets pywavelets 'setuptools<72' matplotlib typer-slim peft xxhash demucs > s_sys.log 2>&1 && uv pip install --system --break-system-packages --no-deps -e ACE-Step-1.5/acestep/third_parts/nano-vllm >> s_sys.log 2>&1; echo SYSDONE >> s_sys.log) &
(uv venv bp --python 3.11 > s_bp.log 2>&1 && pip --python bp/bin/python install basic-pitch librosa onnxruntime 'setuptools<72' >> s_bp.log 2>&1 && pip --python bp/bin/python uninstall -y tensorflow tensorflow-estimator tensorflow-io-gcs-filesystem >> s_bp.log 2>&1; echo BPDONE >> s_bp.log) &
wait; echo ALLDONE > setup.done
