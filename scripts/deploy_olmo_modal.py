"""
Deploy OLMo 3.1 32B on Modal with an OpenAI-compatible API via vLLM.

Usage:
    # Set variant first (instruct, sft, or dpo)
    export MODEL_VARIANT=instruct

    # 1. Pre-download weights (only needed once per variant)
    modal run scripts/deploy_olmo_modal.py

    # 2. Deploy
    modal deploy scripts/deploy_olmo_modal.py

    # 3. Dev server
    modal serve scripts/deploy_olmo_modal.py

After deploying, Modal prints the app URL. Use it as base_url in
configs/models/olmo_modal.yaml (see that file for details).
"""
from __future__ import annotations

import os

import modal

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
VARIANT_MAP = {
    "instruct": "allenai/OLMo-3.1-32B-Instruct",
    "sft": "allenai/OLMo-3.1-32B-Instruct-SFT",
    "dpo": "allenai/OLMo-3.1-32B-Instruct-DPO",
}
VARIANT = os.environ.get("MODEL_VARIANT", "instruct").lower()
MODEL_ID = VARIANT_MAP[VARIANT]
N_GPU = 2
VLLM_PORT = 8000
MINUTES = 60
HF_CACHE_DIR = "/root/.cache/huggingface"

app = modal.App(f"olmo-31-32b-{VARIANT}")

vllm_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.0-devel-ubuntu22.04", add_python="3.12"
    )
    .entrypoint([])
    .uv_pip_install(
        "vllm==0.13.0",
        "huggingface-hub==0.36.0",
    )
    .env({"HF_XET_HIGH_PERFORMANCE": "1", "MODEL_VARIANT": VARIANT})
)

hf_cache_vol = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
vllm_cache_vol = modal.Volume.from_name("vllm-cache", create_if_missing=True)


@app.function(
    image=vllm_image,
    volumes={HF_CACHE_DIR: hf_cache_vol},
    timeout=30 * MINUTES,
)
def download_model():
    from huggingface_hub import snapshot_download

    hf_cache_vol.reload()
    snapshot_download(MODEL_ID, cache_dir=HF_CACHE_DIR)
    hf_cache_vol.commit()


@app.function(
    image=vllm_image,
    gpu=f"A100-80GB:{N_GPU}",
    scaledown_window=20 * MINUTES,
    timeout=30 * MINUTES,
    volumes={
        HF_CACHE_DIR: hf_cache_vol,
        "/root/.cache/vllm": vllm_cache_vol,
    },
    # secrets=[modal.Secret.from_name("huggingface-secret")],
)
@modal.concurrent(max_inputs=100)
@modal.web_server(port=VLLM_PORT, startup_timeout=10 * MINUTES)
def serve():
    import subprocess

    cmd = [
        "vllm",
        "serve",
        MODEL_ID,
        "--served-model-name", MODEL_ID,
        "--host", "0.0.0.0",
        "--port", str(VLLM_PORT),
        "--tensor-parallel-size", str(N_GPU),
        "--gpu-memory-utilization", "0.90",
        "--max-model-len", "10000",
        "--trust-remote-code",
        "--enforce-eager",
        "--uvicorn-log-level=info",
    ]

    print("Starting vLLM:", *cmd)
    subprocess.Popen(" ".join(cmd), shell=True)


@app.local_entrypoint()
def main():
    """Download model weights into the cache volume (skips if already cached)."""
    print(f"Ensuring {MODEL_ID} is cached in huggingface-cache volume...")
    download_model.remote()
    print("Model cached. Server is ready to accept requests.")
