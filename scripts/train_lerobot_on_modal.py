import modal
import modal.gpu

app = modal.App("train-lerobot")

flash_attn_wheel = (
    "https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/"
    "flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp313-cp313-linux_x86_64.whl"
)

image = (
    modal.Image.debian_slim(python_version="3.13")
    .pip_install(
        "torch==2.6.0",
        "numpy==2.2.4",
        "transformers",
        "accelerate",
        "lerobot==0.4.4",
        "peft"
    )
    # Install flash-attn without building from source
    .pip_install(flash_attn_wheel)
    .pip_install("lerobot[groot]==0.4.4")
    #.uv_pip_install("flash-attn", extra_options="--no-build-isolation", gpu=modal.gpu.A10G(count=1))
)

@app.function(image=image, secrets=[modal.Secret.from_name("hf-token-secret")], timeout=24*60*60, gpu=modal.gpu.A100(count=1), memory=80000, cpu=1.0)
def train():
    import subprocess
    subprocess.run(["lerobot-train", "--dataset.video_backend=pyav", "--policy.type=groot", "--dataset.repo_id=crislmfroes/xarm6-sim-v4", "--policy.repo_id=crislmfroes/groot-xarm6-sim-v4", "--wandb.enable=false", "--steps=20000", "--batch_size=8"])

@app.local_entrypoint()
def main():
    pass