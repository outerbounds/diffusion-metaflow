import os

VIDEO_MODEL_ORG = "stabilityai"
VIDEO_MODEL_NAME = "stable-video-diffusion-img2vid"
VIDEO_MODEL_PATH = "./video-models"


def download_model(model_path=VIDEO_MODEL_PATH):
    from huggingface_hub import hf_hub_download

    hf_hub_download(
        repo_id=f"{VIDEO_MODEL_ORG}/{VIDEO_MODEL_NAME}",
        filename="svd.safetensors",
        local_dir=model_path,
        local_dir_use_symlinks=False,
    )
