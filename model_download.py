# make sure you have signed the waiver on hugging face hub before downloading the model.
import os
from base import MODEL_PATH, MODEL_VERSION
import torch
from diffusers import AutoPipelineForText2Image
pipe = AutoPipelineForText2Image.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16, variant="fp16", use_safetensors=True,
    use_auth_token=os.environ["HF_TOKEN"],
)
pipe.save_pretrained(MODEL_PATH)
