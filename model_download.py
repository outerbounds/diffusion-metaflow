# make sure you have signed the waiver on hugging face hub before downloading the model.
import os
import diffusion
import video_diffusion
import torch
from diffusers import AutoPipelineForText2Image

video_diffusion.download_model(video_diffusion.VIDEO_MODEL_PATH)
diffusion.download_model(diffusion.IMAGE_MODEL_PATH)
