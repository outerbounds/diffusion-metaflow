import math
from torch import autocast
import torch
import os
from diffusers import AutoPipelineForText2Image

SD_XL_BASE = "stable-diffusion-xl-base-1.0"
SUPPORTED_PIPELINES = ["StableDiffusionXLPipeline", "StableDiffusionPipeline"]

IMAGE_MODEL_NAME = SD_XL_BASE
IMAGE_MODEL_ORG = "stabilityai"
IMAGE_MODEL_PATH = "./models"


def download_model(model_path=IMAGE_MODEL_PATH):
    image_pipe = AutoPipelineForText2Image.from_pretrained(
        f"{IMAGE_MODEL_ORG}/{IMAGE_MODEL_NAME}",
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True,
        use_auth_token=os.environ["HF_TOKEN"],
    )
    image_pipe.save_pretrained(model_path)


def _is_pipeline_supported(pipe):
    return pipe.__class__.__name__ in SUPPORTED_PIPELINES


def _retrieve_images(pipe, pipe_output):
    if pipe.__class__.__name__ == "StableDiffusionXLPipeline":
        return pipe_output.images
    elif pipe.__class__.__name__ == "StableDiffusionPipeline":
        return pipe_output["sample"]
    else:
        raise ValueError(
            "Unsupported pipeline, please use one of the following pipelines: %s"
            % ", ".join(SUPPORTED_PIPELINES)
        )


def generate_images(
    model,
    prompt,
    batch_size,
    generator=None,
    height=512,
    width=768,
    num_steps=52,
):
    _prompt = [prompt] * batch_size
    output = model(
        _prompt,
        height=height,
        width=width,
        generator=generator,
        num_inference_steps=num_steps,
    )
    return _retrieve_images(model, output)


def _create_batchsizes(num_images, batch_size):
    bs = []
    num_loops = math.ceil(num_images / batch_size)
    for i in range(1, num_loops + 1, 1):
        if (num_images - i * batch_size) >= 0:
            bs.append(batch_size)
        else:
            bs.append((num_images - (i - 1) * batch_size))
    return bs


def infer_prompt(
    model_path,
    prompt,
    num_images=3,
    batch_size=3,
    width=512,
    height=512,
    num_steps=51,
    seed=420,
):
    pipe = AutoPipelineForText2Image.from_pretrained(
        model_path,
        variant="fp16",
        torch_dtype=torch.float16,
        use_safetensors=True,
    )
    if not _is_pipeline_supported(pipe):
        raise ValueError(
            "Unsupported pipeline, please use one of the following pipelines: %s"
            % ", ".join(SUPPORTED_PIPELINES)
        )
    print("Using pipeline", pipe)
    generator = torch.cuda.manual_seed(seed)
    pipe = pipe.to("cuda")

    all_images = []
    for _batch_size in _create_batchsizes(num_images, batch_size):
        all_images.extend(
            generate_images(
                pipe,
                prompt,
                _batch_size,
                height=height,
                width=width,
                generator=generator,
                num_steps=num_steps,
            )
        )
        print("Finished batch of images")
    return all_images
