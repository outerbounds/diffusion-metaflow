import math
from torch import autocast
import torch
from diffusers import StableDiffusionPipeline


def generate_images(
    model,
    prompt,
    batch_size,
    generator=None,
    use_gpu=True,
    height=512,
    width=768,
    num_steps=52,
):
    _prompt = [prompt] * batch_size
    if use_gpu:
        with autocast("cuda"):
            return model(
                _prompt,
                height=height,
                width=width,
                generator=generator,
                num_inference_steps=num_steps,
            )["sample"]
    else:
        return model(
            _prompt,
            height=height,
            width=width,
            generator=generator,
            num_inference_steps=num_steps,
        )["sample"]


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
    use_gpu=True,
):
    pipe = StableDiffusionPipeline.from_pretrained(
        model_path,
        revision="fp16",
        torch_dtype=torch.float16,
    )
    generator = torch.manual_seed(seed)
    if use_gpu:
        generator = torch.cuda.manual_seed(seed)
        pipe = pipe.to("cuda")

    all_images = []
    for _batch_size in _create_batchsizes(num_images, batch_size):
        all_images.extend(
            generate_images(
                pipe,
                prompt,
                _batch_size,
                use_gpu=use_gpu,
                height=height,
                width=width,
                generator=generator,
                num_steps=num_steps,
            )
        )
    return all_images
