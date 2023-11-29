import math
import copy
import os
from glob import glob
from pathlib import Path
from typing import Optional

from typing import List
import cv2
import numpy as np
import torch
from einops import rearrange, repeat
from fire import Fire
from omegaconf import OmegaConf
from torch import autocast
from PIL import Image
from torchvision.transforms import ToTensor, functional as TF
from torch import Tensor

CURRENT_FILE_DIR = os.path.dirname(os.path.abspath(__file__))

from .sgm_util import default, instantiate_from_config


def unload_model(model):
    model.cpu()
    torch.cuda.empty_cache()


def load_model_fully(model_name, num_frames, num_steps, device):
    if model_name == "stable-video-diffusion-img2vid":
        model_config = os.path.join(CURRENT_FILE_DIR, "configs", "svd.yaml")
        num_frames = default(num_frames, 14)
    elif model_name == "stable-video-diffusion-img2vid-xt":
        model_config = os.path.join(CURRENT_FILE_DIR, "configs", "svd_xt.yaml")
        num_frames = default(num_frames, 25)
    else:
        supported_version = ", ".join(
            ["stable-video-diffusion-img2vid", "stable-video-diffusion-img2vid-xt"]
        )
        raise ValueError(
            f"Version {model_name} does not exist. Supported versions: {supported_version}"
        )

    num_steps = default(num_steps, 30)
    model = load_model(
        model_config,
        device,
        num_frames,
        num_steps,
    )
    return model


def _image_to_tensor(input_img_path, resize=True):
    with Image.open(input_img_path) as image:
        if image.mode == "RGBA":
            image = image.convert("RGB")

        image = ToTensor()(image)
        if resize:
            print(f"Resizing {image.size()} to (1024, 576)")
            image = TF.resize(TF.resize(image, 1024), (576, 1024))

        _, w, h = image.size()

        if h % 64 != 0 or w % 64 != 0:
            width, height = map(lambda x: x - x % 64, (w, h))
            image = image.resize((width, height))
            print(
                f"WARNING: Your image is of size {h}x{w} which is not divisible by 64. We are resizing to {height}x{width}!"
            )

        image = image * 2.0 - 1.0
        return image


def _inference_validation_and_warnings(
    image: Tensor,
    motion_bucket_id,
    fps_id,
):
    assert image.shape[1] == 3
    H, W = image.shape[2:]
    if (H, W) != (576, 1024):
        print(
            "WARNING: The conditioning frame you provided is not 576x1024. This leads to suboptimal performance as model was only trained on 576x1024. Consider increasing `cond_aug`."
        )
    if motion_bucket_id > 255:
        print("WARNING: High motion bucket! This may lead to suboptimal performance.")

    if fps_id < 5:
        print("WARNING: Small fps value! This may lead to suboptimal performance.")

    if fps_id > 30:
        print("WARNING: Large fps value! This may lead to suboptimal performance.")


def _get_image_tensor_and_model_inputs(
    input_img_path: str,
    motion_bucket_id: int,
    fps_id: int,
    cond_aug: float,
    num_frames: int,
    device: str,
    resize=False,
):
    image = _image_to_tensor(input_img_path, resize=resize)
    image = image.unsqueeze(0).to(device)
    _inference_validation_and_warnings(image, motion_bucket_id, fps_id)

    value_dict = {}
    value_dict["motion_bucket_id"] = motion_bucket_id
    value_dict["fps_id"] = fps_id
    value_dict["cond_aug"] = cond_aug
    value_dict["cond_frames_without_noise"] = image
    value_dict["cond_frames"] = image + cond_aug * torch.randn_like(image)
    return image, value_dict


def _get_shape(image: Tensor, num_frames):
    H, W = image.shape[2:]
    F = 8
    C = 4
    return (num_frames, C, H // F, W // F)


def _get_dimensions(image: Tensor, num_frames):
    H, W = image.shape[2:]
    F = 8
    C = 4
    T = num_frames
    return H, W, C, F, T


def _write_video_to_folder(vid: Tensor, save_path, frame_rate):
    base_count = len(glob(os.path.join(save_path, "*.mp4")))
    video_path = os.path.join(save_path, f"{base_count:06d}.mp4")

    writer = cv2.VideoWriter(
        video_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        frame_rate,
        (vid.shape[-1], vid.shape[-2]),
    )

    vid = (rearrange(vid, "t c h w -> t h w c") * 255).cpu().numpy().astype(np.uint8)
    for frame in vid:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        writer.write(frame)

    writer.release()
    return video_path


# Plucked Direcly from StabilityAI/generative-models[commit:ed09971] : /scripts/demo/streamlit_helpers.py
def do_sample(
    model,
    sampler,
    value_dict,
    num_samples,
    H,
    W,
    C,
    F,
    force_uc_zero_embeddings: Optional[List] = None,
    force_cond_zero_embeddings: Optional[List] = None,
    batch2model_input: List = None,
    return_latents=False,
    filter=None,
    T=None,
    additional_batch_uc_fields=None,
    decoding_t=None,
):
    from sgm.modules.diffusionmodules.guiders import LinearPredictionGuider, VanillaCFG

    force_uc_zero_embeddings = default(force_uc_zero_embeddings, [])
    batch2model_input = default(batch2model_input, [])
    additional_batch_uc_fields = default(additional_batch_uc_fields, [])

    precision_scope = autocast
    with torch.no_grad():
        with precision_scope("cuda"):
            with model.ema_scope():
                if T is not None:
                    num_samples = [num_samples, T]
                else:
                    num_samples = [num_samples]

                _load_model(model.conditioner)
                batch, batch_uc = _get_batch(
                    get_unique_embedder_keys_from_conditioner(model.conditioner),
                    value_dict,
                    num_samples,
                    T=T,
                    additional_batch_uc_fields=additional_batch_uc_fields,
                )

                c, uc = model.conditioner.get_unconditional_conditioning(
                    batch,
                    batch_uc=batch_uc,
                    force_uc_zero_embeddings=force_uc_zero_embeddings,
                    force_cond_zero_embeddings=force_cond_zero_embeddings,
                )
                unload_model(model.conditioner)

                for k in c:
                    if not k == "crossattn":
                        c[k], uc[k] = map(
                            lambda y: y[k][: math.prod(num_samples)].to("cuda"), (c, uc)
                        )
                    if k in ["crossattn", "concat"] and T is not None:
                        uc[k] = repeat(uc[k], "b ... -> b t ...", t=T)
                        uc[k] = rearrange(uc[k], "b t ... -> (b t) ...", t=T)
                        c[k] = repeat(c[k], "b ... -> b t ...", t=T)
                        c[k] = rearrange(c[k], "b t ... -> (b t) ...", t=T)

                additional_model_inputs = {}
                for k in batch2model_input:
                    if k == "image_only_indicator":
                        assert T is not None

                        if isinstance(
                            sampler.guider, (VanillaCFG, LinearPredictionGuider)
                        ):
                            additional_model_inputs[k] = torch.zeros(
                                num_samples[0] * 2, num_samples[1]
                            ).to("cuda")
                        else:
                            additional_model_inputs[k] = torch.zeros(num_samples).to(
                                "cuda"
                            )
                    else:
                        additional_model_inputs[k] = batch[k]

                shape = (math.prod(num_samples), C, H // F, W // F)
                randn = torch.randn(shape).to("cuda")

                def denoiser(input, sigma, c):
                    return model.denoiser(
                        model.model, input, sigma, c, **additional_model_inputs
                    )

                _load_model(model.denoiser)
                _load_model(model.model)
                samples_z = sampler(denoiser, randn, cond=c, uc=uc)
                unload_model(model.model)
                unload_model(model.denoiser)

                _load_model(model.first_stage_model)
                model.en_and_decode_n_samples_a_time = (
                    decoding_t  # Decode n frames at a time
                )
                samples_x = model.decode_first_stage(samples_z)
                samples = torch.clamp((samples_x + 1.0) / 2.0, min=0.0, max=1.0)
                unload_model(model.first_stage_model)

                if filter is not None:
                    samples = filter(samples)

                if return_latents:
                    return samples, samples_z
                return samples


def _load_model(model):
    model.cuda()


def _low_vram_mode(model):
    # move models expect unet to cpu
    model.conditioner.cpu()
    model.first_stage_model.cpu()
    # change the dtype of unet
    model.model.to(dtype=torch.float16)
    torch.cuda.empty_cache()
    model = model.requires_grad_(False)


def sample(
    input_paths: List[str] = [],
    num_frames: Optional[int] = None,
    num_steps: Optional[int] = None,
    version: str = "svd",
    fps_id: int = 6,
    motion_bucket_id: int = 127,
    cond_aug: float = 0.02,
    seed: int = 23,
    decoding_t: int = 14,  # Number of frames decoded at a time! This eats most VRAM. Reduce if necessary.
    device: str = "cuda",
    low_vram_mode: bool = False,
    output_folder: Optional[str] = None,
    resize=True,
):
    """
    Simple script to generate a single sample conditioned on an image `input_path` or multiple images, one for each
    image file in folder `input_path`. If you run out of VRAM, try decreasing `decoding_t`.
    """
    print("Loading Model....", version)
    model = load_model_fully(version, num_frames, num_steps, device)
    if low_vram_mode:
        print("setting low vram mode")
        _low_vram_mode(model)
    if torch.cuda.is_available():
        print("using cuda")
        model.cuda()
    torch.manual_seed(seed)
    all_img_paths = input_paths
    video_paths = []
    for input_img_path in all_img_paths:
        image, value_dict = _get_image_tensor_and_model_inputs(
            input_img_path,
            motion_bucket_id,
            fps_id,
            cond_aug,
            num_frames,
            device,
            resize=resize,
        )
        H, W, C, F, T = _get_dimensions(image, num_frames)
        force_uc_zero_embeddings = ["cond_frames", "cond_frames_without_noise"]
        samples = do_sample(
            model,
            model.sampler,
            value_dict,
            1,
            H,
            W,
            C,
            F,
            T=T,
            batch2model_input=["num_video_frames", "image_only_indicator"],
            force_uc_zero_embeddings=force_uc_zero_embeddings,
            force_cond_zero_embeddings=None,
            return_latents=False,
            decoding_t=decoding_t,
        )
        video_path = _write_video_to_folder(samples, output_folder, fps_id)
        video_paths.append(video_path)
    return video_paths


def get_unique_embedder_keys_from_conditioner(conditioner):
    return list(set([x.input_key for x in conditioner.embedders]))


def _get_batch(
    keys,
    value_dict: dict,
    N,
    device: str = "cuda",
    T: int = None,
    additional_batch_uc_fields: List[str] = [],
):
    # Hardcoded demo setups; might undergo some changes in the future

    batch = {}
    batch_uc = {}

    for key in keys:
        if key == "txt":
            batch["txt"] = [value_dict["prompt"]] * math.prod(N)

            batch_uc["txt"] = [value_dict["negative_prompt"]] * math.prod(N)

        elif key == "original_size_as_tuple":
            batch["original_size_as_tuple"] = (
                torch.tensor([value_dict["orig_height"], value_dict["orig_width"]])
                .to(device)
                .repeat(math.prod(N), 1)
            )
        elif key == "crop_coords_top_left":
            batch["crop_coords_top_left"] = (
                torch.tensor(
                    [value_dict["crop_coords_top"], value_dict["crop_coords_left"]]
                )
                .to(device)
                .repeat(math.prod(N), 1)
            )
        elif key == "aesthetic_score":
            batch["aesthetic_score"] = (
                torch.tensor([value_dict["aesthetic_score"]])
                .to(device)
                .repeat(math.prod(N), 1)
            )
            batch_uc["aesthetic_score"] = (
                torch.tensor([value_dict["negative_aesthetic_score"]])
                .to(device)
                .repeat(math.prod(N), 1)
            )

        elif key == "target_size_as_tuple":
            batch["target_size_as_tuple"] = (
                torch.tensor([value_dict["target_height"], value_dict["target_width"]])
                .to(device)
                .repeat(math.prod(N), 1)
            )
        elif key == "fps":
            batch[key] = (
                torch.tensor([value_dict["fps"]]).to(device).repeat(math.prod(N))
            )
        elif key == "fps_id":
            batch[key] = (
                torch.tensor([value_dict["fps_id"]]).to(device).repeat(math.prod(N))
            )
        elif key == "motion_bucket_id":
            batch[key] = (
                torch.tensor([value_dict["motion_bucket_id"]])
                .to(device)
                .repeat(math.prod(N))
            )
        elif key == "pool_image":
            batch[key] = repeat(value_dict[key], "1 ... -> b ...", b=math.prod(N)).to(
                device, dtype=torch.half
            )
        elif key == "cond_aug":
            batch[key] = repeat(
                torch.tensor([value_dict["cond_aug"]]).to("cuda"),
                "1 -> b",
                b=math.prod(N),
            )
        elif key == "cond_frames":
            batch[key] = repeat(value_dict["cond_frames"], "1 ... -> b ...", b=N[0])
        elif key == "cond_frames_without_noise":
            batch[key] = repeat(
                value_dict["cond_frames_without_noise"], "1 ... -> b ...", b=N[0]
            )
        else:
            batch[key] = value_dict[key]

    if T is not None:
        batch["num_video_frames"] = T

    for key in batch.keys():
        if key not in batch_uc and isinstance(batch[key], torch.Tensor):
            batch_uc[key] = torch.clone(batch[key])
        elif key in additional_batch_uc_fields and key not in batch_uc:
            batch_uc[key] = copy.copy(batch[key])
    return batch, batch_uc


def get_batch(keys, value_dict, N, T, device):
    batch = {}
    batch_uc = {}

    for key in keys:
        if key == "fps_id":
            batch[key] = (
                torch.tensor([value_dict["fps_id"]])
                .to(device)
                .repeat(int(math.prod(N)))
            )
        elif key == "motion_bucket_id":
            batch[key] = (
                torch.tensor([value_dict["motion_bucket_id"]])
                .to(device)
                .repeat(int(math.prod(N)))
            )
        elif key == "cond_aug":
            batch[key] = repeat(
                torch.tensor([value_dict["cond_aug"]]).to(device),
                "1 -> b",
                b=math.prod(N),
            )
        elif key == "cond_frames":
            batch[key] = repeat(value_dict["cond_frames"], "1 ... -> b ...", b=N[0])
        elif key == "cond_frames_without_noise":
            batch[key] = repeat(
                value_dict["cond_frames_without_noise"], "1 ... -> b ...", b=N[0]
            )
        else:
            batch[key] = value_dict[key]

    if T is not None:
        batch["num_video_frames"] = T

    for key in batch.keys():
        if key not in batch_uc and isinstance(batch[key], torch.Tensor):
            batch_uc[key] = torch.clone(batch[key])
    return batch, batch_uc


def load_model(
    config: str,
    device: str,
    num_frames: int,
    num_steps: int,
):
    config = OmegaConf.load(config)
    if device == "cuda":
        config.model.params.conditioner_config.params.emb_models[
            0
        ].params.open_clip_embedding_config.params.init_device = device

    config.model.params.sampler_config.params.num_steps = num_steps
    config.model.params.sampler_config.params.guider_config.params.num_frames = (
        num_frames
    )
    model = instantiate_from_config(config.model)
    return model


if __name__ == "__main__":
    Fire(sample)
