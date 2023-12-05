from typing import Union, Optional, List, Dict, Any
from dataclasses import dataclass, field
from enum import Enum
from collections import namedtuple


VIDEO_MODEL_NAME = "stable-video-diffusion-img2vid"

MODELS_BASE_S3_PATH = "models/diffusion-models/"
IMAGE_MODEL_NAME = "stable-diffusion-xl-base-1.0.1"
MODEL_PATH = "./models"
DEFAULT_STYLES = [
    "van gogh",
    "salvador dali",
    "warhol",
    "Art Nouveau",
    "Ansel Adams",
    "basquiat",
]
DEFAULT_SUBJECTS = [
    "mahatma gandhi",
    "dalai lama",
    "alan turing",
]

DEFAULT_PROMPTS = [
    "Astronaut in on the amazon jungle petting a tiger, 8k very detailed and realistic, cinematic lighting, highly detailed, digital painting, artstation, concept art, smooth, sharp focus, illustration",
    "spungebob squarepants building machine learning models at NASA, 8k very detailed and realistic, cinematic lighting, highly detailed, digital painting, artstation, concept art, smooth, sharp focus, illustration, animated",
]

DEFAULT_PROMPTS_CROSS_PRODUCT = {}


@dataclass
class ModelStoreConfig:
    pretrained_model_path: str = (
        MODEL_PATH  # Path to the downloaded model on the local machine.
    )
    force_upload: bool = False  # Force upload the model from the local machine
    s3_prefix: str = (
        MODELS_BASE_S3_PATH  # prefix of the path where models are stored in S3.
    )
    model_version: str = IMAGE_MODEL_NAME  # A unique version for the model. It will be used to create a folder in the S3 bucket.


@dataclass
class VideoModelConfig:
    model_store: ModelStoreConfig = field(default_factory=ModelStoreConfig)
    model_name: str = VIDEO_MODEL_NAME


@dataclass
class ImageModelConfig:
    model_store: ModelStoreConfig = field(default_factory=ModelStoreConfig)
    model_name: str = IMAGE_MODEL_NAME


@dataclass
class ImageInferenceConfig:
    # Informs the inference engine about the properties
    # of the output image and how mechanics of the
    # generation (like batchsize)
    batch_size: int = 1
    height: int = 512
    width: int = 512
    num_steps: int = 50


@dataclass
class MotionBucket:
    bucket_id: Optional[int] = 127
    sample_multiple_buckets: bool = False
    num_motion_buckets_to_sample: Optional[int] = 20
    motion_bucket_range: List[int] = field(default_factory=lambda: [0, 255])


@dataclass
class VideoGenerationConfig:
    # Frame-rate of the generated video
    frame_rate: int = 10
    # Number of frames decoded at a time! This eats most VRAM. Reduce if necessary.
    decoding_timesteps: int = 14
    # number of steps to run the model for
    num_steps: int = 50
    # number of frames to generate
    num_frames: int = 50
    # low vram mode / Runs with half the model weights
    low_vram_mode: bool = False
    # resize the image to the model's input size
    resize: bool = True
    # Motion bucket related configurations. These affect how many videos will be
    # created based on motion buckets.
    motion_bucket: MotionBucket = field(default_factory=MotionBucket)


@dataclass
class PromptConfig:
    prompts: List[str] = field(default_factory=lambda: DEFAULT_PROMPTS)
    num_images: int = 10  # Number of images to create per prompt


@dataclass
class ImageStylePromptConfig:
    styles: List[str] = field(default_factory=lambda: DEFAULT_STYLES)
    subjects: List[str] = field(default_factory=lambda: DEFAULT_SUBJECTS)
    num_images: int = 10  # Number of images to create per (subject, style)
    locations: List[str] = field(default_factory=lambda: [])
    actions: List[str] = field(default_factory=lambda: [])


@dataclass
class TextToImageDiffusionConfig:
    model_config: ImageModelConfig = field(default_factory=ImageModelConfig)
    inference_config: ImageInferenceConfig = field(default_factory=ImageInferenceConfig)
    prompt_config: PromptConfig = field(default_factory=PromptConfig)
    seed: int = 42


@dataclass
class ImageStylePromptDiffusionConfig:
    model_config: ImageModelConfig = field(default_factory=ImageModelConfig)
    inference_config: ImageInferenceConfig = field(default_factory=ImageInferenceConfig)
    style_prompt_config: ImageStylePromptConfig = field(
        default_factory=ImageStylePromptConfig
    )
    seed: int = 42


@dataclass
class ImageToVideoDiffusionConfig:
    model_config: VideoModelConfig = field(default_factory=VideoModelConfig)
    inference_config: VideoGenerationConfig = field(
        default_factory=VideoGenerationConfig
    )
    seed: int = 42


@dataclass
class TextToVideoDiffusionConfig:
    # NO seed set in this objects as the seed is set in the
    # sub objects
    video: ImageToVideoDiffusionConfig = field(
        default_factory=ImageToVideoDiffusionConfig
    )
    image: TextToImageDiffusionConfig = field(
        default_factory=TextToImageDiffusionConfig
    )


@dataclass
class TextToVideoWithStylesDiffusionConfig:
    # NO seed set in this objects as the seed is set in the
    # sub objects
    video: ImageToVideoDiffusionConfig = field(
        default_factory=ImageToVideoDiffusionConfig
    )
    image: ImageStylePromptDiffusionConfig = field(
        default_factory=ImageStylePromptDiffusionConfig
    )


def create_config(filepath, _class):
    from omegaconf import OmegaConf

    conf = OmegaConf.structured(_class)
    OmegaConf.save(conf, filepath)


def load_config(filepath, _class):
    from omegaconf import OmegaConf

    conf = OmegaConf.load(filepath)
    schema = OmegaConf.structured(_class)
    trainconf = OmegaConf.merge(schema, conf)
    return trainconf


StyleVideoOutput = namedtuple(
    "StyleVideoOutput",
    ["subject", "style", "action", "location", "image_paths", "video_paths"],
)


def get_style_video_namedtuple(
    subject: str,
    style: str,
    action: str,
    location: str,
    image_paths: List[str],
    video_paths: Dict[str, str],  # {motion_bucket_id: video_path}
) -> StyleVideoOutput:
    return StyleVideoOutput(subject, style, action, location, image_paths, video_paths)


SUPPORTED_CONFIGS = {
    "text": TextToImageDiffusionConfig,
    "style": ImageStylePromptDiffusionConfig,
    "video": TextToVideoDiffusionConfig,
    "stylevideo": TextToVideoWithStylesDiffusionConfig,
}


if __name__ == "__main__":
    import sys

    assert (
        len(sys.argv) == 3
    ), "usage : `python config.py <type> example.yaml`. Allowed <type> : %s" % (
        ", ".join(SUPPORTED_CONFIGS.keys())
    )
    if sys.argv[1] not in SUPPORTED_CONFIGS:
        raise ValueError(
            "Invalid argument. Must be one of %s"
            % (", ".join(SUPPORTED_CONFIGS.keys()))
        )
    create_config(sys.argv[2], SUPPORTED_CONFIGS[sys.argv[1]])
