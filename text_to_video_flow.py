import os
import tempfile
from metaflow import (
    FlowSpec,
    step,
    Parameter,
    batch,
    card,
    current,
    kubernetes,
    pypi,
    S3,
)
from metaflow.cards import Image, Markdown
from model_store import ModelStore
from config import TextToVideoDiffusionConfig

from base import DIFF_USERS_IMAGE, ArtifactStore, SGM_BASE_IMAGE, TextToImageDiffusion
from config_base import ConfigBase
from utils import unit_convert


class TextToVideo(FlowSpec, ConfigBase, ArtifactStore):
    """
    Create images from prompt values using Stable Diffusion.
    """

    # max_parallel = Parameter(
    #     "max-parallel",
    #     default=1,
    #     type=int,
    #     help="This parameter will limit the amount of parallelisation we wish to do. Based on the value set here, the foreach will fanout to that many workers.",
    # )

    _CORE_CONFIG_CLASS = TextToVideoDiffusionConfig

    def _get_image_model_store(self):
        return ModelStore.from_config(self.config.image.model_config.model_store)

    def _get_video_model_store(self):
        return ModelStore.from_config(self.config.video.model_config.model_store)

    def _upload_images_and_prompts_to_data_store(self, image_prompts):
        # Images have been uploaded to the /<pathspec>/images folder
        from metaflow import current

        with tempfile.TemporaryDirectory() as _dir:
            for idx, tup in enumerate(image_prompts):
                prompt, image = tup
                image.save(os.path.join(_dir, f"{idx}.png"))
                with open(os.path.join(_dir, f"{idx}.txt"), "w") as f:
                    f.write(prompt)
            ModelStore.from_path(os.path.join(current.pathspec)).upload(_dir, "images")

    @property
    def config(self) -> TextToVideoDiffusionConfig:
        return self._get_config()

    @step
    def start(self):
        self.next(self.upload_models)

    @step
    def upload_models(self):
        """
        Upload the Image and Video Models to
        the datastore if they don't exist
        """
        store_configs = [
            self.config.image.model_config.model_store,
            self.config.video.model_config.model_store,
        ]
        for store_config in store_configs:
            model_store = ModelStore.from_config(store_config)
            # Upload the model if it is not present.
            model_store.upload_model_if_none_exists(store_config)
        self.image_model_version = (
            self.config.image.model_config.model_store.model_version
        )
        self.video_model_version = (
            self.config.video.model_config.model_store.model_version
        )
        self.next(self.generate_images)

    @kubernetes(
        image=DIFF_USERS_IMAGE,
        gpu=1,
        cpu=4,
        memory=16000,
        disk=unit_convert(100, "GB", "MB"),
    )
    @card
    @step
    def generate_images(self):
        model_store = self._get_image_model_store()
        prompt_config = self.config.image.prompt_config
        seed = self.config.image.seed
        with tempfile.TemporaryDirectory() as _dir:
            model_store.download(self.image_model_version, _dir)
            image_prompts = TextToImageDiffusion.infer_prompt(
                _dir,
                seed,
                prompt_config.prompts,
                prompt_config.num_images,
                self.config.image.inference_config,
            )
        self._upload_images_and_prompts_to_data_store(image_prompts)
        self.next(self.generate_video_from_images)

    @step
    def generate_video_from_images(self):
        # Based on how StabilityAI has devised the code, We will HAVETO
        # download the model to a folder called checkpoints.
        self.next(self.end)

    @step
    def end(self):
        print("Done!")


if __name__ == "__main__":
    TextToVideo()
