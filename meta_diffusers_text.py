import os
import tempfile
from metaflow import FlowSpec, step, Parameter, card, current, kubernetes, pypi
from metaflow.cards import Image, Markdown
from model_store import ModelStore
from config import TextToImageDiffusionConfig

from base import DIFF_USERS_IMAGE, TextToImageDiffusion, ArtifactStore
from config_base import ConfigBase
from utils import unit_convert
from gpu_profiler import gpu_profile


class TextToImages(FlowSpec, ConfigBase, ArtifactStore, TextToImageDiffusion):
    """
    Create images from prompt values using Stable Diffusion.
    """

    max_parallel = Parameter(
        "max-parallel",
        default=1,
        type=int,
        help="This parameter will limit the amount of parallelisation we wish to do. Based on the value set here, the foreach will fanout to that many workers.",
    )

    local_model_path = Parameter(
        "local-model-path",
        default=None,
        type=str,
        help="Path of local model to use instead of the one in S3",
    )

    _CORE_CONFIG_CLASS = TextToImageDiffusionConfig

    @property
    def config(self) -> TextToImageDiffusionConfig:
        return self._get_config()

    @property
    def model_version(self):
        return self.config.model_config.model_store.model_version

    def _get_model_store(self):
        return ModelStore.from_config(self.config.model_config.model_store)

    @step
    def start(self):
        import random
        import math

        store_config = self.config.model_config.model_store
        model_store = self._get_model_store()
        # Upload the model if it is not present.
        model_store.upload_model_if_none_exists(store_config)
        num_images = self.config.prompt_config.num_images
        # create seed values for each inference step
        random.seed(self.config.seed)
        chunk_size = math.ceil(num_images / self.max_parallel)
        self.rand_seeds = [
            random.randint(1, 10**7) for i in range(0, num_images, chunk_size)
        ]
        print("Creating %s tasks " % str(len(self.rand_seeds)))
        # Fanout the inference over the chunked seed values.
        self.next(self.generate_images, foreach="rand_seeds")

    @kubernetes(
        image=DIFF_USERS_IMAGE,
        gpu=1,
        cpu=4,
        memory=16000,
        disk=unit_convert(100, "GB", "MB"),
    )
    @gpu_profile(interval=1)
    @card
    @step
    def generate_images(self):
        model_store = self._get_model_store()
        prompt_config = self.config.prompt_config
        _seed = self.input
        with tempfile.TemporaryDirectory() as _dir:
            import math

            chunk_size = math.ceil(prompt_config.num_images / self.max_parallel)
            model_path = _dir
            if self.local_model_path is not None:
                model_path = self.local_model_path
            else:
                model_store.download(self.model_version, _dir)
            idx = 0
            for images, prompt in self.infer_prompt(
                model_path,
                _seed,
                prompt_config.prompts,
                chunk_size,
                self.config.inference_config,
            ):
                print("Writing Images!")
                idx += len(images)
                current.card.extend(
                    [Markdown("## Prompt : %s" % prompt)]
                    + [Image.from_pil_image(i) for i in images]
                )
                # save the images.
                for _idx, i in enumerate(images):
                    self.save_artifact("image_%d_%d" % (idx, _idx), i)

        self.next(self.join)

    @step
    def join(self, inputs):
        self.next(self.end)

    @step
    def end(self):
        print("Done!")


if __name__ == "__main__":
    TextToImages()
