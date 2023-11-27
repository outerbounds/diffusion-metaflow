import os
import tempfile
from metaflow import FlowSpec, step, Parameter, batch, card, current, kubernetes, pypi
from metaflow.cards import Image, Markdown
from model_store import ModelStore
from config import TextToVideoDiffusionConfig

from base import DIFF_USERS_IMAGE, ArtifactStore
from config_base import ConfigBase
from utils import unit_convert


class TextToVideo(FlowSpec, ConfigBase, ArtifactStore):
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

    _CORE_CONFIG_CLASS = TextToVideoDiffusionConfig

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
        self.next(self.generate_images)

    @step
    def generate_images(self):
        self.next(self.generate_video_from_images)

    @step
    def generate_video_from_images(self):
        self.next(self.end)

    @step
    def end(self):
        print("Done!")


if __name__ == "__main__":
    TextToVideo()
