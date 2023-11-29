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
    project,
    S3,
)
from metaflow.cards import Image, Markdown
from model_store import ModelStore
from config import TextToVideoDiffusionConfig
import shutil
from base import DIFF_USERS_IMAGE, ArtifactStore, SGM_BASE_IMAGE, TextToImageDiffusion
from custom_decorators import gpu_profile, pip
from config_base import ConfigBase
from utils import unit_convert


@project(name="sdvideo")
class TextToVideo(FlowSpec, ConfigBase, ArtifactStore):
    """
    Create images from prompt values using Stable Diffusion.
    """

    fully_random = Parameter(
        "fully-random",
        default=False,
        is_flag=True,
        type=bool,
        help="This parameter will make the prompt fully random. If this is set to True, then the seed value will be ignored.",
    )

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
                images, prompt = tup
                for _i, image in enumerate(images):
                    image.save(os.path.join(_dir, f"{idx}_{_i}.png"))
                with open(os.path.join(_dir, f"{idx}.txt"), "w") as f:
                    f.write(prompt)
            store = ModelStore.from_path(os.path.join(current.pathspec))
            store.upload(_dir, "images")
            return store.root

    def save_image_and_video(self, image_bytes, video_bytes):
        """
        Save the image and video to the datastore.
        """
        from metaflow import current
        import uuid

        unique_id = uuid.uuid4().hex[:7]
        store_path = os.path.join(current.pathspec, "final_render")
        with tempfile.TemporaryDirectory() as _dir:
            with open(os.path.join(_dir, "image.png"), "wb") as f:
                f.write(image_bytes)
            with open(os.path.join(_dir, "video.mp4"), "wb") as f:
                f.write(video_bytes)
            store = ModelStore.from_path(store_path)
            store.upload(_dir, unique_id)
            return os.path.join(store.root, unique_id)

    @property
    def config(self) -> TextToVideoDiffusionConfig:
        return self._get_config()

    @step
    def start(self):
        self.next(self.upload_models)

    @pip(libraries={"omegaconf": "2.3.0"})
    @card(type="blank")
    @step
    def upload_models(self):
        """
        Upload the Image and Video Models to
        the datastore if they don't exist
        """
        current.card.extend(self.config_report())
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

    @gpu_profile(artifact_prefix="image_gpu_profile")
    @kubernetes(
        image=DIFF_USERS_IMAGE,
        gpu=1,
        cpu=4,
        memory=16000,
        disk=unit_convert(100, "GB", "MB"),
    )
    @card(customize=True)
    @step
    def generate_images(self):
        model_store = self._get_image_model_store()
        prompt_config = self.config.image.prompt_config
        seed = self.config.image.seed
        if self.fully_random:
            # Derive seed from metaflow pathspec
            seed = hash(current.pathspec)
        with tempfile.TemporaryDirectory() as _dir:
            model_store.download(self.image_model_version, _dir)
            image_prompts = TextToImageDiffusion.infer_prompt(
                _dir,
                seed,
                prompt_config.prompts,
                prompt_config.num_images,
                self.config.image.inference_config,
            )
            for images, prompt in image_prompts:
                current.card.extend(
                    [Markdown("## Prompt : %s" % prompt)]
                    + [Image.from_pil_image(i) for i in images]
                )
        self.stored_images_root = self._upload_images_and_prompts_to_data_store(
            image_prompts
        )
        self.next(self.generate_video_from_images)

    @kubernetes(
        image=SGM_BASE_IMAGE,
        gpu=1,
        cpu=4,
        memory=24000,
        disk=unit_convert(100, "GB", "MB"),
    )
    @gpu_profile(artifact_prefix="video_gpu_profile")
    @card
    @step
    def generate_video_from_images(self):
        from video_diffusion import ImageToVideo

        # Based on how StabilityAI has devised the code, We will HAVETO
        # download the model to a folder called checkpoints.
        print("Downloading Video Model")
        model_store = self._get_video_model_store()
        os.makedirs("./checkpoints", exist_ok=True)
        model_store.download(self.video_model_version, "./checkpoints")
        print("Generating Videos")
        self.videos_save_path = []
        seed = self.config.video.seed
        if self.fully_random:
            # Derive seed from metaflow pathspec
            seed = hash(current.pathspec)
        with tempfile.TemporaryDirectory() as _dir:
            image_store = ModelStore(self.stored_images_root)
            image_store.download("images", _dir)
            image_paths = [
                os.path.join(_dir, f) for f in os.listdir(_dir) if ".png" in f
            ]
            _args = [
                self.video_model_version,
                image_paths,
                self.config.video.inference_config,
                seed,
            ]
            for image_bytes, video_bytes in ImageToVideo.generate(*_args):
                print("Saving Video")
                save_path = self.save_image_and_video(image_bytes, video_bytes)
                print("Video Saved To Path %s" % save_path)
                self.videos_save_path.append(save_path)

        shutil.rmtree("./checkpoints")
        self.next(self.end)

    @step
    def end(self):
        print("Done!")


if __name__ == "__main__":
    TextToVideo()
