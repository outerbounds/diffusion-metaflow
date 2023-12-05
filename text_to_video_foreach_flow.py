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
from config import TextToVideoWithStylesDiffusionConfig
import shutil
from base import DIFF_USERS_IMAGE, ArtifactStore, SGM_BASE_IMAGE, TextToImageDiffusion
from custom_decorators import gpu_profile, pip
from config_base import ConfigBase
from utils import unit_convert


@project(name="sdvideo")
class TextToVideoForeach(FlowSpec, ConfigBase, ArtifactStore):
    """
    Create images from prompt values using Stable Diffusion.
    """

    fully_random = Parameter(
        "fully-random",
        default=False,
        type=bool,
        help="This parameter will make the prompt fully random. If this is set to True, then the seed value will be ignored.",
    )

    _CORE_CONFIG_CLASS = TextToVideoWithStylesDiffusionConfig

    def _get_image_model_store(self):
        return ModelStore.from_config(self.config.image.model_config.model_store)

    def _get_video_model_store(self):
        return ModelStore.from_config(self.config.video.model_config.model_store)

    def _upload_images_and_prompts_to_data_store(self, image_prompts):
        # Images have been uploaded to the /<pathspec>/images folder
        from metaflow import current

        _img_files = []
        with tempfile.TemporaryDirectory() as _dir:
            # Write each image/Prompt to the Disk.
            for idx, tup in enumerate(image_prompts):
                _prompt_image_set = []
                images, prompt = tup
                # Since each prompt has multiple images
                # we will create a _prompt_image_set which holds
                # all the images for that prompt. _prompt_image_set
                # will only hold the filenames.
                for _i, image in enumerate(images):
                    img_fname = f"{idx}_{_i}.png"
                    image.save(os.path.join(_dir, img_fname))
                    _prompt_image_set.append(img_fname)

                # Once _prompt_image_set is constructed add it to a list.
                # Each value in this new list will have all images for a prompt
                _img_files.append(_prompt_image_set)
                with open(os.path.join(_dir, f"{idx}.txt"), "w") as f:
                    f.write(prompt)
            # store The images to a path in S3. This path is derived based on the pathspec
            store = ModelStore.from_path(os.path.join(current.pathspec))
            store.upload(_dir, "images")
            # For All the image file names in the _img_files list, expand the path to a Full S3 URL.
            image_pths_final = [
                [os.path.join(store.root, "images", _imgpt) for _imgpt in i]
                for i in _img_files
            ]

            return store.root, image_pths_final

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
    def config(self) -> TextToVideoWithStylesDiffusionConfig:
        return self._get_config()

    @staticmethod
    def _make_prompt(subject, action, location, style):
        return f"{subject} {action} {location} {style}"

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

        self.subject_fanout = self.config.image.style_prompt_config.subjects
        self.next(self.generate_images, foreach="subject_fanout")

    def _subject_crossproduct(self, subject):
        import itertools

        prompt_config = self.config.image.style_prompt_config
        return list(
            itertools.product(
                [subject],
                prompt_config.actions,
                prompt_config.locations,
                prompt_config.styles,
            )
        )

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
        self.subject = self.input
        # Run itertools crossproduct to create a prompt_templates of Subject, Style, action, Location,
        prompt_combos = self._subject_crossproduct(self.subject)
        # Make the actual prompt from the prompt_templates
        prompts = [self._make_prompt(*pcombo) for pcombo in prompt_combos]
        # Create a store to hold all the images generated in this step
        model_store = self._get_image_model_store()
        # set a seed based on the parameter
        seed = self.config.image.seed
        if self.fully_random:
            # Derive seed from metaflow pathspec
            seed = hash(current.pathspec)

        prompt_config = self.config.image.style_prompt_config
        with tempfile.TemporaryDirectory() as _dir:
            model_store.download(self.image_model_version, _dir)
            image_prompts = TextToImageDiffusion.infer_prompt(
                _dir,
                seed,
                prompts,
                prompt_config.num_images,
                self.config.image.inference_config,
            )
            # `images`: List[PIL.Image]
            for images, prompt in image_prompts:
                current.card.extend(
                    [Markdown("## Prompt : %s" % prompt)]
                    + [Image.from_pil_image(i) for i in images]
                )
        # Save all the images generated to a directory in S3 which uses the pathspec as a part of the path
        (
            self.stored_images_root,
            image_pths_final,
        ) = self._upload_images_and_prompts_to_data_store(image_prompts)
        # `self.prompt_results` will be a List[Tuple[str, str,str, str, str ]]
        # where each tuple is a combination of (subject, action, location, style, image_path)
        self.prompt_results = [
            (*prompt_definition, image_paths)
            for prompt_definition, image_paths in zip(prompt_combos, image_pths_final)
        ]
        self.next(self.generate_video_from_images)

    def _prompt_results_to_image_dict(self):
        data = {}
        for prompt_result in self.prompt_results:
            subject, action, location, style, image_paths = prompt_result
            for image_path in image_paths:
                fn = os.path.basename(image_path)
                data[fn] = prompt_result
        return data

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
        from config import get_style_video_namedtuple

        # Based on how StabilityAI has devised the code, We will HAVETO
        # download the model to a folder called checkpoints.
        print("Downloading Video Model")
        model_store = self._get_video_model_store()
        os.makedirs("./checkpoints", exist_ok=True)
        model_store.download(self.video_model_version, "./checkpoints")
        print("Generating Videos")

        self.style_outputs = []
        seed = self.config.video.seed
        if self.fully_random:
            # Derive seed from metaflow pathspec
            seed = hash(current.pathspec)
        with tempfile.TemporaryDirectory() as _dir:
            # Downloads the Images
            image_store = ModelStore(self.stored_images_root)
            image_store.download("images", _dir)
            # Create a dictionary mapping file name with the images + prompt
            # This dictionary can now hold the results of the video generation
            prompt_dict = self._prompt_results_to_image_dict()
            path_dict = {
                k: {
                    "video_paths": {},
                    "results_tuple": prompt_dict[
                        k
                    ],  # results_tuple hold all the data cruched in the last step
                }
                for k in prompt_dict
            }

            image_paths = [
                os.path.join(_dir, f) for f in sorted(os.listdir(_dir)) if ".png" in f
            ]
            _args = [
                self.video_model_version,
                image_paths,
                self.config.video.inference_config,
                seed,
            ]
            for (
                image_path,
                image_bytes,
                video_bytes,
                motion_bucket_id,
            ) in ImageToVideo.generate(*_args):
                print("Saving Video")
                save_path = self.save_image_and_video(image_bytes, video_bytes)
                print("Video Saved To Path %s" % save_path)
                img_file_name = os.path.basename(image_path)
                path_dict[img_file_name]["video_paths"][motion_bucket_id] = save_path

        self.style_outputs = [
            get_style_video_namedtuple(
                *(
                    *path_dict[image_path]["results_tuple"],
                    path_dict[image_path]["video_paths"],
                )
            )
            for image_path in path_dict
        ]
        shutil.rmtree("./checkpoints")
        self.next(self.join_subjects)

    @step
    def join_subjects(self, inputs):
        self.style_outputs = [j for i in inputs for j in i.style_outputs]
        self.merge_artifacts(
            inputs,
            include=[
                "video_model_version",
                "image_model_version",
            ],
        )
        self.next(self.end)

    @step
    def end(self):
        print("Finished successfully")


if __name__ == "__main__":
    TextToVideoForeach()
