import shutil
from metaflow import Parameter, IncludeFile, JSONType
from metaflow.metaflow_config import DATATOOLS_S3ROOT
import os
from metaflow._vendor import click
import tempfile
from config import ImageInferenceConfig

DIFF_USERS_IMAGE = "valayob/sdvideo-base:1.1"


def safe_mkdirs(_dir):
    try:
        os.makedirs(_dir)
    except FileExistsError:
        pass


class ArtifactStore:
    def get_artifact(self, name):
        return getattr(self, name, None)

    def save_artifact(self, name, value):
        """
        This method helps store keys to `self` that are dynamically constructed.
        We store things to self because loading all artifacts in object can create on huge object which would take a lot of time when loading with the metaflow client.
        Storing individual artifacts via this method can make the access to individual artifacts very fast when using the Metaflow client. If they are accessed via `Task` object the the object also offers a dictionary like interface to access the data.
        """
        # `ARTIFACT_REFERENCES` will hold all artifacts saved via `save_artifact` method.
        # These references can be accessed via `get_artifact` method of the class.
        if self.get_artifact("ARTIFACT_REFERENCES") is None:
            self.ARTIFACT_REFERENCES = []
        self.ARTIFACT_REFERENCES.append(name)
        setattr(self, name, value)


class TextToImageDiffusion:
    """
    ## TextToImageDiffusion
    This is a wrapper over `diffusers` library. It exposes `metaflow.Parameter`'s neccessary for calling stable diffusion library.

    ### Parameters
        - `batch_size` : controls the number of images to send to the GPU per batch
        - `output_width` : width of the output image
        - `output_height` : width of the input image
        - `num_steps` : number of steps to run the reverse diffusion process.
    """

    def infer_prompt(
        self,
        model_path,
        seed,
        prompts,
        num_images,
        inference_conf: ImageInferenceConfig,
    ):
        """
        Extract the images for each prompt and yield the (images, prompt).
        """
        from diffusion import infer_prompt

        img_prompts = []
        for prompt in prompts:
            images = infer_prompt(
                model_path,
                prompt,
                num_images=num_images,
                batch_size=inference_conf.batch_size,
                width=inference_conf.width,
                height=inference_conf.height,
                num_steps=inference_conf.num_steps,
                seed=seed,
            )
            img_prompts.append((images, prompt))
        return img_prompts
