import shutil
from metaflow import Parameter
from metaflow.metaflow_config import DATASTORE_SYSROOT_S3
import os
from metaflow._vendor import click

DIFF_USERS_IMAGE = "valayob/diffusers-base"
MODEL_PATH = "./models"
MODELS_BASE_S3_PATH = "models/diffusion-models/"
MODEL_VERSION = "stable-diffusion-v1-4"


def safe_mkdirs(_dir):
    try:
        os.makedirs(_dir)
    except FileExistsError:
        pass


class ModelOperations:
    """
    ## ModelOpeartions

    This class is responsible to for saving and loading a model from an S3 bucket. 
    This class can be used like a mixin with any FlowSpec class. 
    The style of writing Metaflow flows allows scoping parameters and functionality. 
    This will use the `metaflow.metaflow_config.DATASTORE_SYSROOT_S3` ie the `METAFLOW_DATASTORE_SYSROOT_S3` configuration variable 
    to determine the root path for S3.

    ### Parameters
        - `pretrained_model_path` : Path to the model on the local machine
        - `force_upload` : Even if the model is present in the S3 bucket, upload the model again.
        - `s3_base_path` : S3 prefix for the path of the S3 saved model.
        - `model_version` : Version of the Stable diffusion model.
    """

    pretrained_model_path = Parameter(
        "model-path",
        type=click.Path(),
        default=MODEL_PATH,
        help="Path to the downloaded model on the local machine.",
    )

    force_upload = Parameter(
        "force-upload",
        is_flag=True,
        default=False,
        help="Force upload the model from the local machine",
    )

    s3_base_path = Parameter(
        "s3-prefix",
        type=str,
        default=MODELS_BASE_S3_PATH,
        help="prefix of the path where models are stored in S3.",
    )

    model_version = Parameter("model-version", type=str, default=MODEL_VERSION)

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

    @staticmethod
    def _walk_directory(root):
        path_keys = []
        for path, subdirs, files in os.walk(root):
            for name in files:
                # create a tuple of (key, path)
                path_keys.append(
                    (
                        os.path.relpath(os.path.join(path, name), root),
                        os.path.join(path, name),
                    )
                )
        return path_keys

    def _upload_model(self):
        # This takes place on local from where we upload model to s3
        from metaflow import S3

        with S3(s3root=self.s3_model_path) as s3:
            s3.put_files(self._walk_directory(self.pretrained_model_path))

    def download_model(self, folder):
        from metaflow import S3

        safe_mkdirs(folder)
        with S3(s3root=self.s3_model_path) as s3:
            for s3obj in s3.get_all():
                move_path = os.path.join(folder, s3obj.key)
                if not os.path.exists(os.path.dirname(move_path)):
                    safe_mkdirs(os.path.dirname(move_path))
                shutil.move(s3obj.path, os.path.join(folder, s3obj.key))

    @property
    def s3_model_path(self):
        return os.path.join(
            os.path.dirname(DATASTORE_SYSROOT_S3),
            self.s3_base_path,
            self.model_version,
        )

    def upload_model_if_none_exists(self):
        if not self.force_upload:
            from metaflow import S3

            with S3(s3root=self.s3_model_path) as s3:
                # if we are not force uploading then ensure that all paths are covered
                if len(s3.list_paths()) > 0:
                    print("Model already cached on the cloud. Not force uploading")
                    return
        print("Uploading Model")
        self._upload_model()


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

    batch_size = Parameter(
        "batch-size",
        type=int,
        default=4,
        help="controls the number of images to send to the GPU per batch",
    )

    output_width = Parameter(
        "width", type=int, default=512, help="width of the output image"
    )

    output_height = Parameter(
        "height", type=int, default=512, help="Height of the output image"
    )

    num_steps = Parameter(
        "num-steps", type=int, default=60, help="Number of steps to run inference"
    )

    def infer_prompt(self, prompts, model_path, num_images, seed):
        """
        Extract the images for each prompt and yield the (images, prompt).
        """
        from diffusion import infer_prompt

        for prompt in prompts:
            images = infer_prompt(
                model_path,
                prompt,
                num_images=num_images,
                batch_size=self.batch_size,
                width=self.output_width,
                height=self.output_height,
                num_steps=self.num_steps,
                seed=seed,
            )
            yield (images, prompt)
