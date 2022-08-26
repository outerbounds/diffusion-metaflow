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

    pretrained_model_path = Parameter(
        "model-path", type=click.Path(), default=MODEL_PATH
    )

    force_upload = Parameter("force-upload", is_flag=True, default=False)

    s3_base_path = Parameter(
        "s3-prefix",
        type=str,
        default=MODELS_BASE_S3_PATH,
        help="prefix of the path where models are stored in S3.",
    )

    model_version = Parameter("model-version", type=str, default=MODEL_VERSION)

    def get_artifact(self, name):
        return getattr(self,name)

    def save_artifact(self, name ,value):
        setattr(self,name, value)


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
