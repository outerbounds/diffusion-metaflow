from metaflow import S3, current, Parameter
from metaflow.metaflow_config import DATATOOLS_S3ROOT
from config import ModelStoreConfig

import os
import shutil


class ModelStore:
    @classmethod
    def from_path(cls, base_prefix):
        return cls(os.path.join(DATATOOLS_S3ROOT, base_prefix))

    @classmethod
    def from_config(cls, config: ModelStoreConfig):
        return cls(os.path.join(DATATOOLS_S3ROOT, config.s3_prefix))

    def __init__(self, model_store_root) -> None:
        # model_store_root is a S3 path to where all files for the model will be loaded and saved
        self._model_store_root = model_store_root

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

    def _upload_model(self, model_path, store_key):
        # This takes place on local from where we upload model to s3
        final_path = os.path.join(self._model_store_root, store_key)
        with S3(s3root=final_path) as s3:
            s3.put_files(self._walk_directory(model_path))

    def already_exists(self, store_key):
        # This takes place on local from where we download model from s3
        final_path = os.path.join(self._model_store_root, store_key)
        with S3(s3root=final_path) as s3:
            if len(s3.list_paths()) == 0:
                return False
        return True

    def _download_model(self, store_key, download_path):
        """
        Parameters
        ----------
        store_key : str
            Key suffixed to the model_store_root to save the model to
        download_path : str
            Path to the folder where the model will be downloaded
        """
        final_path = os.path.join(self._model_store_root, store_key)
        os.makedirs(download_path, exist_ok=True)
        with S3(s3root=final_path) as s3:
            for s3obj in s3.get_all():
                move_path = os.path.join(download_path, s3obj.key)
                if not os.path.exists(os.path.dirname(move_path)):
                    os.makedirs(os.path.dirname(move_path), exist_ok=True)
                shutil.move(s3obj.path, os.path.join(download_path, s3obj.key))

    def upload(self, model_path, store_key):
        """
        Parameters
        ----------
        model_path : str
            Path to the model to be saved
        store_key : str
            Key suffixed to the model_store_root to save the model to
        """
        self._upload_model(model_path, store_key)

    def download(self, store_key, download_path):
        """
        Parameters
        ----------
        store_key : str
            Key suffixed to the model_store_root to download the model from
        download_path : str
            Path to the folder where the model will be downloaded
        """
        if not self.already_exists(store_key):
            raise ValueError(
                f"Model with key {store_key} does not exist in {self._model_store_root}"
            )
        self._download_model(store_key, download_path)

    def upload_model_if_none_exists(self, store_config: ModelStoreConfig):
        if not store_config.force_upload:
            if self.already_exists(store_config.model_version):
                print("Model already cached on the cloud. Not force uploading")
                return
        print("Uploading Model")
        self._upload_model(
            store_config.pretrained_model_path, store_config.model_version
        )
