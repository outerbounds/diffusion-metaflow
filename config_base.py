from config import load_config
import tempfile
from metaflow import IncludeFile, Parameter, JSONType


def _to_file(file_bytes, extension=None):
    params = {
        "suffix": f".{extension.replace('.', '')}" if extension is not None else None,
        "delete": True,
        "dir": "./",
    }
    latent_temp = tempfile.NamedTemporaryFile(**params)
    latent_temp.write(file_bytes)
    latent_temp.seek(0)
    return latent_temp


class ConfigBase:
    """
    Base class for all config needed for this flow as well as any dependent flows.

    This class can be inherited by downstream classes or even used a mixin.

    This class is meant for reuse in Metaflow flows which want to resue the configuration parameters of this training flow so
    that they can call downstream flows with the same configuration parameters.

    Example:
    --------
    - Upstream flow which is preparing data is inheriting the configuration schema / parameters from this class
    - This way correct configuration parsed in both flows while we can also pass the configuration from the upstream flow to the downstream flow while ensuring that the configuration is valid.
    - This pattern is very useful when we have a complex configuration schema and we want to reuse it in multiple flows. These flows may be invoked asynchronously using event handlers, so having a common configuration schema parser is very useful.

    All downstream flows will have to inherit this class and set the `config` property in this class.
    This way we will be able to access the config directly.
    The `_CORE_CONFIG_CLASS` property of this class should be set to the class which will be used to parse the configuration.

    ```
    _CORE_CONFIG_CLASS = ImageStylePromptDiffusionConfig
    @property
    def config(self) -> ImageStylePromptDiffusionConfig:
        return self._get_config()
    ```
    """

    def _resolve_config(self):
        if self._CORE_CONFIG_CLASS is None:
            raise ValueError(
                "Please set the _CORE_CONFIG_CLASS property of this class to the class which will be used to parse the configuration"
            )
        if (
            self.experiment_config is not None
            and self.experiment_config_file is not None
        ):
            raise ValueError("Cannot specify both --config or --config-file")
        elif self.experiment_config is None and self.experiment_config_file is None:
            raise ValueError("Must specify either --config or --config-file")
        if self.experiment_config is not None:
            return load_config(self.experiment_config, self._CORE_CONFIG_CLASS)
        if self.experiment_config_file is not None:
            temf = _to_file(
                bytes(self.experiment_config_file, "utf-8"),
            )
            return load_config(temf.name, self._CORE_CONFIG_CLASS)

    _config = None

    _CORE_CONFIG_CLASS = None

    def _get_config(self):
        if self._config is not None:
            return self._config
        self._config = self._resolve_config()
        return self._config

    experiment_config_file = IncludeFile(
        "config-file", help="experiment config file path", default=None
    )

    experiment_config = Parameter(
        "config", help="experiment config", default=None, type=JSONType
    )

    def config_report(self):
        from metaflow.cards import Markdown
        from omegaconf import OmegaConf

        return [
            Markdown(f"## Experiment Config"),
            Markdown(f"```\n{OmegaConf.to_yaml(self.config)}```"),
        ]
