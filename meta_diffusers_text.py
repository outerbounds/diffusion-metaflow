import os
import tempfile
from metaflow import FlowSpec, step, Parameter, batch, card, current, kubernetes, pypi
from metaflow.cards import Image, Markdown
from metaflow.metaflow_config import DATASTORE_SYSROOT_S3

from base import DIFF_USERS_IMAGE, ModelOperations, TextToImageDiffusion

DEFAULT_PROMPT = "mahatma gandhi, tone mapped, shiny, intricate, cinematic lighting, highly detailed, digital painting, artstation, concept art, smooth, sharp focus, illustration"

def _convert(number, base_unit, convert_unit):
    # base_unit : GB or MB or KB or B
    # convert_unit : GB or MB or KB or B
    # number : number of base_unit
    # return : number of convert_unit
    units = ["B", "KB", "MB", "GB"]
    if base_unit not in units or convert_unit not in units:
        raise ValueError("Invalid unit")
    base_unit_index = units.index(base_unit)
    convert_unit_index = units.index(convert_unit)
    factor = pow(1024, abs(base_unit_index - convert_unit_index))
    if base_unit_index < convert_unit_index:
        return round(number / factor, 3)
    else:
        return round(number * factor, 3)

class TextToImages(FlowSpec, ModelOperations, TextToImageDiffusion):
    """
    Create images from prompt values using Stable Diffusion.
    """

    prompts = Parameter("prompt", type=str, default=DEFAULT_PROMPT, multiple=True)

    num_images = Parameter(
        "num-images", type=int, default=10, help="Number of images to create per prompt"
    )

    # Todo : make seed for a full range more customizable.
    seed = Parameter("seed", default=42, type=int)

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

    @step
    def start(self):
        import random
        import math

        import itertools

        # Uploade the model if it is not present.
        self.upload_model_if_none_exists()

        # create seed values for each inference step
        random.seed(self.seed)
        chunk_size = math.ceil(self.num_images / self.max_parallel)
        self.rand_seeds = [
            random.randint(1, 10**7) for i in range(0, self.num_images, chunk_size)
        ]
        print("Creating %s tasks " % str(len(self.rand_seeds)))
        # Fanout the inference over the chunked seed values.
        self.next(self.generate_images, foreach="rand_seeds")

    @kubernetes(image=DIFF_USERS_IMAGE, gpu=1, cpu=4, memory=16000, disk=_convert(100, "GB", "MB"))
    @card
    @step
    def generate_images(self):
        with tempfile.TemporaryDirectory(self.model_version) as _dir:
            import math

            chunk_size = math.ceil(self.num_images / self.max_parallel)
            model_path = _dir
            if self.local_model_path is not None:
                model_path = self.local_model_path
            else:
                self.download_model(folder=_dir)
            idx = 0
            for images, prompt in self.infer_prompt(
                self.prompts, model_path, chunk_size, self.input
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
