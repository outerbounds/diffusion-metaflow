import os
import tempfile
from metaflow import FlowSpec, step, Parameter, batch, card, current
from metaflow.cards import Image, Markdown
from metaflow.metaflow_config import DATASTORE_SYSROOT_S3

from base import DIFF_USERS_IMAGE, ModelOperations, TextToImageDiffusion

DEFAULT_PROMPT = "mahatma gandhi, tone mapped, shiny, intricate, cinematic lighting, highly detailed, digital painting, artstation, concept art, smooth, sharp focus, illustration"


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

    max_gpus = Parameter(
        "max-gpus",
        default=4,
        type=int,
        help="This parameter will limit the amount of parallelisation we wish to do. If there are about a 100 images per prompt, this parameter will help derive the number of chunks that we need to make and the total images in each chunk.",
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
        chunk_size = math.ceil(self.num_images / self.max_gpus)
        self.rand_seeds = [
            random.randint(1, 10**7) for i in range(0, self.num_images, chunk_size)
        ]
        # Fanout the inference over the chunked seed values.
        self.next(self.generate_images, foreach="rand_seeds")

    @card
    @batch(image=DIFF_USERS_IMAGE, gpu=1, cpu=4, memory=8000)
    @step
    def generate_images(self):
        with tempfile.TemporaryDirectory(self.model_version) as _dir:
            import math

            chunk_size = math.ceil(self.num_images / self.max_gpus)
            self.download_model(folder=_dir)
            idx = 0
            for images, prompt in self.infer_prompt(
                self.prompts, _dir, chunk_size, self.input
            ):
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
