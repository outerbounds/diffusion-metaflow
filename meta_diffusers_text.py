import os
import tempfile
from metaflow import FlowSpec, step, Parameter, batch, card, current
from metaflow.cards import Image, Markdown
from metaflow.metaflow_config import DATASTORE_SYSROOT_S3

from base import DIFF_USERS_IMAGE, ModelOperations

DEFAULT_PROMPT = "Jack black as iron man, tone mapped, shiny, intricate, cinematic lighting, highly detailed, digital painting, artstation, concept art, smooth, sharp focus, illustration"


class TextToImages(FlowSpec, ModelOperations):

    prompts = Parameter("prompt", type=str, default=DEFAULT_PROMPT, multiple=True)

    num_images = Parameter("num-images", type=int, default=10)

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

    seed = Parameter("seed", default=420, type=int)

    # Todo : make seed for a full range more customizable.

    max_gpus = Parameter(
        "max-gpus",
        default=4,
        type=int,
        help="This parameter will limit the amount of parallelisation we wish to do. If there are about a 100 images, this parameter will help derive the number of chunks that we need to make and the total images in each chunk.",
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
            random.randint(1, 10**7)
            for i in range(0, self.num_images, chunk_size)
        ]
        # Fanout the inference over the chunked seed values.
        self.next(self.generate_images, foreach="rand_seeds")

    @card
    @batch(image=DIFF_USERS_IMAGE, gpu=1, cpu=8, memory=20000, queue=os.environ["BATCH_QUEUE"])
    @step
    def generate_images(self):
        import diffusion
        with tempfile.TemporaryDirectory(self.model_version) as _dir:
            import math
            chunk_size = math.ceil(self.num_images / self.max_gpus)
            self.download_model(folder=_dir)
            idx = 0
            for prompt in self.prompts:
                images = diffusion.infer_prompt(
                    _dir,
                    prompt,
                    num_images= chunk_size, 
                    batch_size= self.batch_size,
                    width= self.output_width, 
                    height= self.output_height, 
                    num_steps= self.num_steps,
                    seed= self.input
                )
                idx+=len(images)
                current.card.extend(
                    [Markdown("## Prompt : %s" % prompt)]+[Image.from_pil_image(i) for i in images ]
                )
                # save the bytes value of the images.
                for _idx, i in enumerate(images):
                    self.save_artifact("image_%d_%d"%(idx,_idx), i)

        self.next(self.join)

    @step
    def join(self, inputs):
        self.next(self.end)

    @step
    def end(self):
        print("Done!")


if __name__ == "__main__":
    TextToImages()
