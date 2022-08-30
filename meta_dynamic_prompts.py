import os
import tempfile
from metaflow import FlowSpec, step, Parameter, batch, card, current
from metaflow.cards import Image, Markdown, Table
from metaflow.metaflow_config import DATASTORE_SYSROOT_S3

from base import DIFF_USERS_IMAGE, ModelOperations, TextToImageDiffusion

DEFAULT_STYLES = ",".join(["van gogh", "salvador dali", "warhol","Art Nouveau", "Ansel Adams", "basquiat", ])

DEFAULT_PROMPT = "Jack black as iron man"

class DynamicPromptsToImages(FlowSpec, ModelOperations, TextToImageDiffusion):
    
    input_prompts = Parameter("prompt", type=str, default=DEFAULT_PROMPT, multiple=True)

    styles = Parameter("styles", type=str, default=DEFAULT_STYLES, help="Comma seperated list of styles")
    
    num_images = Parameter("num-images", type=int, default=10)

    # Todo : make seed for a full range more customizable.
    seed = Parameter("seed", default=420, type=int)

    max_gpus = Parameter(
        "max-gpus",
        default=4,
        type=int,
        help="This parameter will limit the amount of parallelisation we wish to do. If there are about a 100 images, this parameter will help derive the number of chunks that we need to make and the total images in each chunk.",
    )

    @staticmethod
    def create_prompt(prompt, style):
        return "%s by %s" % (prompt, style)

    @step
    def start(self):
        import random
        import math

        import itertools
        styles = self.styles.split(',')
        # Uploade the model if it is not present.
        self.upload_model_if_none_exists()
        self.prompt_combo = list(itertools.product(self.input_prompts, styles))
        self.prompts = [ self.create_prompt(p, s) for p, s in self.prompt_combo]
        # create seed values for each inference step
        random.seed(self.seed)
        chunk_size = math.ceil(self.num_images / self.max_gpus)
        self.rand_seeds = [
            random.randint(1, 10 ** 7) for i in range(0, self.num_images, chunk_size)
        ]

        # Fanout the inference over the chunked seed values.
        self.next(self.generate_images, foreach="rand_seeds")

    @card
    @batch(image=DIFF_USERS_IMAGE, gpu=1, cpu=4, memory=16000)
    @step
    def generate_images(self):
        with tempfile.TemporaryDirectory(self.model_version) as _dir:
            import math
            chunk_size = math.ceil(self.num_images / self.max_gpus)
            self.download_model(folder=_dir)
            idx = 0
            prompt_idx = 0
            self.image_index = []
            # _table_chunks will hold one image each for a `style, prompt` pair. 
            # Since each prompt will create the same values
            self.table_chunks = {i : [] for i in range(chunk_size)}
            print("Creating %d images each for %d prompts" %(chunk_size, len(self.prompts)))
            for images, prompt in self.infer_prompt(self.prompts,_dir, chunk_size, self.input):
                idx += len(images)
                _prmt, _style = self.prompt_combo[prompt_idx]
                # save the images.
                for _idx, i in enumerate(images):
                    art_name = "image_%d_%d" % (idx, _idx)
                    self.save_artifact(art_name, i)
                    self.table_chunks[_idx].append(
                        (_prmt, _style, art_name)
                    )
                    self.image_index.append(
                        (_prmt, _style, art_name)
                    )
                prompt_idx+=1
        self.next(self.paint_images)
    
    @card(type='blank')
    @step
    def paint_images(self):
        for idx, v in enumerate(self.table_chunks.values()):
            current.card.extend([Markdown("## Prompts Tables %d " %idx)]+self.create_table(v))
        self.next(self.join)

    def create_table(self, table_chunk, cols=3,rows=2):

        def _create_table(prompt, styles, artifact_map):
            table_rows = [
                [" "] + [Markdown("%s" % s) for s in styles]
            ]
            row = [Markdown("%s" % prompt)]
            for k in styles:
                art = self.get_artifact(artifact_map[prompt][k])
                row.append(Image.from_pil_image(art))
            table_rows.append(row)
            return Table(table_rows)

        pmap = {}        
        for _prmt, _style, art_name in table_chunk:
            if _prmt not in pmap:
                pmap[_prmt] = {_style:art_name}
            else:
                pmap[_prmt][_style] = art_name

        all_styles = self.styles.split(',')
        all_tables = []
        for p in self.input_prompts:
            for _styles in [all_styles[i:i + cols] for i in range(0, len(all_styles), cols)]:
                all_tables.append(_create_table(p, _styles, pmap,))
        return all_tables

    @step
    def join(self, inputs):    
        self.next(self.end)

    @step
    def end(self):
        print("Done!")


if __name__ == "__main__":
    DynamicPromptsToImages()
