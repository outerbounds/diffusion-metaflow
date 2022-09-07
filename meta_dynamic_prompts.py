import math
import os
import tempfile
import uuid
from metaflow import FlowSpec, step, Parameter, batch, card, current
from metaflow.cards import Image, Markdown, Table, get_cards
from metaflow.metaflow_config import DATASTORE_SYSROOT_S3

from base import DIFF_USERS_IMAGE, ModelOperations, TextToImageDiffusion

DEFAULT_STYLES = ",".join(
    [
        "van gogh",
        "salvador dali",
        "warhol",
        "Art Nouveau",
        "Ansel Adams",
        "basquiat",
    ]
)

DEFAULT_PROMPT = [
    "Mahatma gandhi",
    "dalai lama",
    "alan turing",
]


def create_chunk_ranges(arr, chunk_size):
    num_splits = math.ceil(len(arr) / chunk_size)
    index_list = []  # Will hold start,end delimiters
    for i in range(1, num_splits + 1):
        end = i * chunk_size
        start = (i - 1) * chunk_size
        if end > len(arr):
            end = start + (len(arr) - start)
        index_list.append((start, end))
    return index_list


def create_card_url(ui_url, task):
    fl, rn, st, ts = task.pathspec.split("/")
    cards = get_cards(task)
    return "%s/api/flows/%s/runs/%s/steps/%s/tasks/%s/cards/%s" % (
        ui_url,
        fl,
        rn,
        st,
        ts,
        cards[0].hash,
    )


def create_prompt(prompt, style):
    return "%s by %s" % (prompt, style)


class DynamicPromptsToImages(FlowSpec, ModelOperations, TextToImageDiffusion):
    """
    Create multiple prompts in different styles using Stable Diffusion
    """

    input_prompts = Parameter(
        "subject",
        type=str,
        default=DEFAULT_PROMPT,
        multiple=True,
        help="The subject based on which images are generated",
    )

    styles = Parameter(
        "styles",
        type=str,
        default=DEFAULT_STYLES,
        help="Comma seperated list of styles",
    )

    num_images = Parameter(
        "num-images",
        type=int,
        default=10,
        help="Number of images to create per (prompt, style)",
    )

    images_per_card = Parameter(
        "images-per-card",
        type=int,
        default=10,
        help="Maximum number of images to show per @card",
    )

    metaflow_ui_url = Parameter(
        "ui-url",
        type=str,
        default=None,
        help="Url to the Metaflow UI. If provided then an index card is created for the `join_styles` @step",
    )

    seed = Parameter("seed", default=42, type=int, help="Seed to use for inference.")

    @staticmethod
    @staticmethod
    def create_image_id():
        return "image_%s" % str(uuid.uuid4())

    @step
    def start(self):
        import random

        styles = self.styles.split(",")
        # Uploade the model if it is not present.
        self.upload_model_if_none_exists()

        # create seed values for each inference step . Since we are parallelizing over styles we will create the same number
        random.seed(self.seed)
        self.style_rand_seeds = list(
            zip([random.randint(1, 10**7) for i in range(len(styles))], styles)
        )

        # Fanout over the styles provided from the parameters.
        self.next(self.generate_images, foreach="style_rand_seeds")

    @card
    @batch(image=DIFF_USERS_IMAGE, gpu=1, cpu=4, memory=16000)
    @step
    def generate_images(self):
        import itertools

        self.inference_seed, self.inference_style = self.input
        # Merge the prompts and style created in the foreach
        self.prompt_combo = list(
            itertools.product(self.input_prompts, [self.inference_style])
        )
        self.prompts = [create_prompt(p, s) for p, s in self.prompt_combo]

        # Download the model, run the model on the prompt and save the image given by the model.
        with tempfile.TemporaryDirectory(self.model_version) as _dir:
            self.download_model(folder=_dir)
            idx = 0
            prompt_idx = 0
            self.image_index = []
            for images, _ in self.infer_prompt(
                self.prompts, _dir, self.num_images, self.inference_seed
            ):
                idx += len(images)
                _prmt, _style = self.prompt_combo[prompt_idx]

                for i in images:
                    art_name = self.create_image_id()
                    # `save_artifact` will save images to the `self`.
                    self.save_artifact(art_name, i)
                    self.image_index.append((_prmt, _style, art_name))
                prompt_idx += 1

        # Create a `start,end` index list that will help create more foreach's for painting cards.
        # We create a foreach to paint cards so that we can make a light (size-wise) HTML page for each card.
        self.index_list = create_chunk_ranges(
            self.image_index, self.images_per_card
        )  # Will hold start,end
        self.next(self.paint_cards, foreach="index_list")

    @card(type="blank")
    @step
    def paint_cards(self):
        _start, _end = self.input
        # Create cards for the images generated in the `generate_images` step
        self.image_index = self.image_index[_start:_end]
        current.card.extend(
            [Markdown("## Prompts for Style : %s" % self.inference_style)]
            + self.create_table(self.image_index)
        )
        self.next(self.join_cards)

    @step
    def join_cards(self, inputs):
        self.foreach_join_commit(inputs)
        self.next(self.join_styles)

    @step
    def join_styles(self, inputs):
        self.foreach_join_commit(inputs)
        self.next(self.end)

    @card(type="blank")
    @step
    def end(self):
        # Create an index of the cards created in this step's card
        card_indx_md = self.create_card_index()
        if card_indx_md is not None:
            current.card.extend(card_indx_md)
        print("Done!")

    def create_table(self, image_index, cols=3):
        """Create a table of images for each style"""

        def _create_table(prompt, artifacts):
            table_rows = []
            row = [Markdown("## %s" % prompt)]
            for _aname in artifacts:
                art = self.get_artifact(_aname)
                row.append(Image.from_pil_image(art))
            table_rows.append(row)
            return Table(table_rows)

        pmap = {}
        for _prmt, _, art_name in image_index:
            if _prmt not in pmap:
                pmap[_prmt] = [art_name]
            else:
                pmap[_prmt].append(art_name)

        all_tables = []
        for _prmt, _art_names in pmap.items():
            for _arts in [
                _art_names[i : i + cols] for i in range(0, len(_art_names), cols)
            ]:
                all_tables.append(_create_table(_prmt, _arts))
        return all_tables

    def foreach_join_commit(self, inputs):
        """Saves the artifacts to the main object so that it can be accessed later via metaflow client."""
        self.image_index = []
        images = set()
        for input in inputs:
            for _, _, art_name in input.image_index:
                assert art_name not in images
                images.add(art_name)
                self.save_artifact(
                    art_name,
                    getattr(
                        input,
                        art_name,
                    ),
                )
            self.image_index.extend(input.image_index)

    def create_card_index(self):
        """construct the URL of each card in one single index and add it to current card"""
        card_paths = []
        if self.metaflow_ui_url is None:
            return
        mf_ui_url = self.metaflow_ui_url
        if self.metaflow_ui_url.endswith("/"):
            mf_ui_url = self.metaflow_ui_url[:-1]

        from metaflow import current, Run, parallel_map, Task

        run_pathspec = "/".join(([current.flow_name, current.run_id]))
        tasks_pathspecs = [t.pathspec for t in list(Run(run_pathspec)["paint_cards"])]

        def make_md_str(pthspc):
            t = Task(pthspc)
            style = t["inference_style"].data
            prompts = ", ".join(set([p for p, _, _ in t["image_index"].data]))
            url_path = create_card_url(mf_ui_url, t)
            md_str = "[[%s]](%s)" % (create_prompt(prompts, style), url_path)
            return md_str

        md_strs = parallel_map(make_md_str, tasks_pathspecs)
        card_paths.extend([Markdown(m) for m in md_strs])
        return [Markdown("# Path To Cards On Metaflow UI")] + card_paths


if __name__ == "__main__":
    DynamicPromptsToImages()
