import math
import os
import tempfile
import uuid
from metaflow import FlowSpec, step, Parameter, batch, card, current, kubernetes
from metaflow.cards import Image, Markdown, Table, get_cards
from metaflow.metaflow_config import UI_URL

from base import DIFF_USERS_IMAGE, ArtifactStore, TextToImageDiffusion
from config_base import ConfigBase
from model_store import ModelStore
from config import ImageStylePromptDiffusionConfig

from utils import create_chunk_ranges, create_prompt, create_card_index, unit_convert


class DynamicPromptsToImages(FlowSpec, ConfigBase, ArtifactStore, TextToImageDiffusion):
    """
    Create multiple prompts in different styles using Stable Diffusion
    """

    images_per_card = Parameter(
        "images-per-card",
        type=int,
        default=10,
        help="Maximum number of images to show per @card",
    )

    metaflow_ui_url = Parameter(
        "ui-url",
        type=str,
        default=UI_URL,
        help="Url to the Metaflow UI. If provided then an index card is created for the `join_styles` @step",
    )

    _CORE_CONFIG_CLASS = ImageStylePromptDiffusionConfig

    @property
    def config(self) -> ImageStylePromptDiffusionConfig:
        return self._get_config()

    def _get_model_store(self):
        return ModelStore.from_config(self.config.model_config.model_store)

    @property
    def model_version(self):
        return self.config.model_config.model_store.model_version

    @staticmethod
    def create_image_id():
        return "image_%s" % str(uuid.uuid4())

    @step
    def start(self):
        import random

        # Upload the model if it is not present.
        store_config = self.config.model_config.model_store
        model_store = self._get_model_store()
        # Upload the model if it is not present.
        model_store.upload_model_if_none_exists(store_config)

        _seed = self.config.seed
        styles = self.config.style_prompt_config.styles
        # create seed values for each inference step . Since we are parallelizing over styles we will create the same number
        random.seed(_seed)
        self.style_rand_seeds = list(
            zip([random.randint(1, 10**7) for i in range(len(styles))], styles)
        )

        # Fanout over the styles provided from the parameters.
        self.next(self.generate_images, foreach="style_rand_seeds")

    @card
    @kubernetes(
        image=DIFF_USERS_IMAGE,
        gpu=1,
        cpu=4,
        memory=16000,
        disk=unit_convert(100, "GB", "MB"),
    )
    @step
    def generate_images(self):
        import itertools

        self.inference_seed, self.inference_style = self.input
        # Merge the prompts and style created in the foreach
        subjects = self.config.style_prompt_config.subjects
        self.prompt_combo = list(itertools.product(subjects, [self.inference_style]))
        self.prompts = [create_prompt(p, s) for p, s in self.prompt_combo]

        # Download the model, run the model on the prompt and save the image given by the model.
        with tempfile.TemporaryDirectory() as _dir:
            self._get_model_store().download(self.model_version, _dir)
            idx = 0
            prompt_idx = 0
            self.image_index = []
            for images, _ in self.infer_prompt(
                _dir,
                self.inference_seed,
                self.prompts,
                self.config.style_prompt_config.num_images,
                self.config.inference_config,
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
        card_indx_md = create_card_index(self.metaflow_ui_url)
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


if __name__ == "__main__":
    DynamicPromptsToImages()
