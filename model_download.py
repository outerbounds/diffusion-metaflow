# make sure you have signed the waiver on hugging face hub before downloading the model.
import os
from base import MODEL_PATH, MODEL_VERSION
from diffusers import StableDiffusionPipeline, LMSDiscreteScheduler

lms = LMSDiscreteScheduler(
    beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear"
)
pipe = StableDiffusionPipeline.from_pretrained(
    "CompVis/%s" % MODEL_VERSION,
    scheduler=lms,
    use_auth_token=os.environ["HF_TOKEN"],
)
pipe.save_pretrained(MODEL_PATH)
