# Run Stable Diffusion With Metaflow 👋

This repository offers you a framework to create massive amounts of AI-generated images using the Stable Diffusion model. 
The Stable Diffusion model is integrated into a Metaflow workflow that will help you scale horizontally or vertically to quickly produce as many images as you need. To run the code in this repository you will need access to a Metaflow deployment configured with S3 storage. If you want to learn more about Metaflow or need help getting set up, find us on [Slack](http://slack.outerbounds.co/)!

![](/images/einstein_grid.png)

<p align = "center">
Stable Diffusion Intepretations of Einstein
</p>

# Operate Metaflow on AWS Infrastructure
Before running the flow ensure that Metaflow-related infrastructure is [deployed](https://outerbounds.com/docs/aws-deployment-guide/) and [configured](https://outerbounds.com/docs/configure-metaflow/) on your AWS account and GPUs are configured for the compute environment (AWS Batch / EKS). 

If you don't have infrastructure setup, you can set it up with this [cloudformation template](https://github.com/outerbounds/metaflow-tools/blob/master/aws/cloudformation/metaflow-cfn-template.yml). To deploy the GPU infrastructure on AWS, change the [ComputeEnvInstanceTypes](https://github.com/outerbounds/metaflow-tools/blob/d0da1fa4f9aa6845f8091d06a1b7a99962986c98/aws/cloudformation/metaflow-cfn-template.yml#L42) in the Cloudformation template or the Cloudformation UI. More detailed instructions on setting up infrastructure can be found [here](https://outerbounds.com/docs/cloudformation/).


# Install Dependencies

## Use `env.yml` with `conda`

We have included a conda environment in the form of a `env.yml` file for you to use. You can install and activate the environment by running the following commands from your terminal:
```
conda install mamba -n base -c conda-forge
mamba env create -f env.yml
conda activate meta-diffusion
```

## Use `requirements.txt` with `venv`
If you prefer to use [venv](https://docs.python.org/3/library/venv.html) you can create and activate a new environment by running the following commands from your terminal:
```
python venv -m ./meta-diffusion
source ./meta-diffusion/bin/activate
```
Note if you get an error installing the `transformers` library, you may need to install the [Rust compiler](https://rustup.rs/). You can do so like:
```
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

# Run the Code
Before running the flow ensure you have the necessary AWS infrastructure setup for Metaflow. These flows require S3 and GPU/s.


## Step 1 : Download the Stable Diffusion Huggingface model
- Ensure that you have signed the waiver for [CompVis/stable-diffusion-v-1-4-original](https://huggingface.co/CompVis/stable-diffusion-v-1-4-original) model on the Huggingface hub.
- Create a [Huggingface hub access token](https://huggingface.co/docs/hub/security-tokens)
- Run the below command after replacing `<myhuggingfacetoken>` with the Huggingface hub token created in the previous step. Run this command only once to download the model to the local machine. 
    ```sh
    HF_TOKEN=<myhuggingfacetoken> python model_download.py
    ```

## Step 2: Run Metaflow Flows


### ⭐ Generate Images from a Simple Prompt ⭐ 
**Source File** : [meta_diffusers_text.py](./meta_diffusers_text.py)


**Run Command** : 
```sh
python meta_diffusers_text.py run \
    --max-parallel 10 \
    --config-file ./text_config.yaml
```

The schema of the [text-config.yaml](./text_config.yaml) can be found the [config.py](./config.py) file.

**Options**:
```
Options:
  --config-file FILEPATH    experiment config file path
  --config JSON             experiment config
  --max-parallel INTEGER    This parameter will limit the amount of
                            parallelisation we wish to do. Based on the value
                            set here, the foreach will fanout to that many
                            workers.  [default: 1]

  --local-model-path TEXT   Path of local model to use instead of the one in
                            S3
```

**Running Locally** : To run this flow locally, ensure that you have installed the `requirements.txt` file and commented the `@kubernetes` decorator in the [flow file](./meta_diffusers_text.py).


### ⭐ Generate Many Images with Diverse Styles ⭐
**Source File** : [meta_dynamic_prompts.py](./meta_dynamic_prompts.py)

**Run Command** : 
```sh
python meta_dynamic_prompts.py run --config-file ./style_config.yaml
```
The schema of the [style_config.yaml](./style_config.yaml) can be found the [config.py](./config.py) file.

**Options**:
```
Options:
  --config-file FILEPATH     experiment config file path
  --config JSON              experiment config
  --images-per-card INTEGER  Maximum number of images to show per @card
                             [default: 10]
```

**Running Locally** : To run this flow locally, ensure that you have installed the `requirements.txt` file and commented the `@kubernetes` decorator in the [flow file](./meta_dynamic_prompts.py).


### ⭐ Generate Videos from a Custom Prompt ⭐

**Source File** : [text_to_video_flow.py](./text_to_video_flow.py)

**Run Command** : 
```sh
python text_to_video_flow.py --package-suffixes .yaml run --config-file ./video_config.yaml
```
The schema of the [video_config.yaml](./video_config.yaml) can be found the [config.py](./config.py) file.

**Extract Videos Post Completion**
```python
from metaflow import Flow, namespace
import os
from model_store import ModelStore
namespace(None)
flow = Flow("TextToVideo")
run = flow.latest_successful_run
store = ModelStore.from_path(run["generate_video_from_images"].task.pathspec)
store.download("final_render","final_render")
```

### ⭐⭐⭐ Generate a Cross Product of Videos from a Set of Custom Prompts ⭐⭐⭐
**Source File** : [text_to_video_foreach_flow.py](./text_to_video_foreach_flow.py)

**Run Command** : 
```sh
python text_to_video_foreach_flow.py --package-suffixes .yaml run --config-file video_config_foreach.yaml
```
The schema of the [video_config_foreach.yaml](./video_config_foreach.yaml) can be found the [config.py](./config.py) file.

**Extract Videos Post Completion**
```python
from notebook_utils import download_foreach_run_videos
download_foreach_run_videos(run_id='<YOUR RUN ID>')
```