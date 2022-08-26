FROM anibali/pytorch:1.8.1-cuda11.1-ubuntu20.04
RUN pip install diffusers==0.2.4 transformers==4.21.2 scipy==1.9.0 pillow ftfy
RUN pip install --upgrade torch