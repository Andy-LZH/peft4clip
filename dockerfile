FROM pytorch/pytorch:latest

# copy models and configs
COPY . /clip_vpt

# setup working directory
WORKDIR /clip_vpt

# install git
RUN apt-get update && apt-get install -y git

# install requirements
RUN pip install -r requirements.txt