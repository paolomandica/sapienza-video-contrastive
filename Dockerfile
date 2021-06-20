FROM pytorch/pytorch:1.8.0-cuda11.1-cudnn8-devel

# Set the working directory
WORKDIR /video-contrastive

# Clone the repo
RUN apt-get update && apt-get install git nano -y
RUN git clone https://github.com/paolomandica/sapienza-video-contrastive.git

# Install requirements
RUN pip install --ignore-installed -r ./sapienza-video-contrastive/requirements.txt

# Clone and install evaluation repo
RUN git clone https://github.com/davisvideochallenge/davis2017-evaluation.git
RUN python ./davis2017-evaluation/setup.py install


ARG USER_ID
ARG GROUP_ID
ENV USERNAME=francolu

RUN addgroup --gid $GROUP_ID $USERNAME
RUN adduser --home /home/$USERNAME --disabled-password --gecos '' --uid $USER_ID --gid $GROUP_ID $USERNAME
USER $USERNAME

RUN pip install accelerator -y

# RUN dvc pull

# CMD ["python", "./sapienza-video-contrastive/code/train.py"]
