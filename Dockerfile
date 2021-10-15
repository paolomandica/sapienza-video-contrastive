FROM pytorch/pytorch:1.8.1-cuda11.1-cudnn8-devel

# Set permission
ARG USER_ID=1038
ARG GROUP_ID=1040
ENV USERNAME=francolu

RUN addgroup --gid $GROUP_ID $USERNAME
RUN adduser --home /home/$USERNAME --disabled-password --gecos '' --uid $USER_ID --gid $GROUP_ID $USERNAME

# Set the working directory and user
WORKDIR /home/$USERNAME

# Clone the repo
RUN apt-get update && apt-get install git nano tree screen wget -y
RUN apt-get install ffmpeg libsm6 libxext6  -y

# Clone and install evaluation repo
RUN git clone https://github.com/davisvideochallenge/davis2017-evaluation.git
RUN python ./davis2017-evaluation/setup.py install

# Install requirements
# RUN pip install --ignore-installed -r https://raw.githubusercontent.com/paolomandica/sapienza-video-contrastive/main/requirements.txt

USER $USERNAME