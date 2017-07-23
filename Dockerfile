FROM nvidia/cuda:8.0-cudnn5-runtime

RUN apt update && apt install -y python3-pip vim
RUN pip3 install tensorflow-gpu keras h5py
