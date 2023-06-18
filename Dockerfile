FROM continuumio/miniconda3
USER root
RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y
# Create the environment:
COPY environment.yml .
RUN conda env create -f environment.yml
# Make RUN commands use the new environment:
RUN echo "conda activate adaface" >> ~/.bashrc
SHELL ["/bin/bash", "--login", "-c"]
COPY requirements.txt .
RUN pip install -r requirements.txt
WORKDIR /adaface
ENTRYPOINT ["./entrypoint.sh"]

