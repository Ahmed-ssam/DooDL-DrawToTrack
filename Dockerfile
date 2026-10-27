# CPU image (classical baseline) by default. For SAM 2.1 on GPU:
#   docker build --build-arg BASE_IMAGE=pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime \
#                --build-arg INSTALL_SAM2=1 -t doodl-gpu .
#   docker run --gpus all -p 8000:8000 doodl-gpu
# Not built or tested in the development sandbox.
ARG BASE_IMAGE=python:3.12-slim
FROM ${BASE_IMAGE}
ARG INSTALL_SAM2=0
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg git && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip install --no-cache-dir . && \
    if [ "$INSTALL_SAM2" = "1" ]; then pip install --no-cache-dir "git+https://github.com/facebookresearch/sam2.git"; fi
ENV DOODL_DATA_DIR=/data
VOLUME /data
EXPOSE 8000
CMD ["doodl", "serve", "--host", "0.0.0.0", "--port", "8000"]
