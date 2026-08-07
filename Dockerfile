FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive
ENV MKL_THREADING_LAYER=GNU
 
WORKDIR /workspace
 
# System deps
RUN apt-get update && apt-get install -y \
    git \
    python3.12 \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*
 
# Install h3dnas in editable mode
COPY submodules/h3dnas /workspace/submodules/h3dnas
RUN pip install -e /workspace/submodules/h3dnas
 
# Copy rest of workspace
COPY . /workspace
 
CMD ["/bin/bash"]