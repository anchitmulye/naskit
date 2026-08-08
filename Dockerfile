FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive
ENV MKL_THREADING_LAYER=GNU

WORKDIR /workspace

# System deps
RUN apt-get update && apt-get install -y \
    git \
    python3.12 \
    python3-pip \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy rest of workspace
COPY . /workspace

CMD ["/bin/bash"]