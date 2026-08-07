FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime
 
WORKDIR /workspace
 
# System deps
RUN apt-get update && apt-get install -y \
    git libgomp1 \
    && rm -rf /var/lib/apt/lists/*
 
# Python deps — matches your pointmlp conda env
RUN pip install --no-cache-dir \
    onnx==1.15.0 \
    onnxruntime-gpu==1.17.0 \
    onnx2torch \
    scikit-learn \
    scipy \
    tqdm \
    h5py \
    addict \
    timm \
    torch-scatter -f https://data.pyg.org/whl/torch-2.1.0+cu121.html \
    pyyaml
 
# Install h3dnas in editable mode
COPY submodules/h3dnas /workspace/submodules/h3dnas
RUN pip install -e /workspace/submodules/h3dnas
 
# Copy rest of workspace
COPY . /workspace
 
CMD ["/bin/bash"]