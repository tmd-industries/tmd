# Libraries required by RDkit
ARG LIBXRENDER_VERSION=1:0.9.10-*
ARG LIBXEXT_VERSION=2:1.3.4-*

FROM docker.io/nvidia/cuda:13.0.2-devel-ubuntu24.04 AS tmd_base_env
ARG LIBXRENDER_VERSION
ARG LIBXEXT_VERSION

# Copied out of anaconda's dockerfile
ARG MAKE_VERSION=4.3-*
ARG GIT_VERSION=1:2.43.0-*
ARG WGET_VERSION=1.21.4-*
RUN (apt-get update || true)  && apt-get install --no-install-recommends -y \
    wget=${WGET_VERSION} git=${GIT_VERSION} make=${MAKE_VERSION} libxrender1=${LIBXRENDER_VERSION} libxext-dev=${LIBXEXT_VERSION} \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

ARG MINIFORGE_NAME=Miniforge3
ARG MINIFORGE_VERSION=24.11.2-0
RUN wget --quiet https://github.com/conda-forge/miniforge/releases/download/${MINIFORGE_VERSION}/${MINIFORGE_NAME}-${MINIFORGE_VERSION}-Linux-$(uname -m).sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc

WORKDIR /opt/

# Setup CMake
ARG CMAKE_VERSION=3.24.3
RUN wget --quiet https://github.com/Kitware/CMake/releases/download/v${CMAKE_VERSION}/cmake-${CMAKE_VERSION}-linux-x86_64.tar.gz -O cmake.tar.gz && \
    tar -xzf cmake.tar.gz && \
    rm -rf cmake.tar.gz

ENV PATH=$PATH:/opt/cmake-${CMAKE_VERSION}-linux-x86_64/bin/

# Copy the environment yml to cache environment when possible
COPY environment.yml /code/tmd/

WORKDIR /code/tmd/

ARG ENV_NAME=tmd

# Create TMD Env
RUN . /opt/conda/etc/profile.d/conda.sh && \
    conda env create -n "${ENV_NAME}" -f environment.yml && \
    conda clean -a && \
    conda activate ${ENV_NAME}

ENV PATH=/opt/conda/envs/${ENV_NAME}/bin:$PATH

ENV CONDA_DEFAULT_ENV=${ENV_NAME}

WORKDIR /code/

# Copy the pip requirements to cache when possible
COPY requirements.txt /code/tmd/
RUN pip install --no-cache-dir -r tmd/requirements.txt


# Container that contains the cuda developer tools which allows building the customs ops
# Used as an intermediate for creating a final slimmed down container with tmd and only the cuda runtime
FROM tmd_base_env AS tmd_cuda_dev
ARG CUDA_ARCH
ENV CMAKE_ARGS="-DCUDA_ARCH:STRING=${CUDA_ARCH}"

COPY . /code/tmd/
WORKDIR /code/tmd/
RUN pip install --no-cache-dir -e . && rm -rf ./build
# Blow away the C++ code to avoid shipping source
RUN rm -rf /code/tmd/tmd/cpp/
# Blow away unnecessary Cuda Libraries. NOTE: cudart is redundant, but tiny
RUN find /usr/local/cuda/targets/x86_64-linux/lib/ -regextype posix-extended -type f,l ! -regex ".*(libcudart|libcurand)\.so\..*" -exec rm -r "{}" \;


# Container with only cuda base, half the size of the tmd_cuda_dev container
# Need to copy curand/cudart as these are dependencies of the TMD GPU code
FROM docker.io/nvidia/cuda:13.0.2-base-ubuntu24.04 AS tmd
ARG LIBXRENDER_VERSION
ARG LIBXEXT_VERSION
RUN (apt-get update || true) && apt-get install --no-install-recommends -y libxrender1=${LIBXRENDER_VERSION} libxext-dev=${LIBXEXT_VERSION} \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy cuda libraries from image, assumes tmd_cuda_dev only has the strictly neccessary libraries
COPY --from=tmd_cuda_dev /usr/local/cuda/targets/x86_64-linux/lib/ /usr/local/cuda/targets/x86_64-linux/lib/

COPY --from=tmd_cuda_dev /opt/conda/ /opt/conda/
COPY --from=tmd_cuda_dev /code/ /code/
COPY --from=tmd_cuda_dev /root/.bashrc /root/.bashrc
RUN ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh
ARG ENV_NAME=tmd
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64/
ENV PATH=/opt/conda/envs/${ENV_NAME}/bin:$PATH
ENV CONDA_DEFAULT_ENV=${ENV_NAME}
