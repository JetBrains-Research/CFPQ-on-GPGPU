FROM nvidia/cuda:10.0-devel-ubuntu18.04

RUN apt update && \
    apt upgrade -y && \
    apt install -y wget git cmake dh-autoreconf libomp-dev

# install git lfs
ARG git_lfs_url=https://github.com/git-lfs/git-lfs/releases/download/v2.7.1/git-lfs-linux-amd64-v2.7.1.tar.gz
RUN wget ${git_lfs_url} -O git-lfs.tar.gz -q && \
    tar -xzf git-lfs.tar.gz && \
    ./install.sh && \
    rm git-lfs.tar.gz install.sh git-lfs README.md CHANGELOG.md

# install m4ri library
RUN git clone https://voudy@bitbucket.org/vkutuev/m4ri.git && \
    cd m4ri && autoreconf --install && autoreconf --install && \
    ./configure --enable-openmp && make install

# install GTgraph library
RUN wget http://www.cse.psu.edu/~kxm85/software/GTgraph/GTgraph.tar.gz -O GTgraph.tar.gz -q && \
    tar -xzf GTgraph.tar.gz 
RUN cd GTgraph && \
    sed -i 's/CC = icc/#CC = icc/g' Makefile.var && \
    sed -i 's/#CC = gcc/CC = gcc/g' Makefile.var && \
    make 
RUN rm GTgraph.tar.gz

# install anaconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh -q && \
    chmod +x miniconda.sh && \
    ./miniconda.sh -b -p /usr/local/miniconda && \
    rm miniconda.sh
ENV PATH /usr/local/miniconda/bin:$PATH
RUN conda install numpy tqdm cudatoolkit numba scipy six

# install mono
RUN apt install gnupg ca-certificates && \
    apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv-keys 3FA7E0328081BFF6A14DA29AA6A19B38D3D831EF && \
    echo "deb https://download.mono-project.com/repo/ubuntu stable-bionic main" | tee /etc/apt/sources.list.d/mono-official-stable.list && \
    apt update
RUN apt install -y mono-devel

RUN mkdir work
WORKDIR /work

CMD ["python", "testing_system.py", "tests.csv", "init.py"]
