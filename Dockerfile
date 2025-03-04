FROM ubuntu:22.04

SHELL ["/bin/bash", "-c"]

WORKDIR /tmp
RUN apt update
RUN apt-get install -y git build-essential zlib1g-dev libncurses5-dev libgdbm-dev libnss3-dev libssl-dev libreadline-dev libffi-dev libsqlite3-dev wget libbz2-dev liblzma-dev

# Install python 3.10.12
RUN wget https://www.python.org/ftp/python/3.10.12/Python-3.10.12.tgz
RUN tar -xf Python-3.10.12.tgz
WORKDIR /tmp/Python-3.10.12
RUN ./configure --enable-optimizations
RUN make -j 12
RUN make altinstall
RUN ln -s /usr/local/bin/python3.10 /usr/local/bin/python
RUN ln -s /usr/local/bin/pip3.10 /usr/local/bin/pip

# Import requirement file first and install required packages
COPY python/requirements.txt .
COPY python/itl/memory/requirements.txt itl/memory/
COPY python/itl/vision/requirements.txt itl/vision/
COPY python/itl/lang/requirements.txt itl/lang/
COPY python/itl/symbolic_reasoning/requirements.txt itl/symbolic_reasoning/
COPY python/itl/action_planning/requirements.txt itl/action_planning/
COPY python/itl/lpmln/requirements.txt itl/lpmln/
RUN pip install --no-cache-dir -r requirements.txt

# # Needed for adding new PPAs
# RUN apt install -y software-properties-common
# RUN apt install -y python3-launchpadlib
# # Add this mesa PPA to ensure vulkan recognizes the GPU
# RUN add-apt-repository -y ppa:kisak/kisak-mesa
RUN apt update && apt upgrade -y

# Some graphics related libraries needed
RUN apt install -y libgl1 libglib2.0-0
RUN apt install -y libvulkan1 mesa-vulkan-drivers vulkan-tools
RUN apt install -y mesa-utils

# Let's add in vim, less and rsync
RUN apt update
RUN apt install -y vim
RUN apt install -y less
RUN apt install -y rsync

# Install virtual display buffer X server and X11 server utils
RUN apt install -y xvfb
RUN apt install -y x11-xserver-utils

# Create a non-root user
RUN useradd --create-home nonroot

# Import the repository content
RUN mkdir -p /home/nonroot/semantic-assembler
RUN mkdir -p /mnt/data_volume
WORKDIR /home/nonroot/semantic-assembler
COPY . .
RUN chown -R nonroot /home/nonroot
RUN chown -R nonroot /mnt/data_volume

# Login as non-root
USER nonroot

# Environment variables
ENV DISPLAY=:99
ENV NVIDIA_DRIVER_CAPABILITIES=all

ENTRYPOINT ["/bin/bash", "tools/container_internal_scripts/eval_with_xvfb.sh"]
