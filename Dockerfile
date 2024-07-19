FROM pytorch/pytorch

# Create user account
ARG USER=ai
ARG UID=1000
ARG PASS=pass
RUN apt update && apt install -y ssh zsh sudo && sed -i 's/^#X11UseLocalhost yes/X11UseLocalhost no/' /etc/ssh/sshd_config && \
    useradd -u $UID -ms /bin/bash ${USER} --groups sudo,video && \
    echo "$USER:$PASS" | chpasswd
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
RUN conda install -n base pandas numpy
USER ${USER}
WORKDIR /home/${USER}
